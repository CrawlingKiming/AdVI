# Path
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# For calculating HPD interval
from model.subclass.HPD import hpd_grid

import arviz as az

def q_return(samples):
    samples_mean = np.mean(samples, axis=0)
    samples_min = np.percentile(samples, 2.5, axis=0)
    samples_max = np.percentile(samples, 97.5, axis=0)
    #print(samples_min, samples_max)
    return samples_min, samples_mean, samples_max

def hpd_return(samples):
    hpd_min = []
    #hpd_mean = []
    hpd_max = []
    N, T = samples.shape
    for t in range(T):
        temp_hpd = az.hdi(samples[:,t],hdi_prob=0.95)
        hpd_min_e, hpd_max_e = temp_hpd
        #hpd_mean_e = temp_modes[0]
        hpd_min.append(hpd_min_e)
        hpd_max.append(hpd_max_e)
        #hpd_mean.append(hpd_mean_e)
    hpd_min = np.asarray(hpd_min)
    hpd_max = np.asarray(hpd_max)
    hpd_mean = np.percentile(samples, 50, axis=0)
    return hpd_min, hpd_mean, hpd_max


class Experiment_Writing():
    no_log_keys = ['project', 'name',
                    'eval_every']

    def __init__(self, log_path):
        self.log_path = log_path

    def loss_store(self):
        raise NotImplementedError

    def forward_img_store(self, samples, truex, data_ground, burden_estimate, model_num):
        #ylim_upper = np.load("../data/Extra/forplot.npy")
        samples, cases_prob, new_r = samples
        B = samples.shape[0]

        res = []

        nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=new_r, probs=cases_prob)
        n = 40
        data_samples = nb.sample((n,))  # n * B * 118 * 6
        data_samples = data_samples.view((n * 500, 118, 6))

        data_ground = np.sum(data_ground, axis=2)
        samples_min, samples_mean, samples_max = hpd_return(np.sum(data_samples.detach().cpu().numpy(), axis=2))#
        #print(samples_mean.shape)
        #samples_min, samples_mean, samples_max = q_return(np.sum(data_samples.detach().cpu().numpy(), axis=2))
        #raise ValueError
        #print(samples_mean.shape)
        cover_num = 0
        for ti in range(118):
            if ((data_ground[0][ti] < samples_max[ti]) and (data_ground[0][ti] > samples_min[ti])):
                cover_num += 1
        res.append(cover_num)

        fig, ax = plt.subplots()
        ax.plot(samples_mean[:], color="red")
        ax.plot(samples_min[:], "--", color="red")
        ax.plot(samples_max[:], "--", color="red")

        ax.plot(data_ground[0][:], "bo", markersize=2)
        ax.set_xlabel('Weeks', fontsize=22)
        ax.set_ylabel('Severe cases (All)', fontsize=22)
        #ax.set_ylim([0, ylim_upper[model_num]])
        ax.tick_params(axis='both', which='major', labelsize=18)
        #ax.tick_params(axis='both', which='minor', labelsize=17)
        # plt.show()
        plt.savefig(os.path.join(self.log_path, "png/Forwardsamples_model_num{}.png".format(model_num)),
                    bbox_inches='tight')
        plt.close(fig=fig)

        groundtruth_sum = np.sum(burden_estimate, axis=2)
        samples_sum = np.sum(samples, axis=2)
        #samples_sum_min, samples_sum_mean, samples_sum_max = q_return(samples_sum)
        samples_sum_min, samples_sum_mean, samples_sum_max = hpd_return(samples_sum)

        #print(groundtruth_sum[0])
        #raise ValueError
        cover_num = 0
        for ti in range(samples_sum_max.shape[0]):
            if ((groundtruth_sum[0][ti] <= samples_sum_max[ti]) and (groundtruth_sum[0][ti] >= samples_sum_min[ti])):
                cover_num += 1
        print(cover_num)
        res.append(cover_num)
        ylim = [[0.0, 140]]
        fig, ax = plt.subplots()

        ax.plot(samples_sum_mean[:], color="red")
        ax.plot(samples_sum_min[:], "--", color="red")
        ax.plot(samples_sum_max[:], "--", color="red")
        ax.plot(groundtruth_sum[0][:], color="green")
        ax.set_xlabel('Time (weeks)', fontsize=22)
        ax.set_ylabel('Disease Burden (All)', fontsize=22)
        ax.set_ylim(ylim[-1])
        ax.tick_params(axis='both', which='major', labelsize=18)
        #ax.tick_params(axis='both', which='minor', labelsize=17)

        plt.savefig(os.path.join(self.log_path, "png/BurdenEstimate_model_num{}.png".format(model_num)),
                    bbox_inches='tight')
        plt.close(fig=fig)
        return res

    def param_img_store(self, params, model_num):
        temp = params
        # print(yset.shape)
        # print(temp.shape)
        #np.save(
        #    os.path.join("./results/MSIR/gensamples", "generated_sampled_epoch{}_interval{}.png".format(epoch, i)),
        #    temp)

        true_beta = [1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381]

        fig, axs = plt.subplots(3, 4, constrained_layout=True)
        # fig.suptitle('MSIR function')
        alpha = 1.0
        axs[0, 0].hist(temp[:, 0], bins=128, color='gray', alpha=alpha)
        axs[0, 0].set_title(r'$w$')
        axs[0, 0].set_xlim([0.2, 0.7])
        axs[0, 0].axvline(x=0.43, color="r")
        axs[0, 1].hist(temp[:, 1], bins=128, color='gray', alpha=alpha)
        axs[0, 1].set_title(r"$\phi$")
        axs[0, 1].set_xlim([6.0, 8.8])
        axs[0, 1].axvline(x=7.35, color="r")
        axs[0, 2].hist(temp[:, 2], bins=128, color='gray', alpha=alpha)
        axs[0, 2].set_title(r'$\rho$')
        axs[0, 2].set_xlim([0.0, 0.05])
        axs[0, 2].axvline(x=0.027, color="r")
        axs[0, 3].set_title(r'$\nu$')
        axs[0, 3].set_xlim([0.5, 1.0])
        axs[0, 3].axvline(x=0.9, color="r")
        axs[0, 3].hist(temp[:, 3], bins=128,color='gray', alpha=alpha)
        axs[1, 0].set_title(r'$\beta_{01}$')
        axs[1, 0].set_xlim([0.0, 3.0])
        axs[1, 0].axvline(x=true_beta[0], color="r")
        axs[1, 0].hist(temp[:, 4], bins=128, color='gray', alpha=alpha)
        axs[1, 1].set_title(r'$\beta_{02}$')
        axs[1, 1].set_xlim([0.5, 4.0])
        axs[1, 1].axvline(x=true_beta[1], color="r")
        axs[1, 1].hist(temp[:, 5], bins=128, color='gray', alpha=alpha)
        axs[1, 2].set_title(r'$\beta_{03}$')
        axs[1, 2].set_xlim([0.0, 2.0])
        axs[1, 2].axvline(x=true_beta[2], color="r")
        axs[1, 2].hist(temp[:, 6], bins=128, color='gray', alpha=alpha)
        axs[1, 3].set_title(r'$\beta_{04}$')
        axs[1, 3].set_xlim([1.5, 4.0])
        axs[1, 3].axvline(x=true_beta[3], color="r")
        axs[1, 3].hist(temp[:, 7], bins=128, color='gray', alpha=alpha)
        axs[2, 0].set_title(r'$\beta_{05}$')
        axs[2, 0].set_xlim([0.0, 2.5])
        axs[2, 0].axvline(x=true_beta[4], color="r")
        axs[2, 0].hist(temp[:, 8], bins=128,color='gray', alpha=alpha)
        axs[2, 1].set_title(r'$\beta_{06}$')
        axs[2, 1].set_xlim([0.0, 1.3])
        axs[2, 1].axvline(x=true_beta[5], color="r")
        axs[2, 1].hist(temp[:, 9], bins=128, color='gray', alpha=alpha)

        fig.delaxes(axs[2, 2])
        fig.delaxes(axs[2, 3])

        plt.savefig(os.path.join(self.log_path, "png/Paramsamples_model_num{}.png".format(model_num)),
                    bbox_inches='tight')
        plt.close(fig=fig)

    def param_ls_img_store(self, params, model_num, itr):
        raise NotImplementedError

class SEIR_plotting():


    def __init__(self, log_path):
        self.log_path = log_path#log_base

        #print("Saving log : {}".format(log_base))

    def forward_img_store(self, samples, truex, extra):

        I_w, age_class = samples # B * 54, B * 3
        B = I_w.shape[0]

        """
        nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=torch.tensor(new_r),
                                                                    logits=torch.log(torch.tensor(for_logit)))
        n = 40
        data_samples = nb.sample((n,))  # n * B * 118 * 6


        data_samples = data_samples.view((n * B, 53))
        
        #print(data_samples.shape, B)
        I_w = data_samples.detach().cpu().numpy()
        """

        fig, ax = plt.subplots(1, 2)
        #samples_min, samples_mean, samples_max = q_return(np.sum(data_samples.detach().cpu().numpy(), axis=0))
        samples_mean = np.mean(I_w, axis=0)
        samples_min = np.percentile(I_w, 97.5, axis=0)
        samples_max = np.percentile(I_w, 2.5, axis=0)

        age_class_min = np.percentile(age_class, 97.5, axis=0)
        age_class_max = np.percentile(age_class, 2.5, axis=0)
        age_class_mean = np.mean(age_class, axis=0)

        #np.save("ABC_mean_samples", samples_mean)
        ax[0].plot(samples_mean[:], color="red")
        ax[0].plot(samples_min[:], "--", color="red")
        ax[0].plot(samples_max[:], "--", color="red")
        # ax[pp].plot(cases_prob[0][:,pp].detach().cpu().numpy(), color = "red")
        ax[0].plot(truex, "ko", markersize=2)
        ax[0].xaxis.set_ticks([0, 20, 40], fontdict={'fontsize':20})
        ax[0].set_xlabel('Time (weeks)', fontsize=20)
        ax[0].set_ylim([0.0, 14000])
        ax[0].set_ylabel('Number', fontsize=20)
        #ax[0].tick_params(axis='both', which='major', labelsize=19)

        age_class_x = np.asarray([1,2,3])
        age_true = np.asarray([0.42, 0.30, 0.28])

        label = ["0.5-5", "5-15", "16+"]
        ax[1].bar(age_class_x - 0.4, age_true, width=0.4, color='k', align='center', label="Observed")
        ax[1].bar(age_class_x, age_class_mean, width=0.4, color='r', align='center', label="Simulated")
        ax[1].xaxis.set_ticks(age_class_x - 0.2)
        ax[1].set_xticklabels(label, fontdict={'fontsize':13})
        ax[1].set_ylim([0.0, 0.5])
        ax[1].set_xlabel('Age groups (years)', fontsize=16)
        ax[1].legend()
        plt.savefig(os.path.join(self.log_path, "./res_img/Forwardsamples_model_{}.png".format(extra)),
                    bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots()

        samples_mean = np.mean(I_w, axis=0)
        samples_min = np.percentile(I_w, 97.5, axis=0)
        samples_max = np.percentile(I_w, 2.5, axis=0)

        age_class_min = np.percentile(age_class, 97.5, axis=0)
        age_class_max = np.percentile(age_class, 2.5, axis=0)
        age_class_mean = np.mean(age_class, axis=0)
        #print(age_class_mean)

        ax.plot(samples_mean[:], color="red")
        ax.plot(samples_min[:], "--", color="red")
        ax.plot(samples_max[:], "--", color="red")
        # ax[pp].plot(cases_prob[0][:,pp].detach().cpu().numpy(), color = "red")
        ax.plot(truex, "ko", markersize=2)
        ax.xaxis.set_ticks([0, 20, 40], fontdict={'fontsize':20})
        ax.set_xlabel('Time (weeks)', fontsize=20)
        ax.set_ylim([0.0, 14000])
        ax.set_ylabel('Number', fontsize=20)


        plt.savefig(os.path.join(self.log_path, "./res_img/Only_Forwardsamples_model_{}.png".format(extra)),
                    bbox_inches='tight')
        plt.close()
        """
        cover_num = 0
        for ti in range(truex.shape[0]):
            #print(truex[ti],samples_max[ti])
            if (truex[ti] <= samples_min[ti]) and (truex[ti] >= samples_max[ti]):
                cover_num += 1
        print(cover_num/53)
        """
        print("I_w MSE: {}".format(np.mean(np.square(samples_mean - truex))))
        np.save("./assets/BFAF_mean_samples", samples_mean)
        #t = I_w - samples_mean #[:, np.newaxis]
        #plt.hist(t[:,25])
        #plt.show()
        #self.writer.add_image('forward_{}'.format(model_num), fig, itr)

    def param_img_store(self, params, model_num, itr):
        raise NotImplementedError

    def param_ls_img_store(self, params, extra=1):
        #np.save("ABC_param_samples", params)
        np.save("./assets/BFAF_param_samples", params)
        #print(len(params))
        for p_idx in range(len(params)):
            sample_list = []
            #print(p_idx)
            temp = params[p_idx]
            #temp = temp.cpu().detach().numpy()
            sample_list.append(temp)

            # fig = plt.figure()
            temp = np.asarray(sample_list)
            temp = temp[0]

            fig, axs = plt.subplots(2, 3, constrained_layout=True)

            axs[0, 0].hist(temp[:, 0], bins=128, edgecolor = 'black', color='gray', alpha=0.3)
            axs[0, 0].set_title(r'$\beta$', size=23)
            axs[0, 0].tick_params(axis='x', which='major', labelsize=13)
            axs[0, 0].set_xlim([0.5 * 1e-6, 3 * 1e-6])
            axs[0, 1].hist(temp[:, 1],  bins=128, edgecolor = 'black', color='gray', alpha=0.3)
            axs[0, 1].set_title(r'$N_{0}$', size=23)
            axs[0, 1].tick_params(axis='x', which='major', labelsize=12)
            axs[0, 1].set_xlim([140*1e3, 300 * 1e3]) #130
            axs[0, 2].hist(temp[:, 2],  bins=128, edgecolor = 'black', color='gray', alpha=0.3)
            axs[0, 2].set_title(r'$f_{e}$', size=23)
            axs[0, 2].tick_params(axis='x', which='major', labelsize=12)
            axs[0, 2].set_xlim([0.0, 0.001])#0.0005
            axs[1, 0].set_title(r'$a_{sh}$', size=23)
            axs[1, 0].set_xlim([0.8, 1.2])
            axs[1, 0].hist(temp[:, 3], bins=128, edgecolor = 'black', color='gray', alpha=0.3)
            axs[1, 0].tick_params(axis='x', which='major', labelsize=13)
            axs[1, 1].set_title(r'$a_{rt}$', size=23)
            axs[1, 1].set_xlim([0.09, 0.11])#0.09
            axs[1, 1].hist(temp[:, 4],  bins=128, edgecolor = 'black', color='gray', alpha=0.3)
            axs[1, 1].tick_params(axis='x', which='major', labelsize=13)
            fig.delaxes(axs[1,2])

            plt.savefig(os.path.join(os.path.join(self.log_path, "./res_img/param_samples_model_{}.png".format(extra))))
            # plt.close(fig = fig)
            plt.close(fig=fig)

            hpd_mu = []
            modes_mu = []
            print(temp[:,0].shape)
            temp = temp * 1e6
            #temp[:, 1] = temp[:, 1] * 1e3
            for j in range(5):
                #j =2
                if j == 0 :
                    temp_hpd, _, _, temp_modes = hpd_grid(sample=temp[:, j], alpha=0.05)
                    #continue
                else :
                    temp_hpd, _, _, temp_modes = hpd_grid(sample=temp[:, j], alpha=0.05)
                a, b = temp_hpd[0]
                c = temp_modes[0]
                temp_hpd = (a/1e6, b/1e6)
                hpd_mu.append(temp_hpd)
                modes_mu.append(c/1e6)

            print(hpd_mu, modes_mu)

class SIR_plotting():
    no_log_keys = ['project', 'name',
                    'eval_every']

    def __init__(self, log_path):
        self.log_path = log_path

    def forward_img_store(self, samples, truex, groundtruth, model_num):
        truex = truex.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        A, B, C =samples.shape

        samples = samples[:, :18, :] + np.random.randn(A, B, C) * 0.1
        truex= truex[:18, :]
        groundtruth = groundtruth[:18, :]

        samples_min, samples_mean, samples_max = hpd_return(samples[:, :, -2])#

        cover_num = 0
        for ti in range(1, truex.shape[0]):

            if (truex[ti, -2] <= samples_max[ti]) and (truex[ti, -2] >= samples_min[ti]):
                cover_num += 1

        fig, ax = plt.subplots()

        ax.plot(samples_mean, color="gray")
        ax.plot(samples_min, "--", color="gray")
        ax.plot(samples_max, "--", color="gray")
        ax.plot(truex[:, -2], "rx", markersize=8, label="Infected")

        samples_min, samples_mean, samples_max = hpd_return(samples[:, :, -1])#
        for ti in range(1, truex.shape[0]):
            if (truex[ti, -1] <= samples_max[ti]) and (truex[ti, -1] >= samples_min[ti]):
                cover_num += 1

        ax.plot(samples_mean, color="gray")
        ax.plot(samples_min, "--", color="gray")
        ax.plot(samples_max, "--", color="gray")
        ax.plot(truex[:, -1], "go", markersize=8, label="recovered")

        x = np.arange(0, 17, 1)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(x+1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim([0.0, 115.0])

        ax.set_xlabel('Time (days)', fontsize=22)
        ax.set_ylabel('Number', fontsize=22)
        ax.legend(loc='upper left', fontsize='x-large')


        plt.savefig(os.path.join(self.log_path, "./res_img/Forwardsamples_model_{}.png".format(model_num)),
                    bbox_inches='tight')
        plt.close(fig=fig)


        return (cover_num / 34)

    def param_img_store(self, params, model_num):
        hpd_l = []
        hpd_r = []
        modes_mu = []
        mse = []
        coverage = []

        ans = [1.5, 0.5, 99]


        sample_list = []
        #temp = params[-1]
        temp = params
        #temp = temp.cpu().detach().numpy()
        sample_list.append(temp)

        # fig = plt.figure()
        temp = np.asarray(sample_list)
        temp = temp[0]


        fig, axs = plt.subplots(1, 3)

        axs[0].hist(temp[:, 1], bins=32,color='gray')
        axs[0].set_title(r"$\beta$")
        axs[0].set_xlim([1.0, 2.0])
        axs[0].axvline(x=1.5, color="r")
        axs[1].hist(temp[:, 2], bins=32,color='gray')
        axs[1].set_title(r"$\gamma$")
        axs[1].set_xlim([0.2, 0.8])
        axs[1].axvline(x=0.5, color="r")
        axs[2].hist(temp[:, 0], bins=32,color='gray')
        axs[2].set_title("Initial Susceptible")
        axs[2].axvline(x=99, color="r")
        axs[2].set_xlim([80.0, 120.0])
        temp[:, [1, 2]] = temp[:, [2, 1]]
        temp[:, [0, 2]] = temp[:, [2, 0]]
        #print(temp[:, 0])

        for j in range(3):
            param_cov = 0
            temp_hpd, _, _, temp_modes = hpd_grid(sample=temp[:, j], alpha=0.05)
            a, b = temp_hpd[0]
            c = temp_modes[0]
            tm = np.sqrt(np.mean(np.square(np.mean(temp[:, j]) - ans[j])))

            mse.append(tm)
            hpd_l.append(a)
            hpd_r.append(b)
            modes_mu.append(c)

            if a <= ans[j] and b >= ans[j]:
                param_cov +=1
            coverage.append(param_cov)

        fig.set_size_inches(20.5, 8.5)

        plt.savefig(os.path.join(os.path.join(self.log_path, "./res_img/param_samples_{}.png".format(model_num))))
        plt.close(fig=fig)

        return hpd_l, hpd_r, modes_mu, mse, coverage

    def param_ls_img_store(self, params, model_num, itr):

        raise NotImplementedError

