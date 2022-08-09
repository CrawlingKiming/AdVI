# Path
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
# Logging frameworks
from torch.utils.tensorboard import SummaryWriter

from utils.exp_utils import get_args_table, clean_dict
import torch

# Experiment
# For calculating HPD interval

def add_exp_args(parser):

    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--resume', type=eval, default=False)
    parser.add_argument('--iteration', type=int, default=300)
    parser.add_argument('--model_num', type=int, default=10)
    #parser.add_argument('--data_num', type=int, default=0)

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=15)
    parser.add_argument('--check_every', type=int, default=25)
    parser.add_argument('--log_tb', type=eval, default=False)
    parser.add_argument('--log_wandb', type=eval, default=False)

class Experiment_Writing():
    no_log_keys = ['project', 'name',
                    'eval_every']

    def __init__(self, args,
                 data_id, model_id):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        log_base = os.path.join(dir_path, 'results/MSIR/log')
        self.log_base = log_base
        print("Saving log : {}".format(log_base))
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        log_path = os.path.join(self.log_base, data_id, model_id, args.name)
        self.check_every = args.eval_every

        self.log_path = log_path
        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id

        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
#        if args.log_tb:
        self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
        self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)

    def save_args(self, args):

        # Save args
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(os.path.join(self.log_path,'args_table.txt'), "w") as f:
            f.write(str(args_table))

    def create_folders(self):

        # Create log folder
        os.makedirs(self.log_path)
        #print("Storing logs in:", self.log_path)

    def loss_store(self, loss, nll, itr,time, model_num):
        loss = loss.detach().cpu().numpy()
        nll = nll.detach().cpu().numpy()
        nll = np.mean(nll)
        #print(loss,nll, time)
        self.writer.add_scalar('Loss/train/ELBO_{}'.format(model_num), loss, itr)
        self.writer.add_scalar('Loss/train/nll_{}'.format(model_num), nll, itr)
        self.writer.add_scalar('Loss/train/time_{}'.format(model_num), time, 0)
        print(
            'Iter: {}/{},loss:{}, NLL:{}, Time :{}'.format(itr + 1, 400, loss, nll, time), end='\r')

    def forward_img_store(self, samples, truex, groundtruth, model_num, itr):

        # samples B * 118 * 3
        truex = truex.detach().cpu().numpy()
        fig, ax = plt.subplots(4, 1)
        samples_mean = np.mean(samples, axis=0)
        samples_min = np.percentile(samples, 97.5, axis=0)
        samples_max = np.percentile(samples, 2.5, axis=0)

        #print(truex.shape )
        samples_sum = np.sum(samples, axis=2)
        #print(samples_sum.shape)
        samples_sum_mean = np.mean(samples_sum, axis=0)
        samples_sum_min = np.percentile(samples_sum, 97.5, axis=0)
        samples_sum_max = np.percentile(samples_sum, 2.5, axis=0)
        truex_sum = np.sum(truex, axis=2)
        groundtruth_sum = np.sum(groundtruth, axis=2)
        #samples_min = np.min(samples, axis=0)
        #samples_max = np.max(samples, axis=0)
        ylim = [[0.0, 25], [0.0, 60], [0.0, 130], [0.0, 190]]
        for pp in range(3):
            ax[pp].plot(samples_mean[:, pp], color="red")
            ax[pp].plot(samples_min[:, pp], "--", color="red")
            ax[pp].plot(samples_max[:, pp], "--", color="red")
            # ax[pp].plot(cases_prob[0][:,pp].detach().cpu().numpy(), color = "red")
            ax[pp].plot(truex[0][:, pp], "bo", markersize=2)
            ax[pp].plot(groundtruth[0][:, pp], color="green")
            ax[pp].set_ylim(ylim[pp])
        pp = 3
        ax[pp].plot(samples_sum_mean[:], color="red")
        ax[pp].plot(samples_sum_min[:], "--", color="red")
        ax[pp].plot(samples_sum_max[:], "--", color="red")
        # ax[pp].plot(cases_prob[0][:,pp].detach().cpu().numpy(), color = "red")
        ax[pp].plot(truex_sum[0][:], "bo", markersize=2)
        ax[pp].plot(groundtruth_sum[0][:], color="green")
        ax[pp].set_ylim(ylim[pp])

        cover_num = 0
        for ti in range(truex_sum.shape[0]):
            if (truex_sum[0][ti] <= samples_sum_max[ti]) and (truex_sum[0][ti] >= samples_sum_min[ti]):
                cover_num += 1



        self.writer.add_scalar('Loss/train/coverage_{}'.format(model_num), 100 * cover_num / truex_sum.shape[0], itr)

        plt.savefig(os.path.join(self.log_path, "Forwardsamples_model_num{}_interval{}.png".format(model_num, itr)),
                    bbox_inches='tight')
        plt.close(fig=fig)

        #self.writer.add_image('forward_{}'.format(model_num), fig, itr)

    def param_img_store(self, params, model_num, itr):
        sample_list = []
        temp = params[-1]
        temp = temp.cpu().detach().numpy()
        sample_list.append(temp)

        # fig = plt.figure()
        temp = np.asarray(sample_list)

        temp = temp[0]
        # print(yset.shape)
        # print(temp.shape)
        #np.save(
        #    os.path.join("./results/MSIR/gensamples", "generated_sampled_epoch{}_interval{}.png".format(epoch, i)),
        #    temp)

        true_beta = [1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381]

        fig, axs = plt.subplots(3, 4, constrained_layout=True)
        fig.suptitle('MSIR function')

        axs[0, 0].hist(temp[:, 0], bins=128)
        axs[0, 0].set_title("W")
        axs[0, 0].set_xlim([0.0, 1.0])
        axs[0, 0].axvline(x=0.43, color="r")
        axs[0, 1].hist(temp[:, 1], bins=128)
        axs[0, 1].set_title("Phi")
        axs[0, 1].set_xlim([2.0, 8.8])
        axs[0, 1].axvline(x=7.35, color="r")
        axs[0, 2].hist(temp[:, 2], bins=128)
        axs[0, 2].set_title("Rho")
        axs[0, 2].set_xlim([0.0, 0.05])
        axs[0, 2].axvline(x=0.027, color="r")
        axs[0, 3].set_title("Nu")
        axs[0, 3].set_xlim([1.0, 2.0])
        axs[0, 3].axvline(x=1.5, color="r")
        axs[0, 3].hist(temp[:, 3], bins=128)

        for j in range(6):
            axs[j // 4 + 1, j % 4].hist(temp[:, j + 4], bins=128)
            axs[j // 4 + 1, j % 4].set_xlim([0.0, 4.0])
            axs[j // 4 + 1, j % 4].set_title("b{}".format(j + 1))
            axs[j // 4 + 1, j % 4].axvline(x=true_beta[j], color="r")

        """
        hpd_mu = []
        modes_mu = []
        for j in range(3):
            temp_hpd, _ , _, temp_modes = hpd_grid(sample=temp[:,j], alpha=0.05)
            hpd_mu.append(temp_hpd)
            modes_mu.append(temp_modes)
        print(hpd_mu)
        print(modes_mu)
        print("")
        """

        plt.savefig(os.path.join(os.path.join(self.log_path, "param_samples_model_num{}_interval{}.png".format(model_num, itr))))
        # plt.close(fig = fig)
        plt.close(fig=fig)
        plt.clf()
        #self.writer.add_image('params_{}'.format(model_num), fig, itr)

    def param_ls_img_store(self, params, model_num, itr, qq = 0):


        for p_idx in range(len(params)):
            sample_list = []
            temp = params[p_idx]
            temp = temp.cpu().detach().numpy()
            sample_list.append(temp)

            # fig = plt.figure()
            temp = np.asarray(sample_list)

            temp = temp[0]
            # print(yset.shape)
            # print(temp.shape)
            #np.save(
            #    os.path.join("./results/MSIR/gensamples", "generated_sampled_epoch{}_interval{}.png".format(epoch, i)),
            #    temp)

            print(np.mean(temp, axis=0))
            true_beta = [1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381]

            fig, axs = plt.subplots(3, 4, constrained_layout=True)
            #fig.suptitle('MSIR function')

            axs[0, 0].hist(temp[:, 0], bins=128)
            axs[0, 0].set_title(r'$w$')
            axs[0, 0].set_xlim([0.2, 0.7])
            axs[0, 0].axvline(x=0.43, color="r")
            axs[0, 1].hist(temp[:, 1], bins=128)
            axs[0, 1].set_title(r"$\phi$")
            axs[0, 1].set_xlim([6.0, 8.8])
            axs[0, 1].axvline(x=7.35, color="r")
            axs[0, 2].hist(temp[:, 2], bins=128)
            axs[0, 2].set_title(r'$\rho$')
            axs[0, 2].set_xlim([0.0, 0.05])
            axs[0, 2].axvline(x=0.027, color="r")
            axs[0, 3].set_title(r'$\nu$')
            axs[0, 3].set_xlim([0.5, 1.5])
            axs[0, 3].axvline(x=0.9, color="r")
            axs[0, 3].hist(temp[:, 3], bins=128)

            for j in range(6):
                axs[j // 4 + 1, j % 4].hist(temp[:, j + 4], bins=128)
                axs[j // 4 + 1, j % 4].set_xlim([0.0, 4.0])
                axs[j // 4 + 1, j % 4].set_title("b{}".format(j + 1))
                axs[j // 4 + 1, j % 4].axvline(x=true_beta[j], color="r")

            fig.delaxes(axs[2, 2])
            fig.delaxes(axs[2, 3])
            """
            if p_idx == len(params)-1 :
                hpd_mu = []
                modes_mu = []
                for j in range(4):
                    temp_hpd, _, _, temp_modes = hpd_grid(sample=temp[:, j], alpha=0.05)
                    hpd_mu.append(temp_hpd)
                    modes_mu.append(temp_modes)
                print(hpd_mu)
                print(modes_mu)
                print("")
            """

            plt.savefig(os.path.join(os.path.join(self.log_path, "param_samples_model_num{}_interval{}_Ladder{}{}.png".format(model_num, itr, p_idx,qq))))
            # plt.close(fig = fig)
            plt.close(fig=fig)


class SIR_Experiment_Writing():
    no_log_keys = ['project', 'name',
                    'eval_every']

    def __init__(self, args,
                 data_id, model_id):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        log_base = os.path.join(dir_path, 'results/SIR/log')
        self.log_base = log_base
        print("Saving log : {}".format(log_base))
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        log_path = os.path.join(self.log_base, data_id, model_id, args.name)
        self.check_every = args.eval_every

        self.log_path = log_path
        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id

        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
#        if args.log_tb:
        self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
        self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)

    def save_args(self, args):

        # Save args
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(os.path.join(self.log_path,'args_table.txt'), "w") as f:
            f.write(str(args_table))

    def create_folders(self):

        # Create log folder
        os.makedirs(self.log_path)
        #print("Storing logs in:", self.log_path)

    def loss_store(self, loss, nll, itr,time, model_num):
        loss = loss.detach().cpu().numpy()
        nll = nll.detach().cpu().numpy()
        nll = np.mean(nll)
        #print(loss,nll, time)
        self.writer.add_scalar('Loss/train/ELBO_{}'.format(model_num), loss, itr)
        self.writer.add_scalar('Loss/train/nll_{}'.format(model_num), nll, itr)
        self.writer.add_scalar('Loss/train/time_{}'.format(model_num), time, 0)
        print(
            'Iter: {}/{},loss:{}, NLL:{}, Time :{}'.format(itr + 1, 400, loss, nll, time), end='\r')

    def forward_img_store(self, samples, truex, groundtruth, model_num, itr):
        truex = truex.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        #print(samples.shape)
        #print(truex.shape)
        samples = samples[:, :17, :]
        truex= truex[:17, :]
        groundtruth = groundtruth[:17, :]


        samples_mean = np.mean(samples, axis=0)
        samples_min = np.percentile(samples, 2.5, axis=0)
        samples_max = np.percentile(samples, 97.5, axis=0)

        fig, ax = plt.subplots()
        #ax[0].plot(samples_mean[:, 0], color="red")
        #ax[0].plot(samples_min[:, 0], "--", color="red")
        #ax[0].plot(samples_max[:, 0 ], "--", color="red")

        #ax.plot(samples_mean[:, 1], color="green")
        ax.plot(groundtruth[:, 1], color="green")
        ax.plot(samples_min[:, 1], "--", color="green")
        ax.plot(samples_max[:, 1], "--", color="green")
        ax.plot(truex[:, 1], "go", markersize=2)

        #ax.plot(samples_mean[:, 2], color="red")
        ax.plot(groundtruth[:, 2], color="red")
        ax.plot(samples_min[:, 2], "--", color="red")
        ax.plot(samples_max[:, 2], "--", color="red")
        ax.plot(truex[:, 2], "rx", markersize=2)

        x = np.arange(0, 17, 1)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(x+1)

        ax.set_ylim([0.0, 110.0])
        cover_num = 0

        for ti in range(truex.shape[0]):

            if (truex[ti, 1] <= samples_max[ti, 1]) and (truex[ti, 1] >= samples_min[ti, 1]):
                cover_num += 1

        for ti in range(truex.shape[0]):
            if (truex[ti, 2] <= samples_max[ti, 2]) and (truex[ti, 2] >= samples_min[ti, 2]):
                cover_num += 1

        #self.writer.add_scalar('Loss/train/coverage_{}'.format(model_num), 100 * cover_num / truex.shape[0], itr)
        print(cover_num / 34)

        plt.savefig(os.path.join(self.log_path, "Forwardsamples_model_num{}_interval{}.png".format(model_num, itr)),
                    bbox_inches='tight')
        plt.close(fig=fig)
        plt.clf()
        #self.writer.add_image('forward_{}'.format(model_num), fig, itr)

    def param_img_store(self, params, model_num, itr):
        sample_list = []
        temp = params[-1]
        temp = temp.cpu().detach().numpy()
        sample_list.append(temp)

        # fig = plt.figure()
        temp = np.asarray(sample_list)

        temp = temp[0]
        #print(temp.shape)
        hpd_mu = []
        modes_mu = []


        #print(modes_mu)

        # print(yset.shape)
        # print(temp.shape)
        #np.save(
        #    os.path.join("./results/MSIR/gensamples", "generated_sampled_epoch{}_interval{}.png".format(epoch, i)),
        #    temp)

        fig, axs = plt.subplots(1, 3)
        #fig.suptitle('SIR function')

        axs[0].hist(temp[:, 1], bins=128)
        axs[0].set_title(r"$\beta$")
        axs[0].set_xlim([1.0, 2.0])
        axs[0].axvline(x=1.5, color="r")
        axs[1].hist(temp[:, 2], bins=128)
        axs[1].set_title(r"$\gamma$")
        axs[1].set_xlim([0.2, 0.8])
        axs[1].axvline(x=0.5, color="r")
        axs[2].hist(temp[:, 0], bins=100)
        axs[2].set_title("Initial Susceptible")
        axs[2].axvline(x=99, color="r")
        axs[2].set_xlim([80.0, 120.0])
        temp[:, [1, 2]] = temp[:, [2, 1]]
        temp[:, [0, 2]] = temp[:, [2, 0]]
        #for j in range(3):
        #    temp_hpd, _ , _, temp_modes = hpd_grid(sample=temp[:,j], alpha=0.05)
        #    hpd_mu.append(temp_hpd)
        #    modes_mu.append(temp_modes)

        #print(hpd_mu)
        #for idx in range(3):
        #    axs[idx].axvline(x=modes_mu[idx], color = "green")
        #    x0, x1 = hpd_mu[idx][0]
        #    axs[idx].axvline(x=x0, color='grey', linestyle='--', linewidth=1)
        #    axs[idx].axvline(x=x1, color='grey', linestyle='--', linewidth=1)

        fig.set_size_inches(20.5, 8.5)
        #print(hpd_mu)
        #print(modes_mu)
        #print("")

        plt.savefig(os.path.join(os.path.join(self.log_path, "param_samples_model_num{}_interval{}.png".format(model_num, itr))))
        # plt.close(fig = fig)
        plt.close(fig=fig)
        plt.clf()
        #self.writer.add_image('params_{}'.format(model_num), fig, itr)

    def param_ls_img_store(self, params, model_num, itr):

        raise NotImplementedError


class SEIR_Experiment_Writing():
    no_log_keys = ['project', 'name',
                    'eval_every']

    def __init__(self, args,
                 data_id, model_id):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        log_base = os.path.join(dir_path, 'results/SEIR/log')
        self.log_base = log_base
        print("Saving log : {}".format(log_base))
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        log_path = os.path.join(self.log_base, data_id, model_id, args.name)
        self.check_every = args.eval_every
        self.log_path = log_path
        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id

        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
#        if args.log_tb:
        self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
        self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)

    def save_args(self, args):
        # Save args
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(os.path.join(self.log_path,'args_table.txt'), "w") as f:
            f.write(str(args_table))

    def create_folders(self):
        # Create log folder
        os.makedirs(self.log_path)

    def loss_store(self, loss, nll, itr,time, model_num):
        loss = loss.detach().cpu().numpy()
        nll = nll.detach().cpu().numpy()
        nll = np.mean(nll)

        self.writer.add_scalar('Loss/train/ELBO_{}'.format(model_num), loss, itr)
        self.writer.add_scalar('Loss/train/nll_{}'.format(model_num), nll, itr)
        self.writer.add_scalar('Loss/train/time_{}'.format(model_num), time, 0)
        print(
            'Iter: {}/{},loss:{}, NLL:{}, Time :{}'.format(itr + 1, 400, loss, nll, time), end='\r')

    def forward_img_store(self, samples, truex, groundtruth, model_num, itr):
        I_w, age_class = samples # B * 54, B * 3

        truex = truex.detach().cpu().numpy() # 54, 3
        truex = truex[:, -1]#np.sum(truex, axis=1)
        fig, ax = plt.subplots(1, 2)
        #print(truex.shape)
        #print(truex)
        B = I_w.shape[0]
        #print(I_w.shape)
        #I_w = torch.tensor(I_w)

        #I_w = data_samples.detach().cpu().numpy()
        #raise ValueError
        samples_mean = np.mean(I_w, axis=0)
        samples_min = np.percentile(I_w, 97.5, axis=0)
        samples_max = np.percentile(I_w, 2.5, axis=0)

        age_class_min = np.percentile(age_class, 97.5, axis=0)
        age_class_max = np.percentile(age_class, 2.5, axis=0)
        age_class_mean = np.mean(age_class, axis=0)
        #print(age_class_mean)

        ax[0].plot(samples_mean[:], color="red")
        ax[0].plot(samples_min[:], "--", color="red")
        ax[0].plot(samples_max[:], "--", color="red")
        # ax[pp].plot(cases_prob[0][:,pp].detach().cpu().numpy(), color = "red")
        ax[0].plot(truex, "ko", markersize=2)
        ax[0].set_xlabel('Time (weeks)', fontsize=10)

        age_class_x = np.asarray([1,2,3])
        age_true = np.asarray([0.42, 0.30, 0.28])

        label = ["0.5-5", "5-15", "16+"]
        ax[1].bar(age_class_x - 0.4, age_true, width=0.4, color='k', align='center', label="Observed")
        ax[1].bar(age_class_x, age_class_mean, width=0.4, color='r', align='center', label="Simulated")
        ax[1].xaxis.set_ticks(age_class_x - 0.2)
        ax[1].set_xticklabels(label)
        ax[1].set_ylim([0.0, 0.5])
        ax[1].set_xlabel('Age groups (years)', fontsize=10)
        ax[1].legend()
        plt.savefig(os.path.join(self.log_path, "Forwardsamples_model_num{}_interval{}.png".format(model_num, itr)),
                    bbox_inches='tight')
        plt.close(fig=fig)

        truex = np.reshape(truex, (1, 53))
        #print(truex.shape)
        #print(samples.shape)
        print(np.mean(np.square((age_class-age_true)*100)))
        print(np.mean(np.square(I_w - truex)))
        """
        cover_num = 0
        for ti in range(truex.shape[0]):
            #print(truex[ti],samples_max[ti])
            if (truex[ti] <= samples_min[ti]) and (truex[ti] >= samples_max[ti]):
                cover_num += 1

        age_cover = 0
        for ti in range(3):
            if (age_true[ti] <= age_class_min[ti]) and (age_true[ti] >= age_class_max[ti]):
                age_cover += 1

        print(cover_num/54)
        print(age_cover)
        print(np.mean(np.square((age_class-age_true)*100)))
        print(np.mean(samples_max-samples_min))
        self.writer.add_scalar('Loss/train/coverage_{}'.format(model_num), 100 * cover_num / truex.shape[0], itr)
        """

    def param_img_store(self, params, model_num, itr):
        raise NotImplementedError

    def param_ls_img_store(self, params, model_num, itr):

        for p_idx in range(len(params)):
            sample_list = []
            #print(p_idx)
            temp = params[p_idx]
            temp = temp.cpu().detach().numpy()
            sample_list.append(temp)

            temp = np.asarray(sample_list)

            temp = temp[0]

            fig, axs = plt.subplots(2, 3, constrained_layout=True)

            axs[0, 0].hist(temp[:, 0], bins=128, edgecolor='black', color='gray', alpha=0.3)
            axs[0, 0].set_title(r'$\beta$', size=23)
            axs[0, 0].set_xlim([0.5 * 1e-6, 3 * 1e-6])
            axs[0, 1].hist(temp[:, 1], bins=128, edgecolor='black', color='gray', alpha=0.3)
            axs[0, 1].set_title(r'$N_{0}$', size=23)
            axs[0, 1].set_xlim([140 * 1e3, 300 * 1e3])  # 130
            axs[0, 2].hist(temp[:, 2], bins=128, edgecolor='black', color='gray', alpha=0.3)
            axs[0, 2].set_title(r'$f_{e}$', size=23)
            axs[0, 2].set_xlim([0.0, 0.001])  # 0.0005
            axs[1, 0].set_title(r'$a_{sh}$', size=23)
            axs[1, 0].set_xlim([0.8, 1.2])
            axs[1, 0].hist(temp[:, 3], bins=128, edgecolor='black', color='gray', alpha=0.3)
            axs[1, 1].set_title(r'$a_{rt}$', size=23)
            axs[1, 1].set_xlim([0.08, 0.11])  # 0.09
            axs[1, 1].hist(temp[:, 4], bins=128, edgecolor='black', color='gray', alpha=0.3)
            fig.delaxes(axs[1, 2])

            plt.savefig(os.path.join(os.path.join(self.log_path, "param_samples_model_num{}_interval{}_Ladder{}.png".format(model_num, itr, p_idx))))
            plt.close(fig=fig)

