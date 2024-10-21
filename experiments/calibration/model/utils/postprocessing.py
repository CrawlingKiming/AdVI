import numpy as np
import torch
import math 

def rej_sampling(loss_fn, mycan, model, truex, args, nsamp=1000, bs=5000, maxit=100, bound_factor=1.5, logz=False, verbose=True):
    # p(x) <= M * q(x)
    # logp < logM + logq
    xacc = []
    logr = []
    new_r_ls = []
    samples_ls = []
    cases_prob_ls = []
    nacc = 0

    for i in range(maxit):
        print(i) if verbose else None
        if args.dataset == "MSIR":
            samples, cases_prob, new_x, new_r, log_probq, log_probp = loss_fn(can_model=mycan, model=model, observation=truex, args=args, eval=True, recover=True, itr=100)
        
        log_ratio = log_probp - log_probq 
        logM = torch.max(log_ratio)
       

        logr.append(log_ratio)

        logM = logM + math.log(bound_factor)
        log_acc_prob =  (log_ratio + logM) + torch.log(bs)

        acc = (torch.log(torch.random.uniform([args.eval_size])) < log_acc_prob).cpu().detach().numpy()
        print(acc)
        nacc += np.sum(acc)
        #xprop = xprop.numpy()
        xacc.append(new_x[acc, :])
        samples_ls.append(samples[acc, :])
        cases_prob_ls.append(cases_prob[acc, :])
        new_r_ls.append(new_r[acc, :])
        if nacc >= nsamp:
            break
    xacc = torch.vstack(xacc) # Error maybe 
    samples = torch.vstack(samples_ls)
    new_r = torch.vstack(new_r_ls)
    cases_prob = torch.vstack(cases_prob_ls)
    print(xacc.shape) if verbose else None
    #xacc = xacc[:nsamp]

    return samples, cases_prob, new_r, xacc