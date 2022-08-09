from scipy.stats import genpareto
import numpy as np
import torch

def PSIS(rs):
    """
    PSIS fitting
    """
    rs = rs.clone().detach().cpu().numpy()
    S = rs.shape[0]
    M = min([int(S/5), int(3 * np.sqrt(S))])
    idx = rs.argsort()
    ws = rs[idx] # Order from lowest to highest
    ws_M = ws[M:]
    k, loc, scale = genpareto.fit(ws_M)
    return k

def PSIS_sampling(rs):
    """
    Pareto Smoothed Importance Sampling

    input : (B,) shape numpy, unnormalized, raw ratios
    output :
    """
    S = rs.shape[0]
    M = int(S / 5)
    idx = rs.argsort()
    ws = rs[idx] # Order from lowest to highest
    ws_M = ws[S-M:]
    assert len(ws_M) == M
    ws_m = ws[:S-M]
    k, loc, scale = genpareto.fit(ws_M)
    #k = (M*k + 50 * 0.5) / (M + 50),
    #print(k, loc, scale)
    z = np.linspace(start=1, stop=M, num=M)

    incdf = genpareto.ppf((z-0.5)/M, k, loc=loc, scale=scale)
    incdf = np.minimum(incdf, max(ws))
    ws = np.concatenate((ws_m, incdf))

    ii = idx.argsort()
    ws = ws[ii]

    return ws

def IS_truncation(rs):
    """
    Truncated IS

    input : (B,) shape numpy, unnormalized, raw ratios
    output :
    """
    #print(rs)
    S = rs.shape[0]
    #M = int(3 * np.sqrt(S))#int(S/5) # min([int(S / 5), int(3 * np.sqrt(S))])
    rbar = torch.mean(rs)
    trun = rbar*int(np.sqrt(S))

    r_min = rs - trun
    #print(trun)
    ws = torch.clamp(r_min, min=0.0) + trun#.detach()
    return ws

if __name__ == "__main__":
    """
    tt = np.asarray([2.6564e-19, 0.0000e+00, 1.8915e-09, 6.8283e-10, 1.4946e-14, 2.6439e-13,
        4.9561e-15, 3.3311e-23, 3.3128e-17, 6.2369e-15, 9.5742e-33, 0.0000e+00,
        3.0620e-23, 4.0904e-36, 0.0000e+00, 1.0587e-09, 1.0235e-08, 3.7449e-31,
        2.3794e-24, 3.9686e-01, 3.7549e-32, 5.1779e-08, 1.2972e-22, 2.7028e-29,
        7.2263e-23, 1.2651e-20, 4.0362e-19, 4.1048e-10, 8.3817e-07, 2.1784e-07,
        0.0000e+00, 5.6191e-13, 1.6760e-15, 8.0887e-16, 0.0000e+00, 4.5942e-25,
        3.6891e-09, 2.9349e-18, 3.6124e-21, 2.4173e-15, 5.1908e-13, 4.9467e-04,
        1.0242e-22, 1.9271e-12, 1.0511e-01, 1.4090e-15, 4.3390e-10, 2.9071e-34,
        0.0000e+00, 3.6342e-35, 3.7890e-18, 9.2290e-14, 2.8286e-01, 0.0000e+00,
        1.1285e-15, 7.5107e-31, 9.5600e-38, 3.1249e-39, 2.1422e-16, 2.2123e-07,
        1.2481e-26, 1.4196e-05, 1.8736e-08, 0.0000e+00, 9.8074e-10, 0.0000e+00,
        2.5214e-23, 3.9788e-12, 1.4947e-25, 2.6091e-29, 4.1940e-14, 7.8903e-11,
        2.8659e-11, 8.9069e-12, 6.0045e-04, 9.2771e-16, 1.0477e-12, 3.4065e-24,
        0.0000e+00, 2.8549e-06, 2.0319e-08, 1.1712e-12, 8.0258e-11, 7.4522e-20,
        2.1273e-36, 1.3360e-33, 0.0000e+00, 2.1083e-01, 3.0813e-04, 8.7777e-13,
        2.0518e-19, 5.6173e-09, 1.5133e-12, 2.0208e-19, 1.3807e-07, 6.0283e-09,
        1.0025e-18, 2.3038e-17, 4.7584e-06, 0.0000e+00, 0.0000e+00, 2.2210e-04,
        4.9331e-16, 6.4811e-19, 1.9639e-05, 8.0731e-15, 2.1879e-08, 0.0000e+00,
        2.1045e-15, 3.6317e-24, 2.6091e-20, 0.0000e+00, 1.3985e-12, 7.0088e-38,
        2.1113e-19, 2.7310e-08, 0.0000e+00, 1.8064e-13, 6.5297e-07, 1.1269e-19,
        5.5105e-17, 2.6652e-03, 7.5337e-26, 6.3227e-15, 2.8869e-30, 2.2028e-09,
        2.1029e-08, 1.9884e-42])
    """

    tt = np.asarray([1,2,3,4,5,10,6,7,8,9,100])

    #print(np.sum(tt))
    print(PSIS_sampling(tt))





