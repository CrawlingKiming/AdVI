import numpy as np
import torch
from torch.distributions import Normal, Gamma, HalfNormal, MultivariateNormal
from survae.distributions import StandardNormal
from survae.transforms import Sigmoid
from .subclass.InverseFlow import SqFlow, MSIR_Flow, Build_Shuffle_Transform, Build_Spline_Transform, GaussianModel, Build_Spline_Order_6_Transform
from .subclass.layers import ShiftBijection, ScaleBijection, Exp
from .subclass.surjective import BoundSurjection_S
from .subclass.diagnostic import PSIS_sampling, TruncatedNormal
from .subclass.fourier import Fourier
import arviz as az 
import matplotlib.pyplot as plt

def get_ladder(args):
    if args.AIS:
        idx_ladder = [10, 20]#[args.num_flows//4,args.num_flows//2, args.num_flows]
        temp_ladder = [args.temp, 1]
        w_ladder_ls = [[6, 1]]


    else:
        idx_ladder = [args.num_flows]
        temp_ladder = [1]
        w_ladder_ls = [[1]]

    assert len(idx_ladder) == len(temp_ladder)

    return idx_ladder, temp_ladder, w_ladder_ls

def Bound_T():
    transforms = []
    transforms += [ShiftBijection(shift=torch.ones(size=(10,)) * 2.)]
    transforms += [ScaleBijection(scale=torch.ones(size=(10,)) * 0.25)]
    transforms += [ScaleBijection(
        scale=torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1500, 5e4, 200, 7e7, 1.0]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1e7, 1.0]]))) 


    return transforms

def Sig_T():
    transforms = []
    transforms += [Sigmoid()]
    transforms += [ScaleBijection(
        scale=torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1500, 5e4, 100, 6e7, 1.0]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2e7, 1.0]])))

    return transforms

class CanModel(torch.nn.Module):

    def __init__(self, idx_ladder, data_shape, pretrained=None, bound_q=None):
        super(CanModel, self).__init__()

        self.pretrained = pretrained
        self.bound_q = bound_q # Bound Surjection

        self.idx_ladder = idx_ladder
        self.base_dist = StandardNormal((data_shape,))

        self.zi = self.base_dist.sample(num_samples = 1)
        self.x = None 

    def forward(self, num_samples, model, param=False):
        """
        sampled_x : Sampled Parameters from Approx. q
        log_probq_ls : log_prob of Approx. q
        log_probz_ls : log_prob of initial dist z
        """

        ladder = self.idx_ladder
        x = self.base_dist.sample(num_samples = num_samples) 
        log_probq_ls, log_probz_ls, z_ls = model.log_prob(x, ladder)
        z_concat = torch.cat(z_ls, 0) # (L*B) * D

        if self.bound_q:
            z_concat, entropy = self.bound_q(z_concat)
            entropy_ls = torch.chunk(entropy, len(self.idx_ladder), dim=0)
            log_probp_ls = entropy_ls
            log_probq_ls = list(map(lambda x, y: x - y, log_probq_ls, log_probp_ls))

        log_probp_transformed, _, sampled_x = self.pretrained.log_prob(x=z_concat)
        log_probq_ls = list(map(lambda x, y: x - y, log_probq_ls, torch.chunk(log_probp_transformed, len(self.idx_ladder), dim=0)))
        sampled_x_ls = torch.chunk(sampled_x, len(self.idx_ladder), dim=0)

        return sampled_x_ls, log_probq_ls, log_probz_ls


def build_WRF_model(args, data_shape):

    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    D = data_shape
    bound_S = BoundSurjection_S()

    if args.bound_surjection :
        pretrained_transforms = Bound_T()
    else :
        pretrained_transforms = Sig_T()

    NormalFlow = MSIR_Flow(base_dist=StandardNormal((D,)),
                           transforms=pretrained_transforms).to(args.device)

    if args.gaussian:
        model1 = GaussianModel(StandardNormal((D,)), D)
        ladd_step = 1
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder, bound_q=None).to(args.device)
        return mycan, model1
    else:
        model1_transforms1 = Build_Spline_Transform(D=D, num_bin=24, num_flows=args.num_flows)

        ladd_step = 3

    if args.bound_surjection:
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder
                        ,bound_q=bound_S).to(args.device)
    else:
        print("Non BS")
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder, bound_q=None).to(args.device)

    model1 = SqFlow(base_dist=StandardNormal((D,)), transforms=model1_transforms1,ladd_step=ladd_step).to(args.device)

    return mycan, model1

def ingamma_log(x, alpha, beta):
    x = x
    const = alpha * torch.log(beta) - torch.lgamma(alpha)
    x = -1 * (alpha + 1) * torch.log(x) - beta / x
    res = const + x
    return res

def get_WRF_loss(can_model, model, observation, args, itr, eval = False, recover= False):
    mycan = can_model
    # Observations
    obs0, cov_theta_est, mat_Y_em, mat_Y_mean, mle_est, param_par_rescale, param_par = observation 

    obs_new = obs0 - mat_Y_mean[:,0]

    tt = 480
    t_dist = np.abs(np.arange(tt) - np.arange(tt)[:, np.newaxis])
    t_dist = torch.tensor(t_dist, dtype=torch.float, device=args.device)
    cov_t_est = torch.exp(-1 * t_dist / mle_est[6])
    cov_thetanew_p = cov_t_est * (1 + mle_est[5])
    cov_theta_inv_est = torch.linalg.inv(cov_theta_est)
    
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    sampled_x_ls, log_probq_ls, log_probz_ls = mycan(num_samples = args.batch_size, model = model)
    if eval:
        sampled_x_ls, log_probq_ls, log_probz_ls = mycan(num_samples=args.eval_size, model=model)
    if recover:
        sampled_x_ls, _, _ = mycan(num_samples = args.batch_size, model = model, plain = True)

    sampled_x = torch.cat(sampled_x_ls, 0)
    if recover:
        sampled_x = sampled_x.to(args.device)

    x = sampled_x
    B = x.shape[0]

    theta = x[:, :5]
    range_disc = x[:, [5]]
    sill_disc = x[:, [6]]
    nugget_disc = x[:, [7]]
    sill_y = x[:, [8]]

    cov_t_est = cov_t_est.repeat(B, 1, 1)
    cov_thetanew_p = cov_thetanew_p.repeat(B, 1, 1) # B 480 480 
    cov_theta_inv_est = cov_theta_inv_est.repeat(B, 1, 1) # B 400 400
    mat_Y_em = mat_Y_em.repeat(B, 1, 1) # B 400 480 
    cov_theta_est = cov_theta_est.repeat(B, 1, 1) # B 400 400
    param_par_rescale = param_par_rescale.repeat(B, 1, 1) # B 400 5 
    obs_new = obs_new.repeat(B,1)
    theta = theta[:, None, :] # B 1 5 

    dist_theta = -1 * torch.abs(param_par_rescale - theta)
    
    cov_theat1_p = torch.exp(dist_theta[:, :, 0] / mle_est[0])
    #print(dist_theta)
    
    for j in range(1, 5):
        cov_theat1_p = cov_theat1_p * torch.exp(dist_theta[:, :, j] / mle_est[j]) # B 400
    cov_theta1_p = cov_theat1_p[:, None, :] # B 1 400

    mean_em_p = cov_theta1_p @ cov_theta_inv_est @ mat_Y_em # B 480 
    temp = cov_theta1_p @ cov_theta_inv_est # @ torch.transpose(cov_theta1_p,1,2)
    temp = temp @ torch.transpose(cov_theta1_p,1,2)
    cov_em_p = cov_t_est
    for i in range(B):
        cov_em_p[i, :, :] *= temp[i] # B 480 480 

    cov_em_p = cov_thetanew_p - cov_em_p
    for i in range(B):
        cov_em_p[i, :, :] *= sill_y[i] # B 480 480 

    t_dist2 = torch.square(t_dist) 
    #print(t_dist)
    tcov_disc = torch.abs(t_dist.repeat(B, 1, 1))
    cov_disc = torch.zeros(tcov_disc.shape, device=args.device)
    for i in range(B):
        cov_disc[i, :, :] = sill_disc[i] * torch.exp(-1 * (tcov_disc[i, :, :] / (range_disc[i]))) # B 480 480 
    
    diag = torch.eye(480, device=args.device)
    diag = diag[None, :, :]
    diag = diag.repeat(B, 1, 1)
    tempd = torch.zeros(diag.shape, device=args.device)
    #print(nugget_disc.shape)
    for i in range(B):
        tempd[i, :, :] = nugget_disc[i] * diag[i, :, :] 
    cov_disc = cov_disc + tempd
    total_cov = cov_disc + cov_em_p
    like = torch.zeros((B,1), device=args.device)


    for i in range(B):
        normal = MultivariateNormal(mean_em_p[i, 0, :], total_cov[i, :, :]) # B, 480
        
        like[i] =  normal.log_prob(obs_new[i,:]) # B, 480

    shape_range = 2
    scale_range = 150 

    gamm = Gamma(2, 1/150)

    shape_sill_disc = torch.tensor(2, device=args.device)
    rate_sill_disc = torch.tensor(10500, device=args.device)
    shape_nug = torch.tensor(2,  device=args.device) 
    rate_nug = torch.tensor(16.5, device=args.device)
    shape_sill = torch.tensor(50, device=args.device)
    rate_sill = torch.tensor(5201459527, device=args.device) 

    pri = gamm.log_prob(range_disc).to(args.device) + ingamma_log(sill_disc, shape_sill_disc, rate_sill_disc) + ingamma_log(nugget_disc, shape_nug, rate_nug) + ingamma_log(sill_y, shape_sill, rate_sill)
    pri = torch.sum(pri, dim=1)

    ######################################
    w_ladder = w_ladder_ls[0]

    loss = 0.0
    
    if eval:
        Y_chunked = torch.chunk(mean_em_p, len(idx_ladder), dim=0) #z_samples
        samples = Y_chunked[-1].detach().cpu().numpy()
        return samples

    like_ls = torch.chunk(like, len(idx_ladder), dim=0)  # Per ladder, we get this one
    pri_ls = torch.chunk(pri, len(idx_ladder), dim=0) 
    loss_list = []

    ## Block for Blocks 
    if args.AIS : 
        if itr < 150: 
            Bl = 0
        else: 
            Bl = len(idx_ladder) -1 
    else:
        Bl = 0 
    #Bl = len(idx_ladder) -1 
    log_probz = log_probz_ls[Bl]
    log_probq = log_probq_ls[Bl ]
    kl_dive = like_ls[Bl]
    pri = pri_ls[Bl]
    T = temp_ladder[Bl]
    anneal = (1 - 1 / T)
    loss_fir = kl_dive / T + pri - log_probq #+ log_probz * anneal  
    
    #print(loss_fir)
    #raise ValueError
    loss = -1 * loss_fir.mean()

    if args.AIS:
        if itr > 400:
            logw = loss_fir.clone().detach()
            logw_2, k = az.psislw(logw.clone().detach().cpu().numpy())            
            w = (logw_2)
            w = torch.tensor(data=w, device=args.device)
            logw = logw_2 
            w = torch.exp(w - torch.max(w))
            loss = -1 * w * loss_fir * 30 #torch.log(loss.mean())
            loss = loss.mean()

    nll = kl_dive + pri
    
    null = 0 # used for debugging
    return loss, nll, null



def only_forward(param_sample, args):
    sampled_x = torch.tensor(data=param_sample, device=args.device)
    x = sampled_x

    term = 500
    tt = torch.linspace(0, term+2, steps=term+1, requires_grad=False).to(args.device)

    myRNA = mRNA(args=args).to(args.device)

    observed_y = ODE_Solver(x, tt, myRNA, args)
    z_samples = observed_y  # [:, -1:, :]
    z_samples = torch.transpose(z_samples, 0, 1)
    z_samples = z_samples[:, -66:] # B, 66

    return z_samples




