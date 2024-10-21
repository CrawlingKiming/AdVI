import math

import torch

from survae.distributions import StandardNormal
from survae.transforms import Sigmoid
from .SIR_Model import ODE_Solver
from .SIR_Model import SIR
from .subclass.InverseFlow import SqFlow, MSIR_Flow, Build_Spline_Order_3_Transform, GaussianModel, Build_Shuffle_Order_3_Transform
from .subclass.diagnostic import PSIS_sampling
from .subclass.layers import ShiftBijection, ScaleBijection
from .subclass.surjective import BoundSurjection_S

import arviz as az
# from MSIR_fullModel import ODE_Solver as ODE_Solver
# from MSIR_fullModel import MSIR as MSIR

def plain_convert(x):
    raise NotImplementedError

def get_ladder(args):
    if args.AIS:
        idx_ladder = [10, 20]
        temp_ladder = [args.temp, 1]
        w_ladder_ls = [[1, 1]]

    else:
        idx_ladder = [args.num_flows]
        temp_ladder = [1]
        w_ladder_ls = [[1]]

    assert len(idx_ladder) == len(temp_ladder)

    return idx_ladder, temp_ladder, w_ladder_ls

def Bound_T():
    transforms=[]

    transforms += [ShiftBijection(shift=torch.ones(size=(3,)) * 2.)]
    transforms += [ScaleBijection(scale=torch.ones(size=(3,)) * 0.25)]
    transforms += [ScaleBijection(scale=torch.tensor([[63.0, 3.0, 3.0]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[37.0, 0.0, 0.0]])))
    return transforms

def Sig_T():
    transforms = []
    transforms += [Sigmoid()]
    transforms += [ScaleBijection(scale=torch.tensor([[63.0, 3.0, 3.0]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[37.0, 0.0, 0.0]])))
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


def build_SIR_model(args, data_shape):

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
        model1_transforms1 = Build_Spline_Order_3_Transform(D=D, num_bin=128, num_flows=args.num_flows)
        ladd_step = 3

    if args.bound_surjection:
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder
                        ,bound_q=bound_S).to(args.device)
    else:
        print("Non BS")
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder, bound_q=None).to(args.device)

    model1 = SqFlow(base_dist=StandardNormal((D,)), transforms=model1_transforms1,ladd_step=ladd_step).to(args.device)

    return mycan, model1

def get_SIR_loss(can_model, model, observation, args, itr, eval = False, recover= False):

    mycan = can_model
    truex  = observation
    model_num = truex.shape[1]
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    #print(truex)
    sampled_x_ls, log_probq_ls, log_probp_ls = mycan(num_samples = args.batch_size, model = model)

    if eval:
        sampled_x_ls, log_probq_ls, log_probp_ls = mycan(num_samples=args.eval_size, model=model)
    if recover :
        sampled_x_ls, _, _, _ = mycan(num_samples = args.batch_size, model = model, plain = True)
        #sampled_x_ls = sampled_x_ls.to(args.device)
    #if i == 0:
    #    sampled_x_ls, log_probq_ls, log_probp_ls, z_ls = mycan(model = model model, plain=True, ladder=idx_ladder)

    sampled_x = torch.cat(sampled_x_ls, 0) # B * 3


    if recover :
        sampled_x = sampled_x.to(args.device)
    gt = truex.clone()
    gt = gt.to(args.device)
    x = sampled_x
    #
    #sampled_x = torch.tensor([[38, 0.5, 0.5]], device=args.device)

    #
    B = x.shape[0]

    S0 = sampled_x[:, [0]]
    IR0 = torch.tensor([[1,0]], dtype=torch.float64, device=args.device).repeat(B,1)
    Z0 = torch.hstack((S0,IR0)).to(args.device)

    term = 21
    tt = torch.linspace(0, term, steps=term + 1, requires_grad=False).to(args.device)
    #print(tt)
    #tt = torch.cat((torch.tensor([0.0], device=args.device), tt[-118:]))
    mySIR = SIR().to(args.device)
    #print(sampled_x)
    observed_y = ODE_Solver(Z0, sampled_x[:,[1,2]],tt, mySIR)
    z_samples = observed_y  # [:, -1:, :]
    samples = z_samples
    samples = torch.transpose(samples, 0, 1)
    #print(samples.shape)
    if eval:
        samples = torch.chunk(samples, len(idx_ladder), dim=0)
        return samples[-1]

    gt = gt.repeat(B,1,1)
    samples = torch.clamp(samples, min=0.0)
    poi_1 = torch.distributions.poisson.Poisson(samples[:,:model_num, 1]+1e-3)
    poi_2 = torch.distributions.poisson.Poisson(samples[:,:model_num, 2])
    #pri_1 = torch.distributions.normal.Normal(torch.tensor([[38]], device=args.device),
    # (torch.tensor([[10]], device=args.device)))
    #pri_2 = torch.distributions.normal.Normal(torch.tensor([[1.5]], device=args.device),
    # (torch.tensor([[3]], device=args.device)))
    p = poi_1.log_prob(gt[:,:model_num,0]) + poi_2.log_prob(gt[:,:model_num, 1]) 
    #pri = pri_1.log_prob(S0) + pri_2.log_prob(sampled_x[:,[1]]) + pri_2.log_prob(sampled_x[:,[2]])

    p = torch.sum(p, dim=1) 

    kl_dive = p #+ pri #-1 * (0.5 * p / (e**2)) + base
    #########################################

    kl_dive_con = kl_dive#torch.sum(kl_dive, dim=1)
    kl_dive_ls = torch.chunk(kl_dive_con, len(idx_ladder), dim=0) # Per ladder, we get this one

    w_ladder = w_ladder_ls[0]

    loss = 0.0
    for idx in range(len(idx_ladder)):
        weight = 1
        log_probp = log_probp_ls[idx]
        log_probq = log_probq_ls[idx]
        kl_dive = kl_dive_ls[idx]
        T = temp_ladder[idx]
        w = w_ladder[idx] #* ((1e4 **(itr/300))**idx)
        anneal = (1 - 1 / T)
        loss2 = - log_probq 
        idx_loss = loss2 + kl_dive / T 
        if args.AIS :
            if itr < 100 and idx == len(idx_ladder) -1:
                weight = 0
                w = w * weight

            if itr >= 100 and idx == 0 :
                weight = 0
                w = w * weight
        loss = loss - 1 * (idx_loss.mean()) * w
        if args.AIS :
            if itr > 200:

                logw = idx_loss.clone().detach()
                logw_2, k = az.psislw(logw.clone().detach().cpu().numpy())

                w = (logw_2)
                w = torch.tensor(data=w, device=args.device)
                logw = logw_2 
                w = torch.exp(w - torch.max(w))
                loss = -1 * w * idx_loss  #torch.log(loss.mean())
                loss = loss.mean()

    nll = kl_dive

    return loss, nll, samples


def only_forward(param_sample, args):

    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)

    sampled_x = torch.tensor(data=param_sample, device=args.device)
    x = sampled_x

    B = x.shape[0]

    S0 = sampled_x[:, [0]]
    IR0 = torch.tensor([[1,0]], dtype=torch.float64, device=args.device).repeat(B,1)
    Z0 = torch.hstack((S0,IR0))

    term = 17
    tt = torch.linspace(0, term, steps=term + 1, requires_grad=False).to(args.device)
    #print(tt)
    #tt = torch.cat((torch.tensor([0.0], device=args.device), tt[-118:]))
    mySIR = SIR().to(args.device)

    observed_y = ODE_Solver(Z0, sampled_x[:,[1,2]],tt, mySIR)
    z_samples = observed_y  # [:, -1:, :]
    samples = z_samples

    samples = torch.transpose(samples, 0, 1)
    return samples


