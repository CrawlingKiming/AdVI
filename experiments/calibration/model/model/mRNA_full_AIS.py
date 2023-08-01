import numpy as np
import torch
from torch.distributions import Normal, Gamma, HalfNormal
from survae.distributions import StandardNormal
from survae.transforms import Sigmoid
from .mRNA_fullModel import ODE_Solver as ODE_Solver
from .mRNA_fullModel import mRNA
from .subclass.InverseFlow import SqFlow, MSIR_Flow, Build_Shuffle_Transform, Build_Spline_Transform, GaussianModel, Build_Spline_Order_11_Transform
from .subclass.layers import ShiftBijection, ScaleBijection, Exp
from .subclass.surjective import BoundSurjection_S
from .subclass.diagnostic import PSIS_sampling, TruncatedNormal
from .subclass.fourier import Fourier

def get_ladder(args):
    if args.AIS:
        idx_ladder = [args.num_flows//4,args.num_flows//2, args.num_flows]
        temp_ladder = [args.temp,2, 1.]
        w_ladder_ls = [[4, 2, 1]]

    else:
        idx_ladder = [args.num_flows]
        temp_ladder = [1]
        w_ladder_ls = [[1]]

    assert len(idx_ladder) == len(temp_ladder)

    return idx_ladder, temp_ladder, w_ladder_ls

def Bound_T():
    transforms = []
    transforms += [ShiftBijection(shift=torch.ones(size=(11,)) * 2.)]
    transforms += [ScaleBijection(scale=torch.ones(size=(11,)) * 0.25)]
    transforms += [ScaleBijection(
        scale=torch.tensor([[0.5, 3.5, 0.6, 0.5, 0.9, 0.8, 1.0, 9.0, 20.0, 20.0, 20.0]]))]

    return transforms

def Sig_T():
    transforms = []
    transforms += [Sigmoid()]
    transforms += [ScaleBijection(
        scale=torch.tensor([[0.5, 3.5, 0.6, 0.5, 0.9, 0.8, 1.0, 9.0, 20.0, 20.0, 20.0]]))]

    return transforms

class CanModel(torch.nn.Module):

    def __init__(self, idx_ladder, data_shape, pretrained=None, bound_q=None):
        super(CanModel, self).__init__()

        self.pretrained = pretrained
        self.bound_q = bound_q # Bound Surjection

        self.idx_ladder = idx_ladder
        self.base_dist = StandardNormal((data_shape,))

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


def build_mRNA_model(args, data_shape):

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
        model1_transforms1 = GaussianModel(StandardNormal((D,)), D, pretrained_transforms)
        ladd_step = 1

    #    model1_transforms1 = Build_Shuffle_Order_5_Transform(D=D, num_flows=args.num_flows, steps=10)
    else:
        #model1_transforms1 = Build_Shuffle_Transform(D=D, num_flows=args.num_flows)
        model1_transforms1 = Build_Spline_Order_11_Transform(D=D, num_bin=24, num_flows=args.num_flows)
        #model1_transforms1 = Build_Auto_Transform(D=D, num_flows=args.num_flows)

        ladd_step = 3

    #model1_transforms1 = Build_Auto_Transform(D=D, num_flows=args.num_flows)

    if args.bound_surjection:
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder
                        ,bound_q=bound_S).to(args.device)
    else:
        print("Non BS")
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder, bound_q=None).to(args.device)

    model1 = SqFlow(base_dist=StandardNormal((D,)), transforms=model1_transforms1,ladd_step=ladd_step).to(args.device)

    return mycan, model1

def get_mRNA_loss(can_model, model, observation, args, itr, eval = False, recover= False):
    mycan = can_model

    # Observations
    gt, sk = observation
    gt = gt.to(args.device)
    sk = sk.to(args.device)

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
    ###############################################
    #x[:, [0,1,2,3,4,5,6,7,8]]= torch.tensor(
    #    [[0.0312, 3.42258214,  0.44889856,  0.23426515,  0.52461510,  0.73610651,  0.88771403,  6.16031968, 14.23507139]],
    #    device=args.device).repeat(x.shape[0], 1)
    x[:, [0]] = 0.8058214 * 0.125 * torch.ones(size=x[:, [5]].shape, device=args.device)
    x[:, [1]] = 0.92258214 * torch.ones(size=x[:,[5]].shape, device=args.device) # 2
    x[:, [2]] = 0.4089856 * torch.ones(size=x[:, [5]].shape, device=args.device) # 3

    x[:, [3]] = 0.373426515 * torch.ones(size=x[:, [5]].shape, device=args.device)
    x[:, [4]] = 0.08426515 * torch.ones(size=x[:, [5]].shape, device=args.device) # 5
    x[:, [5]] = 0.40461510 * torch.ones(size=x[:, [5]].shape, device=args.device)  # 7

    x[:, [6]] = 0.67461510 * torch.ones(size=x[:,[5]].shape, device=args.device)  # 7
    x[:, [7]] = 3.7761510 * torch.ones(size=x[:, [5]].shape, device=args.device)
    x[:, [8]] = 8.8461510 * torch.ones(size=x[:, [5]].shape, device=args.device)
    ##############################################

    # print(x)
    B = x.shape[0]
    gt = gt.repeat(B, 1)
    sk = sk.repeat(B, 1)

    term = 500
    tt = torch.linspace(0, term+2, steps=term+1, requires_grad=False).to(args.device)

    myRNA = mRNA(args=args).to(args.device)
    fou = Fourier()

    observed_y = ODE_Solver(x, tt, myRNA, args)
    z_samples = observed_y  # [:, -1:, :]
    z_samples = torch.transpose(z_samples, 0, 1)
    z_samples = z_samples[:, -66:] # B, 66

    b_hat = fou.coeff(z_samples) # 14, B
    lambd = b_hat.clone()
    lambd = lambd.reshape(7, 2, B)
    #print(lambd.shape, B)
    lambd = torch.sum(torch.square(lambd), 1) # B,
    #print(lambd.shape)
    lambd = lambd.T
    #print(lambd.shape)
    #print(sk)
    #print(sk.shape)
    #raise ValueError
    sig2 = x[:, [-1]]
    tau2 = x[:, [-2]]

    sig_tau = x[:, [9,10]]
    #print(sk.shape, lambd.shape)
    normal1 = Normal(b_hat.T @ fou.basis.to(args.device), torch.sqrt(sig2+1e-3)) # B, 66
    normal2 = TruncatedNormal(loc=sk, scale=torch.sqrt(tau2+1e-3),a=0,b=1e3)#Normal(lambd, torch.sqrt(tau2+1e-3))
    gamm = Gamma(3, 1)
    #print(lambd)
    like_1 = normal1.log_prob(gt) # B, 66
    like_2 = normal2.log_prob(sk) + torch.log(torch.tensor(data=2, device=args.device))#normal2.log_prob(torch.abs(sk-lambd)) # B, 14 #-torch.log(1-normal2.cdf(torch.zeros(size=sk.shape, device=sk.device)) + 1e-3) +
    pri = gamm.log_prob(1 / (sig_tau + 1e-3)) # B, 11
    pri = pri.to(args.device)
    #print(sk, like_2.isinf())
    kl_dive = torch.sum(like_1, dim=1) + torch.sum(like_2, dim=1) + torch.sum(pri, dim=1)
    w_ladder = w_ladder_ls[0]

    loss = 0.0

    if eval:
        Y_chunked = torch.chunk(b_hat.T @ fou.basis.to(args.device), len(idx_ladder), dim=0) #z_samples
        samples = Y_chunked[-1].detach().cpu().numpy()
        return samples

    kl_dive_ls = torch.chunk(kl_dive, len(idx_ladder), dim=0)  # Per ladder, we get this one
    for idx in range(len(idx_ladder)):

        log_probz = log_probz_ls[idx]
        log_probq = log_probq_ls[idx]
        kl_dive = kl_dive_ls[idx]
        T = temp_ladder[idx]

        if args.AIS:
            if idx == len(idx_ladder)-1:
                weight = 1
                if itr < 250:
                    weight = 0
            if idx == len(idx_ladder)-2:
                weight = 1
                if itr >= 250 or itr <100:
                    weight = 0
            if idx == 0:
                weight = 1
                if itr >= 100:
                    weight = 0
        else:
            weight = 1

        w = w_ladder[idx] * weight

        anneal = (1 - 1 / T)
        loss2 = log_probz * anneal - log_probq
        idx_loss = loss2 + kl_dive / T

        loss = loss - 1 * (idx_loss.mean()) * w

    #if args.AIS:
    #    if itr > 201:
    #        logw = idx_loss
    #        w = torch.exp(logw - torch.max(logw))
    #        w_norm = w
    #        w_norm = PSIS_sampling(w.clone().detach().cpu().numpy())
    #        w_norm = torch.tensor(data=w_norm, device=args.device)
    #        w_norm = w_norm / torch.sum(w_norm)
    #        loss = w_norm * (logw)
    #        loss = -1 * loss.mean() * 20

    nll = kl_dive
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




