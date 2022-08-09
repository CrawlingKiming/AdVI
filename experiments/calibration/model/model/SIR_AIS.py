import math

import torch

from survae.distributions import StandardNormal
from survae.transforms import Sigmoid
from .SIR_Model import ODE_Solver
from .SIR_Model import SIR
from .subclass.InverseFlow import SqFlow, MSIR_Flow, Build_Shuffle_Order_Transform, Build_Shuffle_Transform, Build_Shuffle_Order_3_Transform
from .subclass.diagnostic import PSIS_sampling
from .subclass.layers import ShiftBijection, ScaleBijection
from .subclass.surjective import BoundSurjection, BoundSurjection_S


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

    transforms += [ScaleBijection(scale=torch.tensor([[1.0, 1.0, 1.0]]))]
    transforms += [BoundSurjection()]
    transforms += [ShiftBijection(shift=torch.tensor([[1.5, 1.5, 1.5]]))]
    transforms += [ScaleBijection(scale=torch.tensor([[1/3, 1/3, 1/3]]))]
    transforms += [ScaleBijection(scale=torch.tensor([[40.0, 3.0, 1.0]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[80.0, 0., 0.]])))
    return transforms

def Sig_T():
    transforms = []
    transforms += [Sigmoid()]
    transforms += [ScaleBijection(scale=torch.tensor([[40.0, 3.0, 1.0]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[80.0, 0., 0.]])))
    return transforms

class CanModel(torch.nn.Module):

    def __init__(self, idx_ladder, data_shape, pretrained=None, bound_q=None):
        super(CanModel, self).__init__()

        self.pretrained = pretrained
        self.bound_q = bound_q

        self.idx_ladder = idx_ladder
        self.base_dist = StandardNormal((data_shape,))

    def forward(self, num_samples, model, plain=False, param=False):
        """
        Recovering is True, if you want to plane discriminate
        log_probq : log det
        log_probp : base dist
        """
        ladder = self.idx_ladder
        x = self.base_dist.sample(num_samples = num_samples)

        log_probq_ls, log_probz_ls, z_ls = model.log_prob(x, ladder)
        z_concat = torch.cat(z_ls, 0) # (L*B) * D
        log_probp_ls = torch.chunk(torch.log(3 * torch.ones(size=(z_concat.shape[0],),
                                                        device=z_concat.device) / (8 * 4**6)),len(self.idx_ladder), dim=0)
        if self.bound_q :
            param = False
            if param:
                print("Param Sampling")
                z_concat, entropy = self.bound_t(z_concat)
                print(z_concat.shape)
                entropy_ls = torch.chunk(entropy, len(self.idx_ladder), dim=0)

            else :
                z_concat, entropy = self.bound_q(z_concat)
                entropy_ls = torch.chunk(entropy, len(self.idx_ladder), dim=0)
                log_probp_ls = entropy_ls
                log_probq_ls = list(map(lambda x, y: x - y, log_probq_ls, log_probp_ls))

        log_probp_transformed, _, sampled_x = self.pretrained.log_prob(x=z_concat)
        log_probq_ls = list(map(lambda x, y: x - y, log_probq_ls, torch.chunk(log_probp_transformed, len(self.idx_ladder), dim=0)))
        sampled_x_ls = torch.chunk(sampled_x, len(self.idx_ladder), dim=0)

        return sampled_x_ls, log_probq_ls, log_probp_ls, log_probz_ls

def build_SIR_model(args, data_shape):
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    D = data_shape

    bound_S = BoundSurjection_S()

    if args.bound_surjection:
        pretrained_transforms = Bound_T()
    else :
        pretrained_transforms = Sig_T()

    NormalFlow = MSIR_Flow(base_dist=StandardNormal((D,)),
                           transforms=pretrained_transforms).to(args.device)

    if args.AIS:
        model1_transforms1 = Build_Shuffle_Order_3_Transform(D=D, num_flows=args.num_flows, steps=10)#Build_Shuffle_Transform(D=D, num_flows=args.num_flows)#Build_Shuffle_Order_Transform(D=D, num_flows=args.num_flows, steps=2)#Build_Shuffle_Unif_Transform(D=D, num_flows=args.num_flows)#Build_Shuffle_Transform(D=D, num_flows=args.num_flows)
    else:
        model1_transforms1 = Build_Shuffle_Order_Transform(D=D, num_flows=args.num_flows, steps=1)#Build_Shuffle_Transform(D=D, num_flows=args.num_flows)
    #print(model1_transforms1)
    if args.bound_surjection:
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder
                        ,bound_q=bound_S).to(args.device)

    else:
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder, bound_q=None).to(args.device)
    model1 = SqFlow(base_dist=StandardNormal((D,)), transforms=model1_transforms1).to(args.device)

    return mycan, model1

def get_SIR_loss(can_model, model, observation, args, itr, eval = False, recover= False):

    mycan = can_model
    truex = observation
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    #print(truex)
    sampled_x_ls, log_probq_ls, log_probp_ls, z_ls = mycan(num_samples = args.batch_size, model = model)

    if eval:
        sampled_x_ls, log_probq_ls, log_probp_ls, z_ls = mycan(num_samples=args.eval_size, model=model)
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
    if eval:
        samples = torch.chunk(samples, len(idx_ladder), dim=0)
        return samples[-1]

    #print(samples.shape)
    #print(gt.shape)

    gt = gt.unsqueeze(0).repeat(B,1,1)
    #print(samples.shape)

    p = torch.square(samples[:,1:,1:] - gt[:,1:,1:])
    #p = torch.square(samples[:, 2:, [-1]] - gt[:, 2:, [-1]])
    #p = torch.square(samples[:,[17],1:] - gt[:,[17],1:])

    e = 0.1
    #p = (p/(e**2))
    p = torch.sum(p, dim=1)
    #print(p.shape)


    base = -1 * math.log(e * math.sqrt(math.pi * 2))
    kl_dive = -1 * (0.5 * p / (e**2)) + base
    #########################################
    #log_probp = log_probp_ls[0]
    #log_probq = log_probq_ls[0]
    #print(kl_dive.shape)
    #print(B)
    #loss = -log_probq + kl_dive #+ log_probp /3
    #loss = -1 * loss.mean()
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

        loss2 = log_probp * (2 - 1/T) - log_probq #* (2 - 1 / T)
        idx_loss = loss2 + kl_dive / T
        if args.AIS :
            if itr < 60 and idx == len(idx_ladder) -1:
                weight = 0
                w = w * weight

            if itr >= 60 and idx == 0 :
                weight = 0
                w = w * weight
        loss = loss - 1 * (idx_loss.mean()) * w

    if args.AIS :

        if itr >= 60:
            logw = idx_loss
            w = torch.exp(logw - torch.max(logw))
            w_norm = w
            w_norm = PSIS_sampling(w.clone().detach().cpu().numpy())
            #w_norm = IS_truncation(w)
            w_norm = torch.tensor(data=w_norm, device=args.device)
            w_norm = w_norm / torch.sum(w_norm)
            loss = w_norm * (logw)
            #print(loss.mean())
            loss = -1 * loss.mean()

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


