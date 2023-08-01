import numpy as np
import torch

from survae.distributions import StandardNormal
from survae.transforms import Sigmoid
from .SEIR_fullModel import ODE_Solver as ODE_Solver
from .SEIR_fullModel import SEIR as SEIR
from .subclass.InverseFlow import SqFlow, MSIR_Flow, Build_Shuffle_Transform, Build_Shuffle_Order_5_Transform, Build_Auto_Transform, Build_Spline_Transform, GaussianModel, Build_Spline_Order_5_Transform
from .subclass.layers import ShiftBijection, ScaleBijection
from .subclass.surjective import BoundSurjection_S
from .subclass.diagnostic import PSIS_sampling

def plain_convert(x):
    raise NotImplementedError

def get_ladder(args):
    if args.AIS:
        idx_ladder = [args.num_flows//2, args.num_flows]
        temp_ladder = [args.temp, 1.]
        w_ladder_ls = [[2, 1]]

    else:
        idx_ladder = [args.num_flows]
        temp_ladder = [1]
        w_ladder_ls = [[1]]

    assert len(idx_ladder) == len(temp_ladder)

    return idx_ladder, temp_ladder, w_ladder_ls

def Bound_T():
    transforms=[]

    transforms += [ShiftBijection(shift=torch.tensor([[1.5, 1.5, 1.5, 1.5, 1.5]]))]
    transforms += [ScaleBijection(scale=torch.tensor([[1/3, 1/3, 1/3, 1/3, 1/3]]))]
    transforms += [ScaleBijection(scale=torch.tensor([[4.5 * 1e-6, 160*1e3, 0.001, 0.6, 0.3]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[0.5 * 1e-6, 140*1e3, 0, 0.8, 0.09]])))

    return transforms

def Sig_T():
    transforms=[]
    transforms += [Sigmoid()]
    transforms += [ScaleBijection(scale=torch.tensor([[4.5 * 1e-6, 160*1e3, 0.001, 0.6, 0.3]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[0.5 * 1e-6, 140*1e3, 0, 0.8, 0.09]])))

    return transforms

class CanModel(torch.nn.Module):

    def __init__(self, idx_ladder, data_shape, pretrained=None, bound_q=None):
        super(CanModel, self).__init__()

        self.pretrained = pretrained
        self.bound_q = bound_q

        self.idx_ladder = idx_ladder
        self.base_dist = StandardNormal((data_shape,))
        #print("load")

    def forward(self, num_samples, model, plain=False, param=False):
        """
        Recovering is True, if you want to plane discriminate
        log_probq : log det
        log_probp : base dist
        """
        ladder = self.idx_ladder
        x = self.base_dist.sample(num_samples = num_samples)

        log_probq_ls, log_probz_ls, z_ls = model.log_prob(x, ladder)
        #print(log_probq_ls)
        z_concat = torch.cat(z_ls, 0) # (L*B) * D

        log_probp_ls = torch.chunk(torch.log(3 * torch.ones(size=(z_concat.shape[0],),
                                                        device=z_concat.device) / (5 * 4**6)),len(self.idx_ladder), dim=0)
        #np.save("BFAF", temp)
        if self.bound_q :
            z_concat, entropy = self.bound_q(z_concat)
            entropy_ls = torch.chunk(entropy, len(self.idx_ladder), dim=0)
            log_probp_ls = entropy_ls
            log_probq_ls = list(map(lambda x, y: x - y, log_probq_ls, log_probp_ls))

        log_probp_transformed, _, sampled_x = self.pretrained.log_prob(x=z_concat)
        log_probq_ls = list(map(lambda x, y: x - y, log_probq_ls, torch.chunk(log_probp_transformed, len(self.idx_ladder), dim=0)))
        sampled_x_ls = torch.chunk(sampled_x, len(self.idx_ladder), dim=0)

        temp = sampled_x_ls[-1].detach().cpu().numpy()
        ## Used for plotting..
        #np.save("const_BFAF", temp)
        #####################

        return sampled_x_ls, log_probq_ls, log_probz_ls, log_probp_ls


def build_SEIR_model(args, data_shape):
    #raise NotImplementedError
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    D = data_shape
    #print(D)
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
    #elif args.AIS:
    #    model1_transforms1 = Build_Shuffle_Order_5_Transform(D=D, num_flows=args.num_flows, steps=10)
    else:
    #    model1_transforms1 = Build_Shuffle_Transform(D=D, num_flows=args.num_flows)
        model1_transforms1 = Build_Spline_Order_5_Transform(D=D, num_bin=8, num_flows=args.num_flows)
        #model1_transforms1 = Build_Auto_Transform(D=D, num_flows=args.num_flows)

        ladd_step = 2

    #model1_transforms1 = Build_Auto_Transform(D=D, num_flows=args.num_flows)

    if args.bound_surjection:
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder
                        ,bound_q=bound_S).to(args.device)
    else:
        print("Non BS")
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder, bound_q=None).to(args.device)
    model1 = SqFlow(base_dist=StandardNormal((D,)), transforms=model1_transforms1,ladd_step=ladd_step).to(args.device)

    return mycan, model1

def get_SEIR_loss(can_model, model, observation, args, itr, eval = False, recover= False):

    k_diag = 0

    mycan = can_model
    # Observations
    truex = observation
    gt = truex.clone()
    gt = gt.to(args.device)
    gt = gt[:, -1]

    age_perc = torch.tensor([[0.42, 0.30, 0.28]], device=args.device)

    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    sampled_x_ls, log_probq_ls, log_probp_ls, z_ls = mycan(num_samples = args.batch_size, model = model)
    if eval:
        sampled_x_ls, log_probq_ls, log_probp_ls, z_ls = mycan(num_samples=args.eval_size, model=model)
    if recover :
        sampled_x_ls, _, _, _ = mycan(num_samples = args.batch_size, model = model, plain = True)

    sampled_x = torch.cat(sampled_x_ls, 0)
    if recover :
        sampled_x = sampled_x.to(args.device)

    x = sampled_x
    # term = 54 * 7, 54 weeks

    term = 53

    tt = torch.linspace(0, term * 7, steps=term+1, requires_grad=False).to(args.device)
    #print(tt.shape)
    #print(tt)
    #tt = torch.cat((torch.tensor([0.0], device=args.device), tt))

    mySEIR = SEIR(args=args).to(args.device)

    observed_y = ODE_Solver(x, tt, mySEIR, args)
    z_samples = observed_y  # [:, -1:, :]
    z_samples = torch.transpose(z_samples, 0, 1) # B * time(54) * 84
    z_samples = z_samples[:, -53:, :]
    n_age = 21
    q = 2

    I = z_samples[:, :, q*n_age:(q+1)*n_age] # B * 53(weeks) * n_age
    #print(I)
    #print(I.shape)
    I_w = torch.sum(I, dim=2) # B * 54(weeks)
    #print(I_w.shape)
    # 1:5, 6:14, 15:21
    I_a1 = torch.sum(torch.sum(I[:, :, 0:5], dim=2).unsqueeze(-1), dim=1)
    I_a2 = torch.sum(torch.sum(I[:, :, 5:14], dim=2).unsqueeze(-1), dim=1)
    I_a3 = torch.sum(torch.sum(I[:, :, 14:], dim=2).unsqueeze(-1), dim=1) # B * 1

    I_age = torch.cat((I_a1, I_a2, I_a3), dim=1) # B * 3

    #print(I_a1.shape, I_age.shape)
    I_age_sum = torch.sum(I_age, dim=1).unsqueeze(-1).repeat(1, 3)
    I_age_prop = torch.div(I_age, I_age_sum)
    #print(samples_age_prop.shape)

    B = I_w.shape[0]
    #r = x[:, [5]]#torch.tensor([[2]]).repeat(B,1)#torch.tensor([[1.5]]).repeat(B,1).to(args.device)#
    #print(r)
    #a0 = x[:, [6]]#torch.tensor([[225]]).repeat(B,1).to(args.device)#


    gt = gt.repeat(B, 1)
    age_perc = age_perc.repeat(B, 1)

    w_ladder = w_ladder_ls[k_diag]

    loss = 0.0
    #print(I_age_prop.shape)
    e1 = args.smoothing
    e2 = args.smoothing2

    m = torch.distributions.normal.Normal(gt, e1)#MultivariateNormal(gt, torch.diag(e1[0])[:,-1,-1].repeat(B,1))

    dir = torch.distributions.dirichlet.Dirichlet(concentration=e2 * I_age_prop)
    if eval:
        I_w_chunked = torch.chunk(I_w, len(idx_ladder), dim=0)
        I_age_prop_chunked = torch.chunk(I_age_prop, len(idx_ladder), dim=0)
        samples_Iw = I_w_chunked[-1].detach().cpu().numpy()
        samples_age_prop = I_age_prop_chunked[-1].detach().cpu().numpy()

        # print(samples_Iw[-1].shape)
        samples = (samples_Iw, samples_age_prop)
        return samples

    kl_dive1 = torch.sum(m.log_prob(I_w), dim=1)#-1 * (0.5 * p1 / (e1**2))#torch.sum(nb.log_prob(gt), dim=1)
    kl_dive2 = dir.log_prob(age_perc)
    #print(kl_dive1.shape, kl_dive2.shape)
    kl_dive = kl_dive1 + kl_dive2
    kl_dive_ls = torch.chunk(kl_dive, len(idx_ladder), dim=0)  # Per ladder, we get this one
    for idx in range(len(idx_ladder)):
        weight = 1
        log_probz = z_ls[idx]#log_probp_ls[idx]
        log_probq = log_probq_ls[idx]
        kl_dive = kl_dive_ls[idx]
        T = temp_ladder[idx]
        w = w_ladder[idx]
        annealed = 1 - 1/T

        loss2 = log_probz * annealed - log_probq
        idx_loss = loss2 + kl_dive/ T
        if args.AIS :
            if itr < 100 and idx == len(idx_ladder) -1:
                weight = 0
                w = w * weight

            if itr >= 100 -1 and idx == 0 :
                weight = 0
                w = w * weight

            if itr > 101:
                logw = idx_loss
                w = torch.exp(logw - torch.max(logw))
                w_norm = w
                w_norm = PSIS_sampling(w.clone().detach().cpu().numpy())
                w_norm = torch.tensor(data=w_norm, device=args.device)
                w_norm = w_norm / torch.sum(w_norm)
                loss = w_norm * (logw)
                loss = -1 * loss.mean() * 20
        else:
            loss = loss - 1 * (idx_loss.mean()) * w
        print(loss)
    nll = kl_dive

    null = 0
    return loss, nll, null



def only_forward(param_sample, args):
    sampled_x = torch.tensor(data=param_sample, device=args.device).float()

    x = sampled_x

    # term = 54 * 7, 54 weeks
    term = 53
    tt = torch.linspace(0, term * 7, steps=term+1, requires_grad=False).to(args.device)
    #print(tt.shape)
    #print(tt)
    #tt = torch.cat((torch.tensor([0.0], device=args.device), tt))

    mySEIR = SEIR(args=args).to(args.device).double()

    observed_y = ODE_Solver(x, tt, mySEIR, args)
    z_samples = observed_y  # [:, -1:, :]
    z_samples = torch.transpose(z_samples, 0, 1) # B * time(54) * 84
    z_samples = z_samples[:, -53:, :]
    n_age = 21
    q = 2

    I = z_samples[:, :, q*n_age:(q+1)*n_age] # B * 53(weeks) * n_age
    #print(I.shape)
    I_w = torch.sum(I, dim=2) # B * 54(weeks)
    #print(I_w.shape)
    # 1:5, 6:14, 15:21
    I_a1 = torch.sum(torch.sum(I[:, :, 0:5], dim=2).unsqueeze(-1), dim=1)
    I_a2 = torch.sum(torch.sum(I[:, :, 5:14], dim=2).unsqueeze(-1), dim=1)
    I_a3 = torch.sum(torch.sum(I[:, :, 14:], dim=2).unsqueeze(-1), dim=1) # B * 1

    I_age = torch.cat((I_a1, I_a2, I_a3), dim=1) # B * 3
    #print(I_a1.shape, I_age.shape)
    I_age_sum = torch.sum(I_age, dim=1).unsqueeze(-1).repeat(1, 3)
    I_age_prop = torch.div(I_age, I_age_sum)

    #print(I_age_prop.shape) # B * 3
    #r = x[:, [5]]#torch.tensor([[2]]).repeat(B,1)#torch.tensor([[1.5]]).repeat(B,1).to(args.device)#
    #print(r)
    #a0 = x[:, [6]]
    samples_Iw = I_w#I_w_chunked[-1].detach().cpu().numpy()
    samples_age_prop = I_age_prop#I_age_prop_chunked[-1].detach().cpu().numpy()
    #new_r = r.repeat(1, 53) # B , 53
    #new_a0 = a0.repeat(1, 53) # B, 53
    #print(new_r)
    #print(I_w_prob)

    #I_w = torch.clamp(I_w, min=0.0)
    samples = (samples_Iw, samples_age_prop)
    return samples




