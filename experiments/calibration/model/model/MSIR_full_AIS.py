import numpy as np
import torch

from MSIR_fullModel import MSIR as MSIR
from MSIR_fullModel import ODE_Solver as ODE_Solver
from survae.distributions import StandardNormal
from survae.transforms import Sigmoid
from .subclass.InverseFlow import SqFlow, MSIR_Flow, Build_Shuffle_Order_Transform
from .subclass.layers import ShiftBijection, ScaleBijection
from .subclass.surjective import BoundSurjection, BoundSurjection_S, BoundSurjection_M, BoundSurjection_Max, BoundSurjection_Max_sample
from .subclass.diagnostic import PSIS, PSIS_sampling, IS_truncation

def plain_convert(x):
    #print(x)
    #raise ValueError
    #x[:, [8]] = 0.2 * x[:, [8]]
    b = x[:, [0]]
    phi = x[:, [1]]
    rho = x[:, [2]]
    r = x[:, [3]]
    b1 = x[:, [4]]
    b2 = x[:, [5]]
    b3 = x[:, [6]]
    b4 = x[:, [7]]
    b5 = x[:, [8]]
    b6 = x[:, [9]]
    #beta = beta * 4
    b5 = b5 - 0.4

    x = torch.cat((b, phi, rho, r, b1, b2, b3, b4, b5, b6), 1)

    x = torch.sigmoid(x)
    b = x[:, [0]]
    phi = x[:, [1]]
    rho = x[:, [2]] * 0.1
    r = x[:, [3]]
    beta = x[:, 4:]

    beta = beta * 4
    phi = phi * 2 * 3.14
    phi = phi + 2

    x = torch.cat((b, phi, rho, r, beta), 1)

    B = x.shape[0]
    x = torch.tensor(([[0.43, 7.35, 0.027,0.90, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381]]))
    x = x.repeat(B,1)

    return x



def get_ladder(args):
    if args.AIS:
        idx_ladder = [10, 15, 20]#[,args.num_flows // 2, args.num_flows]
        temp_ladder = [9, 2.5, 1]#16#[9, 6, 3.5, 1]#[args.temp, 1]#[12,6,3,1]
        w_ladder_ls = [[3, 1, 1]]
        args.w_ladder = w_ladder_ls
    else:
        idx_ladder = [args.num_flows]
        temp_ladder = [1]
        w_ladder_ls = [[1]]

    assert len(idx_ladder) == len(temp_ladder)

    return idx_ladder, temp_ladder, w_ladder_ls

def Bound_T():
    transforms = []

    #transforms += [ScaleBijection(scale=torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]))]
    #transforms += [BoundSurjection()]
    transforms += [ShiftBijection(shift=torch.tensor([[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]))]
    transforms += [ScaleBijection(scale=torch.tensor([[1/3, 1/3, 1/3, 1/3, 1/3,
                                                       1/3, 1/3, 1/3, 1/3, 1/3]]))]
    transforms += [ScaleBijection(scale=torch.tensor([[1, (2*3.14), 1/10, 1, 4, 4, 4, 4, 4, 4]]))]
    transforms += [ShiftBijection(shift=torch.tensor([[0.0, 2, 0., 0.0, 0.0, 0, 0, 0, 0.0, 0.0]]))]

    return transforms

def Sig_T():
    transforms=[]
    transforms += [Sigmoid()]
    transforms += [ScaleBijection(scale=torch.tensor([[1, (2*3.14), 1/10, 1.0, 4, 4, 4, 4, 4, 4]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[0.0, 2, 0., 0.0, 0, 0, 0, 0, 0, 0]])))

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
        log_probp_ls = torch.chunk(torch.log(10 * torch.ones(size=(z_concat.shape[0],),
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

        if plain:
            for z in z_ls :
                    sampled_x_ls = []
                    sampled_x = plain_convert(z)
                    sampled_x_ls.append(sampled_x)

                #sampled_x_ls.append(sampled_x)

        return sampled_x_ls, log_probq_ls, log_probp_ls, log_probz_ls


#optimizer1 = Adam(list(model1.parameters()), lr=args.lr)
#optimizer2 = Adamax(list(model2.parameters()), lr=args.lr)

def build_MSIR_model(args, data_shape):
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    D = data_shape

    bound_S = BoundSurjection_S()

    if args.bound_surjection:
        pretrained_transforms = Bound_T()

    else :
        pretrained_transforms = Sig_T()

    NormalFlow = MSIR_Flow(base_dist=StandardNormal((D,)),
                               transforms=pretrained_transforms).to(args.device)

    model1_transforms1 = Build_Shuffle_Order_Transform(D=D, num_flows=args.num_flows, steps=2)
    if args.bound_surjection:
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder
                        ,bound_q=bound_S).to(args.device)

    else:
        mycan = CanModel(pretrained=NormalFlow, data_shape=D, idx_ladder=idx_ladder).to(args.device)

    model1 = SqFlow(base_dist=StandardNormal((D,)), transforms=model1_transforms1).to(args.device)

    return mycan, model1

def get_MSIR_loss(can_model, model, observation, args, itr, eval = False, recover= False):
    #print("MSIR")
    k_diag = 0

    mycan = can_model
    truex = observation
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)

    sampled_x_ls, log_probq_ls, log_probp_ls, z_ls = mycan(num_samples = args.batch_size, model = model)
    if eval:
        sampled_x_ls, log_probq_ls, log_probp_ls, z_ls = mycan(num_samples=args.eval_size, model=model)
    #if recover :
    #    sampled_x_ls, _, _, _ = mycan(num_samples = args.batch_size, model = model, plain = True)
        #sampled_x_ls = sampled_x_ls.to(args.device)
    #if i == 0:
    #    sampled_x_ls, log_probq_ls, log_probp_ls, z_ls = mycan(model = model model, plain=True, ladder=idx_ladder)
    #print(sampled_x_ls[0])
    sampled_x = torch.cat(sampled_x_ls, 0)
    gt = truex.clone()
    gt = gt.to(args.device)
    x = sampled_x

    # term = 20 * 52 + 118
    term = 5 * 52 + 118
    tt = torch.linspace(0, term, steps=term + 1, requires_grad=False).to(args.device)
    tt = torch.cat((torch.tensor([0.0], device=args.device), tt[-118:]))
    mySIR = MSIR(args=args).to(args.device)

    observed_y = ODE_Solver(x, tt, mySIR, args)
    z_samples = observed_y  # [:, -1:, :]
    z_samples = torch.transpose(z_samples, 0, 1)
    b = x[:, [0]]
    phi = x[:, [1]]
    rho = x[:, [2]]
    r = x[:, [3]]  # B * 1
    beta = x[:, 4:]

    # print(z_samples.shape)
    # M = z_samples[:, -118:, 0:6]
    S = z_samples[:, -118:, 6:12]
    Is = z_samples[:, -118:, 12:18]
    Im = z_samples[:, -118:, 18:24]
    N = 324935

    B = x.shape[0]
    t = tt[-118:].unsqueeze(0).repeat(B, 1)  # B*t
    func = mySIR
    beta_t_pre = beta.unsqueeze(-1) * func.contact.unsqueeze(0).repeat(B, 1, 1)  # B*I*1 X B*I*I I * I
    beta_cos = 1 + b * torch.cos((2 * torch.tensor(np.pi) * t - 52 * phi) / 52)  # B * t
    beta_cos = beta_cos.unsqueeze(2).unsqueeze(3).repeat(1, 1, 6, 6)
    beta_t_pre = beta_t_pre.unsqueeze(1)
    beta_t = beta_t_pre * beta_cos  # B * t * I * I)
    Y = torch.div((Is + 0.5 * Im), (N * func.frac[None, :]))  # B * I  * 1 * I = B* t * I
    Y = Y.reshape(B * 118, 6)
    beta_t = beta_t.reshape(B * 118, 6, 6)
    lambd = torch.bmm(beta_t, Y.unsqueeze(-1))
    lambd = lambd.reshape(B, 118, 6)


    # z_samples
    cases_age = 0.24 * lambd * S  # B * t * I
    for j in range(B):
        cases_age[j, :, :] *= rho[j]

    # print(cases_age.shape) # B * t * I
    #cases1 = torch.sum(cases_age[:, :, 0: 3], dim=2).unsqueeze(2)
    #cases2 = torch.sum(cases_age[:, :, [3]], dim=2).unsqueeze(2)
    #cases3 = torch.sum(cases_age[:, :, 4: 6], dim=2).unsqueeze(2)
    #cases_fit = torch.cat((cases1, cases2, cases3), 2)  # B * t * 3
    cases_fit = cases_age

    cases_fit_chunked = torch.chunk(cases_fit, len(idx_ladder), dim=0)
    samples = cases_fit_chunked[-1].detach().cpu().numpy()

    for j in range(B):
        cases_fit[j, :, :] /= r[j]
    # cases_fit : B * T * 3
    #cases_sum = cases_fit.sum(dim=-1)

    #new_r = r.repeat(1, t.shape[1] * 3)
    new_r = r.repeat(1, t.shape[1] * 6)
    new_r = new_r.view(B, t.shape[1], 6)
    #new_r = new_r.view(B, t.shape[1], 3)
    cases_prob = cases_fit / (1 + cases_fit)

    #cases_prob = torch.clamp(cases_prob, min=0.0, max=1.0)
    #print(cases_prob)
    gt = gt.repeat(B, 1, 1)
    #gt_sum = gt.sum(dim=-1)

    if recover :
        #print("recover1 ")
        cases_prob_ls = torch.chunk(cases_prob, len(idx_ladder), dim=0)
        new_r_ls = torch.chunk(new_r, len(idx_ladder), dim=0)

        new_r = new_r_ls[-1]
        cases_prob = cases_prob_ls[-1]

        return samples, cases_prob, new_r
    #print(cases_prob.shape, gt.shape)
    #raise ValueError
    nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=new_r, probs=cases_prob)
    kl_dive = nb.log_prob(gt)
    kl_dive = kl_dive.view(-1, 118 * 6)
    #kl_dive = kl_dive.view(-1, 118 * 6)
    #print(new_r.shape)
    #########################################

    kl_dive_con = torch.sum(kl_dive, dim=1)
    kl_dive_ls = torch.chunk(kl_dive_con, len(idx_ladder), dim=0) # Per ladder, we get this one
    #sampled_x_ls = torch.chunk(sampled_x, len(idx_ladder), dim=0)

    w_ladder = w_ladder_ls[0]

    loss = 0.0

    PSIS_diagnos = []
    for idx in range(len(idx_ladder)):
        log_probz = z_ls[idx]
        log_probq = log_probq_ls[idx]
        contribution = log_probp_ls[idx]
        kl_dive = kl_dive_ls[idx]
        T = temp_ladder[idx]
        weight = 1
        if args.bound_surjection:
            if idx == len(idx_ladder)-1:
                weight = 1
                if itr < 301 :
                    weight = 0
            else :
                if idx == 1:
                    if itr < 225:
                        weight = 0
                    if itr > 300:
                        weight = 0.00
                elif itr > 225 :
                    weight = 0.00

                #if idx == 2 :
                #    if itr > 301:
                #        weight = 0.0

                #    elif itr <249:
                #        weight = 0.0
                #    else :
                #        weight = 1.0

        else :
            weight = 1
        #print("recover2")
        #if idx > 1 and itr < 200:
        #    weight = 0
        w = w_ladder[idx] * weight

        if args.bound_surjection:
            anneal = (1 - 1 / T)
            if anneal < 0 :
                anneal = 0.0
            loss2 = (log_probz) * anneal - log_probq
            if itr > 301:
                loss2 = loss2 - contribution


        else :
            loss2 = (log_probz) * (1 - 1 / T) - log_probq

        idx_loss = loss2 + kl_dive / T

        loss = loss - 1 * (idx_loss.mean()) * w

    if eval and not recover:
        return samples, PSIS_diagnos

    if itr > 301:
        logw = idx_loss
        w = torch.exp(logw - torch.max(logw))
        w_norm = w
        w_norm = PSIS_sampling(w.clone().detach().cpu().numpy())
        #w_norm = IS_truncation(w)
        w_norm = torch.tensor(data=w_norm, device=args.device)
        w_norm = w_norm / torch.sum(w_norm)
        loss = w_norm * (logw)
        #print(loss.mean())
        loss = -1 * loss.mean() * 20
        #print(loss)

    nll = kl_dive

    return loss, nll, samples


def only_forward(param_sample, args):
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)

    sampled_x = torch.tensor(data=param_sample, device=args.device)
    x = sampled_x


    # term = 20 * 52 + 118
    term = 5 * 52 + 118
    tt = torch.linspace(0, term, steps=term + 1, requires_grad=False).to(args.device)
    tt = torch.cat((torch.tensor([0.0], device=args.device), tt[-118:]))
    mySIR = MSIR(args=args).to(args.device)

    observed_y = ODE_Solver(x, tt, mySIR, args)
    z_samples = observed_y  # [:, -1:, :]
    z_samples = torch.transpose(z_samples, 0, 1)
    b = x[:, [0]]
    phi = x[:, [1]]
    rho = x[:, [2]]
    r = x[:, [3]]  # B * 1
    beta = x[:, 4:]

    # print(z_samples.shape)
    M = z_samples[:, -118:, 0:6]
    S = z_samples[:, -118:, 6:12]
    Is = z_samples[:, -118:, 12:18]
    Im = z_samples[:, -118:, 18:24]
    N = 324935

    B = x.shape[0]
    t = tt[-118:].unsqueeze(0).repeat(B, 1)  # B*t
    func = mySIR
    beta_t_pre = beta.unsqueeze(-1) * func.contact.unsqueeze(0).repeat(B, 1, 1)  # B*I*1 X B*I*I I * I
    beta_cos = 1 + b * torch.cos((2 * torch.tensor(np.pi) * t - 52 * phi) / 52)  # B * t
    beta_cos = beta_cos.unsqueeze(2).unsqueeze(3).repeat(1, 1, 6, 6)
    beta_t_pre = beta_t_pre.unsqueeze(1)
    beta_t = beta_t_pre * beta_cos  # B * t * I * I)
    Y = torch.div((Is + 0.5 * Im), (N * func.frac[None, :]))  # B * I  * 1 * I = B* t * I
    Y = Y.reshape(B * 118, 6)
    beta_t = beta_t.reshape(B * 118, 6, 6)
    lambd = torch.bmm(beta_t, Y.unsqueeze(-1))
    lambd = lambd.reshape(B, 118, 6)

    cases_age = 0.24 * lambd * S  # B * t * I
    for j in range(B):
        cases_age[j, :, :] *= rho[j]

    # print(cases_age.shape) # B * t * I
    #cases1 = torch.sum(cases_age[:, :, 0: 3], dim=2).unsqueeze(2)
    #cases2 = torch.sum(cases_age[:, :, [3]], dim=2).unsqueeze(2)
    #cases3 = torch.sum(cases_age[:, :, 4: 6], dim=2).unsqueeze(2)
    #cases_fit = torch.cat((cases1, cases2, cases3), 2)  # B * t * 3
    cases_fit = cases_age

    cases_fit_chunked = torch.chunk(cases_fit, len(idx_ladder), dim=0)
    samples = cases_fit_chunked[-1].detach().cpu().numpy()

    for j in range(B):
        cases_fit[j, :, :] /= r[j]

    #new_r = r.repeat(1, t.shape[1] * 3)

    #new_r = new_r.view(B, t.shape[1], 3)
    new_r = r.repeat(1, t.shape[1] * 6)
    new_r = new_r.view(B, t.shape[1], 6)
    cases_prob = cases_fit / (1 + cases_fit)

    cases_prob_ls = torch.chunk(cases_prob, len(idx_ladder), dim=0)
    new_r_ls = torch.chunk(new_r, len(idx_ladder), dim=0)

    new_r = new_r_ls[-1]
    cases_prob = cases_prob_ls[-1]

    return samples, cases_prob, new_r




