import numpy as np
import torch
import hamiltorch 
import math 
from MSIR_fullModel import MSIR as MSIR
from MSIR_fullModel import ODE_Solver as ODE_Solver
from survae.distributions import StandardNormal
from survae.transforms import Sigmoid, Logit
from .subclass.InverseFlow import SqFlow, MSIR_Flow, Build_Shuffle_Order_Transform, \
    GaussianModel, Build_Auto_Transform, Build_Spline_Transform
from .subclass.diagnostic import PSIS_sampling
from .subclass.layers import ShiftBijection, ScaleBijection
from .subclass.surjective import BoundSurjection_S
from torch.distributions.multivariate_normal import MultivariateNormal
import arviz as az

def plain_convert(x):
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
        idx_ladder = [args.num_flows//2, args.num_flows]
        temp_ladder = [args.temp, 1]
        w_ladder_ls = [[args.temp, 1]]#[[3, 1, 1]]
        args.w_ladder = w_ladder_ls
    else:
        idx_ladder = [args.num_flows]
        temp_ladder = [1]
        w_ladder_ls = [[1]]

    assert len(idx_ladder) == len(temp_ladder)

    return idx_ladder, temp_ladder, w_ladder_ls

def Bound_T(args):
    transforms = []

    transforms += [ShiftBijection(shift=torch.tensor([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]))]
    transforms += [ScaleBijection(scale=torch.tensor([[1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4,
                                                       1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4]]))]

    transforms += [ScaleBijection(scale=torch.tensor([[1.0, (2*3.14), 1/10, 1.0, 4, 4, 4, 4, 4, 4]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[0.0, 2.00, 0.0, 0.0, 0, 0, 0, 0, 0, 0]])))

    return transforms

def Sig_T(args):
    transforms=[]
    transforms += [Sigmoid()]

    transforms += [ScaleBijection(scale=torch.tensor([[1.0, (2*3.14), 1/10, 1.0, 4, 4, 4, 4, 4, 4]]))]
    transforms.append(ShiftBijection(shift=torch.tensor([[0.0, 2.00, 0.0, 0.0, 0, 0, 0, 0, 0, 0]])))

    return transforms

class CanModel(torch.nn.Module):

    def __init__(self, idx_ladder, data_shape, pretrained=None, bound_q=None, eval_bool=False):
        super(CanModel, self).__init__()
        """
        pretrained: Should be fixed(or pretrained) flow
        
        """
        self.z_intered = None
        #self.adj = torch.nn
        self.msc = False
        self.eval = eval_bool
        self.pretrained = pretrained
        self.bound_q = bound_q

        self.idx_ladder = idx_ladder
        self.base_dist = StandardNormal((data_shape,))
        self.x_temped = self.base_dist.sample(num_samples = 1)

        
    def _HMC(self, model, x, num_samples=1):
        ladder = self.idx_ladder
        #x = x * torch.tensor([[ -8.0308,  -1.4578,  -0.5492,  -5.3688,  -6.2853,   0.2353,   2.3576,
        #-11.4520,   5.7987,  24.2010]], device='cuda:0')
        #x = self.base_dist.sample(num_samples = num_samples)
        #self.x_temped = x
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
        logJ = -1 * log_probq_ls[-1] + self.base_dist.log_prob(x)
        return sampled_x_ls[-1], logJ

    def forward(self, num_samples, model, rej_const=False, param=False):
        """
        Recovering is True, if you want to plane discriminate
        log_probq : log det
        log_probp : base dist
        """
        ladder = self.idx_ladder
        x = self.base_dist.sample(num_samples = num_samples)
        #if self.eval:
        #    m = (x <= -2.0) | (x >= 2.0)
        #    x = x[~m.all(dim=1)]
        #z = self.base_dist.sample(num_samples = num_samples)
        #x = 1.5 * x
        #print(x.shape)
        #if rej_const:
        #    print("Rejection Sampling")
        #    x = rej_const * x
        #if self.msc:
        #    x = self.lower_tri2  * x
            #lower_fake_param = torch.zeros(x.shape[1], x.shape[1], device=x.device) + torch.eye(x.shape[1], device=x.device)
            #lower_fake_param[self.tril_ind[0, :], self.tril_ind[1, :]] = self.lower_tri.to(x.device)
            
            #lower_fake_param = lower_fake_param[None, :, :].repeat(x.shape[0], 1, 1)
            #final_dist = MultivariateNormal(loc=torch.zeros(size=((1,10)),device=x.device),scale_tril=lower_fake_param[0])  # Much more stable than standard version
            #logprob1 = self.base_dist.log_prob(x)

            #x = torch.matmul(lower_fake_param, x[:, :, None])
            #x = x[:,:,0]
            #x = torch.clamp(x, min=-2.0, max=2.0)
            #print("MSC")
            #x = self.x_temped 
            #x = 1.5 * x
            #x[0] = self.z_intered 
        
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
        #if self.msc:
        #    log_probq_ls[-1]= log_probq_ls[-1] -math.log(self.lower_tri2[0])#+ final_dist.log_prob(x) - logprob1

        return sampled_x_ls, log_probq_ls, log_probz_ls

def build_MSIR_model(args, data_shape):
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    D = data_shape

    bound_S = BoundSurjection_S()

    if args.bound_surjection:
        pretrained_transforms = Bound_T(args)

    else:
        pretrained_transforms = Sig_T(args)


    pretrained = MSIR_Flow(base_dist=StandardNormal((D,)),
                               transforms=pretrained_transforms).to(args.device)

    if args.gaussian:
        model1 = GaussianModel(StandardNormal((D,)),D)
        ladd_step = 1
        mycan = CanModel(pretrained=pretrained, data_shape=D, idx_ladder=idx_ladder, bound_q=None).to(args.device)
        return mycan, model1
    else:
        #model1_transforms1 = Build_Shuffle_Order_Transform(D=D, num_flows=args.num_flows, steps=2)
        #ladd_step = 3
        model1_transforms1 = Build_Spline_Transform(D=D, num_bin=16, num_flows=args.num_flows) #16 64
        #model1_transforms1 = Build_Auto_Transform(D=D, num_flows=args.num_flows)
        ladd_step = 3
    if args.bound_surjection:
        mycan = CanModel(pretrained=pretrained, data_shape=D, idx_ladder=idx_ladder
                        ,bound_q=bound_S).to(args.device)

    else:
        mycan = CanModel(pretrained=pretrained, data_shape=D, idx_ladder=idx_ladder).to(args.device)



    model1 = SqFlow(base_dist=StandardNormal((D,)), transforms=model1_transforms1,ladd_step=ladd_step).to(args.device)

    return mycan, model1

def temp(itr, max_itr,min_itr,N1, N2=1):
    if itr > max_itr:
        itr = max_itr 
    

    result = N1 + torch.arange(0, (N2 - N1) / step + 1) * step

def get_MSIR_loss(can_model, model, observation, args, itr, eval = False, recover= False):
    k_diag = 0

    mycan = can_model
    truex = observation
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)

    #mycan.msc = True  
    sampled_x_ls, log_probq_ls, log_probz_ls= mycan(num_samples = args.batch_size, model=model)
    if eval:
        sampled_x_ls, log_probq_ls,  log_probz_ls = mycan(num_samples=args.eval_size, model=model, rej_const=args.rej_const)

    sampled_x = torch.cat(sampled_x_ls, 0)
    gt = truex.clone()
    gt = gt.to(args.device)
    x = sampled_x

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

    cases_age = 0.24 * lambd * S  # B * t * I
    for j in range(B):
        cases_age[j, :, :] *= rho[j]

    cases_fit = cases_age
    new_r = r.repeat(1, t.shape[1] * 6)
    new_r = new_r.view(B, t.shape[1], 6)
    #print(new_r.shape, cases_fit.shape)
    assert new_r.shape == cases_fit.shape

    cases_fit_chunked = torch.chunk(cases_fit, len(idx_ladder), dim=0)
    #print(cases_fit_chunked.shape)
    samples = cases_fit_chunked[-1].detach().cpu().numpy()

    for j in range(B):
        cases_fit[j, :, :] /= r[j]

    new_r = r.repeat(1, t.shape[1] * 6)
    new_r = new_r.view(B, t.shape[1], 6)
    #####################################################################################
    #new_r = new_r[:,:,0:3]
    ###################################################################################
    cases_prob = cases_fit / (1 + cases_fit)

    gt = gt.repeat(B, 1, 1)

    """
    if recover :
        cases_prob_ls = torch.chunk(cases_prob, len(idx_ladder), dim=0)
        new_r_ls = torch.chunk(new_r, len(idx_ladder), dim=0)

        new_r = new_r_ls[-1]
        cases_prob = cases_prob_ls[-1]

        return samples, cases_prob, new_r
    """
    nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=new_r, probs=cases_prob)
    kl_dive = nb.log_prob(gt)
    ####################################################
    kl_dive = kl_dive.view(-1, 118 * 6)
    ###########################################################
    kl_dive_con = torch.sum(kl_dive, dim=1)
    kl_dive_ls = torch.chunk(kl_dive_con, len(idx_ladder), dim=0) # Per ladder, we get this one
    #sampled_x_ls = torch.chunk(sampled_x, len(idx_ladder), dim=0)

    w_ladder = w_ladder_ls[0]

    Pd = []


    ## Block for Blocks 
    if args.AIS : 
        if itr < 125: 
            Bl_ls = [0]
        else: 
            Bl_ls = [0, 1]
    else:
        Bl_ls = [0] 
    
    ratio_ls = []

    
    for Bl in Bl_ls:
        log_probz = log_probz_ls[Bl]
        log_probq = log_probq_ls[Bl]
        kl_dive = kl_dive_ls[Bl]
        #w = w_ladder[Bl]
        if Bl > 1:
            T = temp_ladder[Bl-1]
        else:
            T = temp_ladder[Bl]
        
        anneal = (1 - 1 / T)
        loss_fir = - (log_probq) + kl_dive / T + log_probz * anneal
        ratio_ls.append(loss_fir)


    if recover :
        cases_prob_ls = torch.chunk(cases_prob, len(idx_ladder), dim=0)
        new_r_ls = torch.chunk(new_r, len(idx_ladder), dim=0)
        new_r = new_r_ls[-1]
        cases_prob = cases_prob_ls[-1]
        x_ls = torch.chunk(x, len(idx_ladder), dim=0)
        new_x = x_ls[-1].clone().detach()
        return samples, cases_prob, new_r, new_x#, log_probq, kl_dive
 
        

    loss = 0.0 
    for ratio in ratio_ls:
        loss = loss + ratio 
    loss = -1 * loss.mean()
    nll = kl_dive

    if eval and not recover:
        return samples, Pd
    
    if args.AIS :
        if itr > 349:

            logw = -1 * loss_fir.clone().detach()
            logw_2, k = az.psislw(logw.clone().detach().cpu().numpy())
            #print(k)
            w = (logw_2)
            w = torch.tensor(data=w, device=args.device)

            logw = logw_2 
            w = torch.exp(w - torch.max(w))

            loss = -1 * w * loss_fir * 250 #torch.log(loss.mean())
            loss = loss.mean()
    
    return loss, nll, samples

class HMC_sampler(torch.nn.Module):
    def __init__(self, canmodel, model, args, gt, MSIR, solver, idx_ladder):
        super(HMC_sampler, self).__init__()
        self.args = args 
        self.gt = gt
        self.model2 = MSIR
        self.solver = solver 
        self.idx_ladder = idx_ladder    
        self.mycan = canmodel  
        self.model = model   

    def log_probp(self, x):
        args = self.args 
        gt = self.gt.to(args.device)
        MSIR = self.model2 
        ODE_Solver = self.solver  
        idx_ladder = self.idx_ladder
        
        #
        x = x[None, :]
        x = torch.clamp(x, max=2.5, min=-2.5)
        print(x)
        x, minuslogJ = self.mycan._HMC(self.model,x)
        #print(minuslogJ.shape)
        #temp = x[torch.abs(x) > 2.5] *0.8 
        #x[torch.abs(x) > 2.5] = temp  
        print(x)
        #print(x.shape)
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
        #print(x)
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

        cases_age = 0.24 * lambd * S  # B * t * I
        for j in range(B):
            cases_age[j, :, :] *= rho[j]

        cases_fit = cases_age


        new_r = r.repeat(1, t.shape[1] * 6)
        new_r = new_r.view(B, t.shape[1], 6)

        cases_fit_chunked = torch.chunk(cases_fit, len(idx_ladder), dim=0)
        #print(cases_fit_chunked.shape)
        samples = cases_fit_chunked[-1].detach().cpu().numpy()

        for j in range(B):
            cases_fit[j, :, :] /= r[j]

        new_r = r.repeat(1, t.shape[1] * 6)
        new_r = new_r.view(B, t.shape[1], 6)
        cases_prob = cases_fit / (1 + cases_fit)

        gt = gt.repeat(B, 1, 1)

        nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=new_r, probs=cases_prob)
        kl_dive = nb.log_prob(gt)
        ####################################################
        kl_dive = kl_dive.view(-1, 118 * 6)
        #kl_dive = kl_dive.view(-1, 118 * 3)
        ###########################################################
        kl_dive_con = torch.sum(kl_dive, dim=1)
        kl_dive_ls = torch.chunk(kl_dive_con, len(idx_ladder), dim=0) # Per ladder, we get this one
        res = kl_dive_ls[-1].sum() + minuslogJ
        return res


def get_Hessian(can_model, model, observation, args, itr, eval = False, recover= False):
    k_diag = 0
    mycan = can_model
    truex = observation
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    
    #sampled_x_ls, log_probq_ls, log_probz_ls= mycan(num_samples = args.batch_size, model=model)
    #if eval:
    #    sampled_x_ls, log_probq_ls,  log_probz_ls = mycan(num_samples=args.eval_size, model=model, rej_const=args.rej_const)

    #x = sampled_x_ls[-1]
    #print(x)
    gt = truex.clone()

    myHMC = HMC_sampler(mycan, model, args, gt, MSIR, ODE_Solver, idx_ladder)
    inputs = torch.zeros(size=(10,), device=args.device, requires_grad=True)
    #print(inputs)
    #y = myHMC.log_probp(inputs)
    #y.backward()
    #print(inputs.grad)
    hess = torch.autograd.functional.hessian(myHMC.log_probp, inputs, outer_jacobian_strategy='reverse-mode')
    
    L, V = torch.linalg.eigh(hess)
    print(L,V)
    trV = torch.clamp(V, min=0.0)
    print(L@trV)


def get_MSIR_HMC_loss(can_model, model, observation, args, itr, eval = False, recover= False):
    k_diag = 0
 
    # 

    mycan = can_model
    truex = observation
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)
    
    #sampled_x_ls, log_probq_ls, log_probz_ls= mycan(num_samples = args.batch_size, model=model)
    #if eval:
    #    sampled_x_ls, log_probq_ls,  log_probz_ls = mycan(num_samples=args.eval_size, model=model, rej_const=args.rej_const)

    #x = sampled_x_ls[-1]
    #print(x)
    gt = truex.clone()

    myHMC = HMC_sampler(mycan, model, args, gt, MSIR, ODE_Solver, idx_ladder)
    num_samples = 2
    step_size = .07
    num_steps_per_sample = 1

    #hamiltorch.set_random_seed(123)
    params_init = mycan.x_temped[0].to(args.device)
    #print(params_init)
    sampler=hamiltorch.Sampler.RMHMC
    integrator=hamiltorch.Integrator.IMPLICIT
    #params_irmhmc = hamiltorch.sample(log_prob_func=myHMC.log_probp, params_init=params_init, 
    #num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, sampler=sampler, integrator=integrator)
    params_hmc = hamiltorch.sample(log_prob_func=myHMC.log_probp, params_init=params_init, 
    num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample)
    #print(params_hmc)
    params_hmc = params_hmc[-1].clone().detach()
    mycan.x_temped = params_hmc[None, :]   
    #print(params_hmc)
    sampled_x_ls, log_probq_ls, log_probz_ls= mycan(num_samples = args.batch_size, model=model)
    sampled_x = torch.cat(sampled_x_ls, 0)

    gt = gt.to(args.device)
    x = sampled_x

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

    cases_age = 0.24 * lambd * S  # B * t * I
    for j in range(B):
        cases_age[j, :, :] *= rho[j]

    cases_fit = cases_age


    #####################################################################

    #sum_1_to_3 = torch.sum(cases_fit[:, :, 0:3], dim=2, keepdim=True)
    #sum_5_to_6 = torch.sum(cases_fit[:, :, 4:6], dim=2, keepdim=True)

    #cases_fit = torch.stack((sum_1_to_3, cases_fit[:,:, [3]], sum_5_to_6), dim=2)[:,:,:,0]
    #print(gt.shape)
    #raise ValueError
    #sum_1_to_3 = torch.sum(gt[:, :, 0:3], dim=2, keepdim=True)
    #sum_5_to_6 = torch.sum(gt[:, :, 5:6], dim=2, keepdim=True)

    #gt = torch.stack((sum_1_to_3, gt[:,:, [4]], sum_5_to_6), dim=2)[:,:,:,0]
    #new_r = r.repeat(1, t.shape[1] * 3)
    #new_r = new_r.view(B, t.shape[1], 3)

    #####################################################################################
    #new_r = new_r[:,:,0:3]
    new_r = r.repeat(1, t.shape[1] * 6)
    new_r = new_r.view(B, t.shape[1], 6)
    #print(new_r.shape, cases_fit.shape)
    assert new_r.shape == cases_fit.shape
    #raise ValueError
    #########################################################################


    cases_fit_chunked = torch.chunk(cases_fit, len(idx_ladder), dim=0)
    #print(cases_fit_chunked.shape)
    samples = cases_fit_chunked[-1].detach().cpu().numpy()

    for j in range(B):
        cases_fit[j, :, :] /= r[j]

    new_r = r.repeat(1, t.shape[1] * 6)
    new_r = new_r.view(B, t.shape[1], 6)
    #####################################################################################
    #new_r = new_r[:,:,0:3]
    ###################################################################################
    cases_prob = cases_fit / (1 + cases_fit)

    gt = gt.repeat(B, 1, 1)

    """
    if recover :
        cases_prob_ls = torch.chunk(cases_prob, len(idx_ladder), dim=0)
        new_r_ls = torch.chunk(new_r, len(idx_ladder), dim=0)

        new_r = new_r_ls[-1]
        cases_prob = cases_prob_ls[-1]

        return samples, cases_prob, new_r
    """
    nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=new_r, probs=cases_prob)
    kl_dive = nb.log_prob(gt)
    ####################################################
    kl_dive = kl_dive.view(-1, 118 * 6)
    #kl_dive = kl_dive.view(-1, 118 * 3)
    ###########################################################
    kl_dive_con = torch.sum(kl_dive, dim=1)
    kl_dive_ls = torch.chunk(kl_dive_con, len(idx_ladder), dim=0) # Per ladder, we get this one
    #sampled_x_ls = torch.chunk(sampled_x, len(idx_ladder), dim=0)

    w_ladder = w_ladder_ls[0]

    Pd = []


    ## Block for Blocks 
    if args.AIS : 
        if itr < 125: 
            Bl_ls = [0]
        else: 
            Bl_ls = [1]
    else:
        Bl_ls = [0] 
    
    ratio_ls = []
    for Bl in Bl_ls:
        log_probz = log_probz_ls[Bl]
        log_probq = log_probq_ls[Bl]
        kl_dive = kl_dive_ls[Bl]
        #w = w_ladder[Bl]
        T = temp_ladder[Bl]
        anneal = (1 - 1 / T)
        loss_fir = - (log_probq) + kl_dive / T # log_probz * anneal
        ratio_ls.append(loss_fir)


    if recover :
        cases_prob_ls = torch.chunk(cases_prob, len(idx_ladder), dim=0)
        new_r_ls = torch.chunk(new_r, len(idx_ladder), dim=0)
        new_r = new_r_ls[-1]
        cases_prob = cases_prob_ls[-1]
        x_ls = torch.chunk(x, len(idx_ladder), dim=0)
        new_x = x_ls[-1].clone().detach()
        #if args.AIS:
            #logw = ratio_ls[-1]
            
            #nplogw = logw.clone().detach().cpu().numpy()
            #logw_1, k = az.psislw(nplogw)
            #w = np.exp(logw_1)
            #fkl = w * (nplogw - np.log(2)) 
            #print(np.mean(fkl))
            #print(k)
            #w_norm = torch.exp(logw- torch.max(logw))
            #w = w_norm / torch.sum(w_norm)
            #fkl = w* (logw+np.log(1.5)) #-np.log(1.2)
            #print(torch.mean(fkl))
            #w_norm = PSIS_sampling(w.clone().detach().cpu().numpy())
        return samples, cases_prob, new_x, new_r, log_probq, kl_dive
        """
        if args.AIS:
            logp2 = kl_dive_ls[-1]
            logp1 = kl_dive_ls[-2]
            T2 = temp_ladder[-1]
            T1 = temp_ladder[-2]
            logal = (logp2 - logp1) / T1 + (logp1 - logp2) / T2 
            al = torch.clamp(torch.exp(logal), max=1.0) 
            idx = torch.bernoulli(al) # Prob of T1 selection, if 1, T1 is selected  
            #print(idx)
            indices = (idx == 1).nonzero(as_tuple=True)[0]
            #il = idx.shape[0]
            #print(idx.shape, 500)
            idx = indices
            new_x = x_ls[-1].clone().detach()
            new_r = new_r_ls[-1].clone().detach()
            cases_prob = cases_prob_ls[-1].clone().detach()
            #idx = idx * torch.arange(0, il, dtype=torch.int8, device=args.device)
            #idx = idx.tolist()
            #print(indices)
            new_x[indices] = x_ls[-2][indices]
            
            cases_prob[idx] = cases_prob_ls[-2][idx]
            new_r[idx] = cases_prob_ls[-2][idx]

            log_probq = log_probq_ls[-1]
            loss_fir = - log_probq + logp2
            print(loss_fir.mean())
            #x_ls = torch.chunk(x, len(idx_ladder), dim=0)
            
            #torch.cat()

        """
        


    loss = loss_fir #* (logw)
    loss = -1 * loss.mean()
    #loss = 0.0 
    #for ratio in ratio_ls:
    #    loss = loss + ratio 
    #loss = loss.mean()
    nll = kl_dive
    #loss = -1 * rato_l.mean() 

    

    return loss, nll, samples


def only_forward(param_sample, args):
    idx_ladder, temp_ladder, w_ladder_ls = get_ladder(args)

    sampled_x = torch.tensor(data=param_sample, device=args.device)
    x = sampled_x

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

    cases_fit = cases_age

    cases_fit_chunked = torch.chunk(cases_fit, len(idx_ladder), dim=0)
    samples = cases_fit_chunked[-1].detach().cpu().numpy()

    for j in range(B):
        cases_fit[j, :, :] /= r[j]

    new_r = r.repeat(1, t.shape[1] * 6)
    new_r = new_r.view(B, t.shape[1], 6)
    cases_prob = cases_fit / (1 + cases_fit)

    cases_prob_ls = torch.chunk(cases_prob, len(idx_ladder), dim=0)
    new_r_ls = torch.chunk(new_r, len(idx_ladder), dim=0)

    new_r = new_r_ls[-1]
    cases_prob = cases_prob_ls[-1]

    return samples, cases_prob, new_r, new_r




