import torch
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import datetime
import matplotlib.pyplot as plt

class MSIR(torch.nn.Module):

    def __init__(self, args = None):
        device = "cpu"
        if args != None :
            device = args.device
            #args.device = "cpu"
        self.age_minus = torch.tensor([1 / 8, 1 / 8, 1 / 8, 1 / 24, 1 / 48, 1 / 144], requires_grad=False, device= device)
        # age.minus = c(1/8,1/8,1/8,1/24,1/48,0)
        self.age_plus = torch.tensor([0, 1 / 8, 1 / 8, 1 / 8, 1 / 24, 1 / 48], requires_grad=False, device= device)
        self.frac = torch.tensor([2, 2, 2, 6, 12, 36], device= device) / (5 * 12)
        self.n_age = 6
        self.pi = torch.tensor(np.pi, device= device)
        self.contact = self.constant_settings().to(device)
        self.M1 = torch.tensor([1, 0, 0, 0, 0, 0], device = device)
        self.M2 = torch.tensor([1 / 8, 1 / 8, 0, 0, 0, 0], device = device)
        self.M3 = torch.tensor([0, 1 / 8, 0, 0, 0,0], device = device)
        self.S1 = torch.tensor([[0, 0, 1 / 8, 0, 0, 0]], device= device)

        super().__init__()

    def mu(self, t):
        t = t.detach().cpu().numpy()
        amplitude = torch.tensor([-.17, .01, .03, .25, .12, .03, -.01, .09, .01, .13, -.31, -.17])
        mu_0 = torch.tensor(1 / (52 * 5))
        # start_date = "2009-12-19"
        date_dt = datetime.date(2009, 12, 19)
        timedelta = datetime.timedelta(days=t * 7)
        current_date = date_dt + timedelta
        mu = current_date.month
        current_mu = mu_0 * (1 + amplitude[mu-1])
        return current_mu

    def forward(self, t, states):
        """
        function of ODE.

        states : stores ODE variables. ex dM dS dR
         In this case for MSIR, this is B * 29
        params : class constants. stores class parameters. we do not need to store this parameter.
        """
        z= states[0] #D_z #the initial variables
        params = states[1] #D_params,B
        b = params[:, [0]]
        phi = params[:, [1]]

        beta = params[:, 7:]
        # Theses params are out of interest.
        gamma_s = params[:, [2]]
        gamma_m = params[:, [3]]
        delta = params[:, [4]]
        tau = params[:, [5]]
        N = params[:, [6]]
        #N = self.N
        #print(z.shape)
        M = z[:,0:6]
        S = z[:,6:12]
        Is = z[:,12:18]
        Im = z[:,18:24]
        R = z[:, 24:29]
        #print(M.shape)
        B = M.shape[0] # Determine Batchsize

        zero_tensor = torch.zeros(b.shape)
        #zero_tensor.get_device()
        mu = self.mu(t)
        n_age = self.n_age
        with torch.set_grad_enabled(True):
            # M, S, Im, Is : B * I
            # R : B * (I -1)
            # Contact : I * I
            # beta, frac : 1 * I
            # Construct beta and lambda
            # b, phi : B * 1

            beta_t_pre = beta.unsqueeze(-1) * self.contact.unsqueeze(0).repeat(B,1,1) # I * I

            beta_cos = 1 + b * torch.cos((2 * self.pi* t - 52 * phi) / 52) # B * 1
            beta_t_pre = beta_t_pre.to(beta_cos.device) # B *I * I
            beta_cos = beta_cos.unsqueeze(-1)#.repeat(1, 6, 6)
            beta_t = beta_t_pre * beta_cos # B * I * I

            Y = torch.div((Is + 0.5 * Im),  (N * self.frac[None, :])) # B * I  * 1 * I = B * I
            #print(Y.shape)
            #print(Y.shape, beta_t.shape)
            #suppose B = 1

            Y = Y.unsqueeze(-1)
            lambd = torch.bmm(beta_t, Y)[:,:, 0] # B * I * 1 -> B* I
            #Y = Y[0]#.unsqueeze(1) # I,
            #beta_t = beta_t[0] # I, I
            #print(Y.shape, beta_t.shape)
            #lambd = torch.matmul(beta_t, Y) # B * I (1 * I )

            #print(lambd.shape)
            #raise ValueError
            lambda_s = 0.24 * lambd
            lambda_m = 0.76 * lambd

            M_min = torch.cat((M[:, [0]], M[: , 0:5]),1) # B 1, B 5
            S_min = torch.cat((S[:, [0]], S[: , 0:5]),1)
            Is_min = torch.cat((Is[:, [0]], Is[: , 0:5]),1)
            Im_min = torch.cat((Im[:, [0]], Im[: , 0:5]),1)
            R_min = torch.cat((R[:, [0]], R[: , 0:5]),1)
            #print(z.shape)
            # N : B * 1 - B

            R_last = N - torch.sum(z, dim=1).unsqueeze(1) # B, 1

            fortau_R = torch.cat((R, R_last), 1)
            # B * I

            # B * I
            dM = mu * N * self.M1 - self.M2\
                 * M + self.M3 * M_min - delta * M
            # B * I
            #age : I, S_min B I
            dS = self.age_plus[None, :] * S_min - self.age_minus[None, :] * S + delta * M \
                - lambd *S + self.S1 * M_min + tau * fortau_R
            dIs = self.age_plus[None, :] * Is_min - self.age_minus[None, :] * Is + lambda_s * S - gamma_s * Is
            dIm = self.age_plus[None, :] * Im_min - self.age_minus[None, :] * Im + lambda_m * S - gamma_m * Im
            dR = self.age_plus[0:5][None, :] * R_min[:, 0:5] - self.age_minus[0:5][None, :] * R \
                 + gamma_s * Is[:, 0:5] + gamma_m * Im[:, 0:5] - tau * R

            z_t = torch.cat((dM, dS, dIs, dIm, dR), 1)

            db = zero_tensor
            params = db.repeat(1, 13).to(beta_cos.device)
            #params = torch.cat((db),1).to(beta_cos.device)
        # reparameterization

        return (z_t, params)

    def constant_settings(self):
        n_age = self.n_age
        frac_pop = np.asarray([2/60,2/60,2/60,6/60,12/60,36/60])
        #frac_pop /= 60
        contact = np.ones(shape=(n_age, n_age))
        for i in range(n_age):
            for j in range(i):
                contact[i,j] = frac_pop[i]/ frac_pop[j]
        #print(contact)
        return torch.tensor(contact, dtype= torch.float32)



def ODE_Solver(params, tt, model, args, t_end = None, initial_params =None):
    #gt : t, dimension
    #params : B , D
    B = params.shape[0]
    use_params = params[:, 0:2]
    beta_params = params[:, 4:]
    assert beta_params.shape[1] == 6

    n_age = 6
    N = 324935
    M_0 = N*13/(5*52)
    eq = (N-M_0)/5
    eq_n = eq / n_age
    M = torch.tensor([[M_0,0,0,0,0,0]])
    S = torch.tensor([[eq_n, eq_n, eq_n, eq_n, eq_n, eq_n]])
    Is = torch.tensor([[eq_n, eq_n, eq_n, eq_n, eq_n, eq_n]])
    Im = torch.tensor([[eq_n, eq_n, eq_n, eq_n, eq_n, eq_n]])
    R = torch.tensor([[eq_n, eq_n, eq_n, eq_n, eq_n]])
    z0 = torch.cat((M,S,Is,Im, R), 1).to(args.device)

    if initial_params != None :
        z0 = initial_params
    z0 = z0.repeat(B,1)

    batch_size = params.shape[0]
    non_params = torch.tensor([[1,2,1/13,1/52, N]])

    non_params = non_params.repeat(batch_size,1).to(args.device)
    final_params = torch.cat((use_params,non_params, beta_params),1)
    func = model.to(args.device)
    tt = tt.to(args.device)
    z_samples, _ = odeint(
        func,
        (z0, final_params),
        tt,
        atol = 1,
        rtol = 0,
        method="dopri5"
    )
    return z_samples


if __name__ == "__main__":

    t0 = 0
    term = 5 * 52 + 118
    tt = torch.linspace(0, term, steps = term + 1)
    func = MSIR()
    reparam = False

    t_end = 1.0
    if reparam :
        t_end = tt[-1]
        #print(t_end)
        func = MSIR()
        tt /= t_end.clone()
        print(tt)

    n_age = 6
    N = 324935
    M_0 = N*13/(5*52)
    eq = (N-M_0)/5
    eq_n = eq / n_age
    M = torch.tensor([[M_0,0,0,0,0,0]])
    S = torch.tensor([[eq_n, eq_n, eq_n, eq_n, eq_n, eq_n]])
    Is = torch.tensor([[eq_n, eq_n, eq_n, eq_n, eq_n, eq_n]])
    Im = torch.tensor([[eq_n, eq_n, eq_n, eq_n, eq_n, eq_n]])
    R = torch.tensor([[eq_n, eq_n, eq_n, eq_n, eq_n]])
    z0 = torch.cat((M,S,Is,Im, R), 1)
    params = torch.tensor([[0.43, 7.35, 1, 2, 1/13, 1/52, N, 1.30537122, 3.11852393, 0.03206678, 3.24118948, 0.31400672, 0.09410381]], requires_grad=True)

    z0 = z0.repeat(2, 1)
    params = params.repeat(2, 1)
    #print(tt)
    print("Shape of the States:", z0.shape, params.shape)

    z_samples, params = odeint_adjoint(
        func,
        (z0, params),
        tt,
        atol=1,
        rtol=0,
        method="dopri5"
    )
    #print(params)
    tt = torch.linspace(0, term, steps = term + 1)
    z_samples = torch.transpose(z_samples, 0, 1)
    print(z_samples.shape)
    M = z_samples[:,-118:, 0:6]
    S = z_samples[:,-118:, 6:12]
    Is = z_samples[:,-118:, 12:18]
    Im = z_samples[:,-118:, 18:24]

    N_res = z_samples[0, :, :].sum(dim=1)
    plt.plot(Is[0].detach().cpu().numpy())
    plt.show()


    B = Is.shape[0]
    t = tt[-118:].unsqueeze(0).repeat(B,1) # B*t

    b = torch.tensor([[0.43]])
    phi = torch.tensor([[7.35]]) # B * 1
    r = torch.tensor([[1.5]])
    rho = torch.tensor([[0.027]])

    #####
    b = b.repeat(B, 1)
    r = r.repeat(B, 1)
    rho = rho.repeat(B, 1)

    #####

    beta_t_pre = func.beta[:, None] * func.contact  # I * I
    beta_t_pre = beta_t_pre.unsqueeze(0).repeat(B, 1, 1)  # B * I * I
    # expand this to B * t * I * I
    # then B* t ( I * I  X  1 )
    beta_cos = 1 + b * torch.cos((2 * torch.tensor(np.pi) * t - 52 * phi) / 52)  # B * t
    # beta_cose B * t * 1 * 1
    beta_cos = beta_cos.unsqueeze(2).unsqueeze(3).repeat(1, 1, 6, 6)
    beta_t_pre = beta_t_pre.unsqueeze(1)

    beta_t = beta_t_pre * beta_cos # B * t * I * I
    Y = torch.div((Is + 0.5 * Im), (N * func.frac[None, :]))  # B * I  * 1 * I = B* t * I

    # Assume B = 1
    Y = Y.reshape(B * 118, 6)
    beta_t = beta_t.reshape(B * 118, 6, 6)
    # Y : B * I -> B * I * 1 * 1
    # beta_t : B * t * I * I
    # B * t * I * 1 * 1 * 1
    lambd = torch.bmm(beta_t, Y.unsqueeze(-1))
    lambd = lambd.reshape(B, 118, 6)

    cases_age = 0.24 * lambd * S # B * t * I
    #print(cases_age.shape)
    for i in range(B):
        cases_age[i, :, :] *= rho[i]

    #print(cases_age.shape) # B * t * I
    cases1 = torch.sum(cases_age[:, :, 0: 3], dim=2).unsqueeze(2)
    cases2 = torch.sum(cases_age[:, :, [3]], dim=2).unsqueeze(2)
    cases3 = torch.sum(cases_age[:, :, 4: 6], dim=2).unsqueeze(2)
    cases_fit = torch.cat((cases1, cases2, cases3), 2) # B * t * 3
    samples = cases_fit.detach().cpu().numpy()

    for i in range(B):
        cases_fit[i, : , :] /= r[i]

    new_r = r.repeat(1, t.shape[1]* 3)
    #print(r.shape, new_r.shape)
    new_r = new_r.view(B, t.shape[1], 3)
    print(new_r.shape, cases_fit.shape)
    cases_prob = cases_fit / (1+ cases_fit)
    nb = torch.distributions.negative_binomial.NegativeBinomial(total_count = new_r, probs = cases_prob)
    sample = nb.sample((1,))

    print(-1 * nb.log_prob(sample).mean())