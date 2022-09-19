import torch
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import datetime
import matplotlib.pyplot as plt

class SEIR(torch.nn.Module):

    def __init__(self, args = None):
        device = "cpu"
        if args != None :
            device = args.device

        self.n_age = torch.tensor(21, device=device)
        self.A = self.constant_settings().to(device)
        self.a = torch.tensor(1/7, device=device)
        self.gamma = torch.tensor(1/7, device=device)

        super().__init__()

    def forward(self, t, states):
        """
        function of ODE.

        states : stores ODE variables. ex dM dS dR
         In this case for MSIR, this is B * 29
        params : class constants. stores class parameters. we do not need to store this parameter.
        """
        z= states[0] #D_z #the initial variables
        params = states[1] #D_params,B
        beta = params[:, [0]]
        N = params[:, [1]]

        S = z[:, 0*self.n_age:self.n_age]
        E = z[:, 1*self.n_age:2*self.n_age]
        I = z[:, 2*self.n_age:3*self.n_age]
        R = z[:, 3*self.n_age:4* self.n_age]

        B = S.shape[0] # Determine Batchsize

        zero_tensor = torch.zeros(beta.shape)

        with torch.set_grad_enabled(True):

            lambd = torch.bmm(self.A.unsqueeze(0).repeat(B,1,1), I.unsqueeze(-1)) # B I I * B I 1 = B I 1
            lambd = beta.unsqueeze(-1).repeat(1,self.n_age,1) * lambd
            lambd = lambd[:,:,0] * S

            intense = lambd#[:,:,0] * S # B I  * B I = B I
            int_N = intense  #torch.div(intense, N.repeat(1, self.n_age))

            dS = -1 * int_N
            dE = int_N - self.a * E
            dI = self.a * E - self.gamma * I
            dR = self.gamma * I

            z_t = torch.cat((dS, dE, dI, dR), 1)

            db = zero_tensor
            params = db.repeat(1, 2).to(self.n_age.device)

        return (z_t, params)


    def constant_settings(self):
        n_age = self.n_age.clone().detach().cpu().numpy()

        A = np.ones(shape=(n_age, n_age)) * 2/3
        for i in range(n_age):
            A[i,i] = 1
            for j in range(i):
                if abs(i-j) == 1 :
                    A[i,j] = 11/12
                    A[j,i] = 11/12
                if abs(i-j) == 2 :
                    A[j,i] = 5/6
                    A[i,j] = 5/6
                if abs(i - j) == 3:
                    A[j, i] = 3 / 4
                    A[i, j] = 3 / 4

        return torch.tensor(A, dtype=torch.float32)



def ODE_Solver(params, tt, model, args, t_end = None, initial_params =None):

    """
    params :
        beta : Transmission day
        N0 : Susceptible population size before outbreak
        fe : Fraction of initally exposed
        ash, art : gamma dist param

    """
    n_age = 21
    B = params.shape[0]

    N0 = params[:,[1]]
    fe = params[:,[2]]
    ash = params[:, [3]]
    art = params[:, [4]]

    age_class = torch.linspace(0.5, 100.5, steps=101, device=args.device).unsqueeze(0).repeat(B,1)
    #print(ash, art)
    gamma_dist = torch.distributions.gamma.Gamma(ash, art)
    dage_class = torch.exp(gamma_dist.log_prob(age_class))
    age_class_s = torch.sum(dage_class[:, 20:], dim=1).unsqueeze(-1)

    dage_class = torch.cat((dage_class[:,:20], age_class_s), dim=1) # B * I  / B * 1
    pre_dist = torch.div(dage_class, torch.sum(dage_class, dim=1).unsqueeze(-1))
    dist_pop = N0.repeat(1,21) * pre_dist
    #print(dist_pop)
    I = torch.tensor([[0]], device=args.device).repeat(B, n_age)
    E = torch.clamp(fe.repeat(1, 21) * dist_pop, min=1)

    R = torch.tensor([[0]], device=args.device).repeat(B, n_age)
    S = torch.clamp(dist_pop - I - E- R, min=1)

    z0 = torch.cat((S, E, I, R), 1).to(args.device)
    final_params = params[:, [0,1]]

    func = model.to(args.device)
    tt = tt.to(args.device) #* 5

    z_samples, _ = odeint(
        func,
        (z0, final_params),
        tt,
        atol = 1e-1,
        rtol = 0,
        method="dopri5"
    )
    return z_samples
