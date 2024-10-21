import torch
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import datetime
import matplotlib.pyplot as plt

class mRNA(torch.nn.Module):

    def __init__(self, args=None):
        device = "cpu"
        if args != None :
            device = args.device

        self.c = torch.tensor(24.44, device=device)

        super().__init__()

    def forward(self, t, states):
        """
        function of ODE.

        states : stores ODE variables. ex dM dS dR
        params : class constants. stores class parameters. we do not need to store this parameter.
        """

        z = states[0]
        params = states[1]

        # B, 1
        th_2 = params[:, [0]]
        th_3 = params[:, [1]]
        th_5 = params[:, [2]]
        th_6 = params[:, [3]]
        B = th_2.shape[0]

        th_1 = 0.04382958 *torch.ones(size=(B,1), device=params.device)
        th_4 = 0.11033000 *torch.ones(size=(B,1), device=params.device)
        th_7 = 0.2172353 *torch.ones(size=(B,1), device=params.device)
        th_8 = 6.157514 *torch.ones(size=(B,1), device=params.device)
        th_9 = 11.131565  *torch.ones(size=(B,1), device=params.device)

        #th_1 = params[:, [0]]
        #th_2 = params[:, [1]]
        #th_3 = params[:, [2]]
        #th_4 = params[:, [3]]
        #th_5 = params[:, [4]]
        #th_6 = params[:, [5]]
        #th_7 = params[:, [6]]
        #th_8 = params[:, [7]]
        #th_9 = params[:, [8]]

        # B, 1
        Y = z[:, [0]]
        W = z[:, [1]]
        Z = z[:, [2]]

        # B = Y.shape[0]  # Determine Batchsize
        zero_tensor = torch.zeros(params.shape, device=params.device)
        with torch.set_grad_enabled(True):
            dY = self.c / (1 + (Z/th_8).pow(8)) - th_1 * Y
            dW = th_2 * Y - (th_3 + th_4) * W + th_6 * Z - th_7 * W * Z.pow(4) / (Z.pow(4) + th_9.pow(4))
            dZ = th_4 * W - (th_5 + th_6) * Z + th_7 * W * Z.pow(4) / (Z.pow(4) + th_9.pow(4))
            z_t = torch.cat((dY, dW, dZ), 1)
            params = zero_tensor#.to(th_1.device)

        return (z_t, params)





def ODE_Solver(params, tt, model, args, t_end = None, initial_params =None):

    B = params.shape[0]
    Y = torch.tensor([[0.0]])
    W = torch.tensor([[0.0]])
    Z = torch.tensor([[0.0]])
    z0 = torch.cat((Y, W, Z), 1)
    z0 = z0.repeat(B, 1).to(args.device)

    func = model.to(args.device)
    tt = tt.to(args.device)

    final_params = params[:, :4]
    z_samples, _ = odeint(
        func,
        (z0, final_params),
        tt,
        atol=1e-2,
        rtol=0,
        method="dopri5"
    )
    z_samples = z_samples - torch.mean(z_samples[-66:, :, :], dim=0)
    return z_samples[:, :, 0]

if __name__ == "__main__":
    term = 500
    tt = torch.linspace(0, term+2, steps=term+1)
    func = mRNA()

    t_end = tt[-66:]

    Y = torch.tensor([[0.0]])
    W = torch.tensor([[0.0]])
    Z = torch.tensor([[0.0]])
    z0 = torch.cat((Y, W, Z), 1)
    print(z0.shape)
    params = torch.tensor(
        [[0.0312, 3.42258214,  0.44889856,  0.23426515,  0.52461510,  0.73610651,  0.88771403,  6.16031968, 14.23507139]],
        requires_grad=True)
    print(params[:, :9])
    z_samples, params = odeint(
        func,
        (z0, params),
        tt,
        atol=1e-2,
        rtol=0,
        method="dopri5"
    )
    z_samples = z_samples - torch.mean(z_samples[-66:,:,:], dim=0)
    z_samples = z_samples.detach().cpu().numpy()
    obs_y = z_samples[-66:,0,0]
    print(z_samples.shape)
    print(obs_y)
    plt.plot(obs_y)
    plt.show()