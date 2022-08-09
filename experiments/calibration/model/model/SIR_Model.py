import torch
from torchdiffeq import odeint, odeint_adjoint
import numpy as np

class SIR(torch.nn.Module):

    def __init__(self, radius=0.2, population = 100):
        super().__init__()
        #self.population = torch.as_tensor([population])

    def forward(self, t, states):
        #parameter setting
        z= states[0] #D_z #the initial variables
        #t = states[1] #t fixed
        params = states[1] #D_params,B
        beta = params[:, [0]]
        gamma = params[:, [1]]

        S = z[:,[0]]
        I = z[:,[1]]
        R = z[:,[2]]

        zero_tensor = torch.zeros(beta.shape)
        #zero_tensor.get_device()
        with torch.set_grad_enabled(True):

            dS = -1 *beta * S * I / 100
            dI = beta * S * I / 100 - gamma * I
            dR = gamma * I

            z_t = torch.cat((dS, dI, dR), 1)

            dbeta = zero_tensor
            dgamma = zero_tensor
            params = torch.cat((dbeta, dgamma),1)

        return (z_t, params)


def ODE_Solver(gt, params, tt, model):
    #gt : t, dimension
    #params : B , D
    #params : B*D repeat t -> (B*t, D)

    #gt = gt[1:, :]

    #tshape = gt.shape[0]
    #batch_size = params.shape[0]
    #params = params.repeat(tshape, 1)
    #gt = gt.repeat(batch_size,1)
    func = model

    z_samples, _ = odeint(
        func,
        (gt, params),
        tt
    )
    return z_samples

if __name__ == "__main__":
    t0 = 0
    tt = torch.linspace(0, 17, steps = 18)
    #print(tt)
    func = SIR()
    z0 = torch.tensor([[99, 1, 0]])
    params = torch.tensor([[1.5, 0.5]], requires_grad=True)
    #print(tt)
    #print(z0.shape)
    z_samples, params = odeint(
        func,
        (z0, params),
        tt,
        method="dopri5"
    )
    z_samples = z_samples[:,0,:]
    z_samples = z_samples.detach().numpy()
    np.save("../../data/SIR/SIR.npy", z_samples)
    print(z_samples.shape)
    print(z_samples)
    print(params.grad_fn)

