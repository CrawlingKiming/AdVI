import math
import torch
import torch.nn.functional as F
from survae.transforms.surjections import Surjection
from survae.utils import sum_except_batch


class BoundSurjection(Surjection):
    '''
    An absolute value layer.
    Uses a fixed inverse which flips the sign with probability 0.5.
    This enforces symmetry across all axes.
    '''

    stochastic_forward = False
    def __init__(self, a = -1.5, b = 1.5):
        super(BoundSurjection, self).__init__()
        self.ub = b
        self.lb = a

    def forward(self, z):

        z = torch.clamp(z, min = -3.0, max = 3.0)
        x = self.bound_function(z)
        ldj = - z.new_ones(z.shape[0]) * math.log(3) * z.shape[1:].numel()
        return x, ldj

    def inverse(self, z):
        raise RuntimeError("Does not support inverse flows.")

    def bound_function(self, z):
        s2mask = (z > self.ub)
        s0mask = (z < self.lb)
        tempz = z.clone()
        s2z = 2 * self.ub - z[s2mask]
        s0z = 2 * self.lb - z[s0mask]

        tempz[s2mask] = s2z
        tempz[s0mask] = s0z

        #overbounded = torch.sum(s2mask) + torch.sum(s0mask)
        return tempz#,overbounded

class SimpleAbsSurjection(Surjection):
    '''
    An absolute value layer.
    Uses a fixed inverse which flips the sign with probability 0.5.
    This enforces symmetry across all axes.
    '''

    stochastic_forward = False

    def forward(self, x):
        z = x.abs()
        ldj = - x.new_ones(x.shape[0]) * math.log(2) * x.shape[1:].numel()
        return z, ldj

    def inverse(self, z):
        s = torch.bernoulli(0.5*torch.ones_like(z))
        x = (2*s-1)*z
        return x