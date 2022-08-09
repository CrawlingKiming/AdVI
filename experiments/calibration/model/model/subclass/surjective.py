import math
import torch
import torch.nn.functional as F
from survae.transforms.surjections import Surjection
from survae.utils import sum_except_batch

from survae.distributions import Distribution
from survae.transforms.surjections import Surjection


from survae.distributions import Distribution
from survae.utils import sum_except_batch

class SHN_Calc(Distribution):
    """A standard half-Normal with zero mean and unit covariance."""

    def __init__(self, shape, a=-1.5, b=1.5):
        super(SHN_Calc, self).__init__()
        #self.shape = torch.Size(shape)
        self.register_buffer('buffer', torch.zeros(1))
        self.lb = a
        self.ub = b

    def log_prob(self, x):

        log_scaling = math.log(2)
        log_base =    - 0.5 * math.log(2 * math.pi)
        log_inner =   - 0.5 * x**2
        log_probs = log_scaling+log_base+log_inner
        log_probs[x < 0] = -math.inf
        return sum_except_batch(log_probs)

    def sample(self, num_samples):
        raise NotImplementedError
        #return torch.randn(num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype).abs()

class BoundMaxSurjection(Surjection):
    '''
    An max pooling layer.
    Args:
        decoder: Distribution, a distribution of shape (3*c, h//2, w//2) with non-negative elements.
    '''

    stochastic_forward = False

    def __init__(self, decoder, a=-1.5, b=1.5):
        super(BoundMaxSurjection, self).__init__()
        assert isinstance(decoder, Distribution)
        self.decoder = decoder
        self.lb = a
        self.ub = b

    def forward(self, x):
        z, xd, k = self._deconstruct_x(x)
        ldj_k = - math.log(4) * z.shape[1:].numel()
        ldj = self.decoder.log_prob(xd) + ldj_k
        return z, ldj

    def inverse(self, z):
        k = torch.randint(0, 4, z.shape, device=z.device)
        xd = self.decoder.sample(z.shape[0])
        x = self._construct_x(z, xd, k)
        return x



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

        z = torch.clamp(z, min = -1.75, max = 1.75)
        #temp_z = z.new_ones(z.shape) / 3 - torch.abs(z) / 9
        #s0mask = z < -1.5
        #s1mask = z > 1.5


        x = self.bound_function(z)
        temp_p = z.new_ones(z.shape)
        mask = torch.abs(z)>1.25
        temp_p[mask] = 1.5 - torch.abs(x[mask])
        #print(temp_p)
        #raise ValueError
        #temp = z.new_ones(z.shape) / 3 - torch.abs(x) / 9
        #zmask = torch.abs(z) > 2.8
        #temp = z.new_ones(z.shape)
        #temp[zmask]

        #ldj = - z.new_ones(z.shape[0]) * math.log(3) * z.shape[1:].numel()
        #temp = z.new_ones(z.shape) / 3 - torch.abs(z) / 9
        #print(temp)
        #temp_p = torch.clamp(temp, min = 1e-10)
        temp_p = torch.clamp(temp_p, min=1e-10)
        temp_p[temp_p == 1] = temp_p[temp_p == 1] - 1e-3
        temp_q = 1 - temp_p
        l = 400
        temp_ldj = z.new_zeros(temp_p.shape)
        for _ in range(l):
            i = torch.bernoulli(temp_p)
            addit = -i * torch.log(temp_p) -  (1-i) * torch.log(temp_q)
            temp_ldj = temp_ldj + addit
        ldj = -1 * torch.sum(temp_ldj, dim=1) / l
        #print(ldj)
        #raise ValueError
        #ldj = -1 * torch.sum(torch.log(temp), dim=1)

        return x, ldj

    def inverse(self, z):
        # Z is always in bound
        pass

    def bound_function(self, z):
        s2mask = z > 1.5 #(z > self.ub)
        s0mask = z < -1.5  #(z < self.lb)
        tempz = z.clone()
        s2z = 2 * self.ub - z[s2mask]
        s0z = 2 * self.lb - z[s0mask]

        tempz[s2mask] = s2z
        tempz[s0mask] = s0z

        #overbounded = torch.sum(s2mask) + torch.sum(s0mask)
        return tempz#,overbounded


class BoundSurjection_S(Surjection):
    '''
    Bound Surjection.

    This layer surjects
    '''

    stochastic_forward = False
    def __init__(self, a = -1.5, b = 1.5):
        super(BoundSurjection_S, self).__init__()
        self.ub = b
        self.lb = a

    def forward(self, z):

        #z = torch.clamp(z, min = -3.0, max = 3.0)
        x = self.bound_function(z)
        z_p1 = self.sig_ldj(z)
        xx = x.clone()#.detach()
        x_p1 = self.sig_ldj(xx)

        p = z_p1 / (z_p1 + x_p1)
        ldj = -1 * torch.log(p)
        #ldj = p * -1 * torch.log(p) + (1-p) * -1 * torch.log(1-p)
        ldj = -1 *ldj#-1 * torch.log(ldj)
        #ldj = -1 * torch.log(torch.sigmoid_( 8 * (-1 * torch.abs(z) + 1.5 )))
        ldj =  torch.sum(ldj, dim=1)

        #x = torch.clamp(x, min=-1.5, max=1.5)
        #print(x, ldj)
        #temp = z.new_ones(z.shape) / 3 - torch.abs(z) / 9
        #print(temp)
        #temp = torch.clamp(temp, min = 1e-40)
        #ldj = torch.sum(torch.log(temp), dim=1)
        #ldj = torch.zeros(size=(z.shape[0],), device=z.device) + 10 * math.log(2)
        #ldj = 5 * ldj
        #print(s0.shape)
        #ldj =  -1 * torch.sum((s0 * torch.log(s0) * 0.95 + s1 * torch.log(s1) * 0.035 + s2 * torch.log(s2) * 0.035), dim=1)

        #ldj = z.new_ones(z.shape[0]) * math.log(1) * z.shape[1:].numel()
        #ldj =  z.new_ones(z.shape[0]) * math.log(1) * z.shape[1:].numel()
        return x, ldj

    def inverse(self, z):
        raise RuntimeError("Does not support inverse flows.")

    def log_prob(self, value):
        return -((value) ** 2) / 2 - math.log(math.sqrt(2 * math.pi))

    def prob(self, value):
        return torch.exp(-((value) ** 2) / 2 - math.log(math.sqrt(2 * math.pi)))

    def bound_function(self, z):
        s2mask = (z > self.ub)
        s0mask = (z < self.lb)
        tempz = z.clone()
        s2z = 2 * self.ub - z[s2mask]
        s0z = 2 * self.lb - z[s0mask]

        tempz[s2mask] = torch.clamp(s2z, min=0)
        tempz[s0mask] = torch.clamp(s0z, max=0)

        #overbounded = torch.sum(s2mask) + torch.sum(s0mask)
        return tempz#,overbounded

    def sig_ldj(self, z):
        return torch.sigmoid_( 8 * (-1 * torch.abs(z) + 1.5 ))


class BoundSurjection_sig(Surjection):
    '''
    An absolute value layer.
    Uses a fixed inverse which flips the sign with probability 0.5.
    This enforces symmetry across all axes.
    '''

    stochastic_forward = False
    def __init__(self, a = -1.5, b = 1.5):
        super(BoundSurjection_sig, self).__init__()
        self.ub = b
        self.lb = a

    def forward(self, z):
        #z = torch.clamp(z, min = -2.99, max = 2.99)
        s2mask = (z > self.ub)
        s0mask = (z < self.lb)

        z[s0mask] = (torch.sigmoid_(self.lb - z[s0mask])-0.5) / 5 + self.lb
        z[s2mask] = - (torch.sigmoid_(z[s2mask] - self.ub)-0.5) / 5 + self.ub

        # 둘 다 mask에 안 걸리면,
        pre_ldj = self.w_cal(z)
        pre_ldj[s0mask] = 1 - pre_ldj[s0mask]
        pre_ldj[s2mask] = 1 - pre_ldj[s2mask]
        #print(pre_ldj[0:2])
        #raise ValueError
        ldj = torch.sum(torch.log(pre_ldj), dim=1)
        #print(ldj)
        x = z
        return x, ldj

    def inverse(self, z):
        # Z is always in bound
        pass

    def bound_function(self, z):
        z = (torch.sigmoid_(torch.clamp(self.lb - z, min=0)) - 0.5)/5 + torch.clamp(z - self.lb, min=0)


    def w_cal(self, z):
        z_small = (z < self.lb + 0.1)
        z_big = (z > self.ub - 0.1)

        z_mask = z_small + z_big
        z = -1 * torch.abs(z)*2 + 3.8
        z[z_mask < 1] = 1
        return z


class BoundSurjection_Max(Surjection):
    '''
    An absolute value layer.
    Uses a fixed inverse which flips the sign with probability 0.5.
    This enforces symmetry across all axes.
    '''

    stochastic_forward = False
    def __init__(self, a = -1.5, b = 1.5):
        super(BoundSurjection_Max, self).__init__()
        self.ub = b
        self.lb = a

    def forward(self, z):
        size = 0.1

        s0mask = (z < self.lb)
        s2mask = (z > self.ub)
        perelement = (s0mask + s2mask)

        s0_rowmask = torch.sum(s0mask, dim=1) # Number of Bound over per batch
        s2_rowmask = torch.sum(s2mask, dim=1)
        #x = torch.clamp(z, min=self.lb, max=self.ub)

        #e1 = torch.randn(size=z[s0_rowmask>0].shape, device=z.device) * size
        #e2 = torch.randn(size=z[s2_rowmask>0].shape, device=z.device) * size

        thetha_b = torch.clamp(z, min=self.lb, max=self.ub)
        ldj = torch.zeros(size=(z.shape[0],), device=z.device)
        final_mask = s0_rowmask + s2_rowmask
        perelement_row = perelement[final_mask > 0]
        #perelement_notrow = perelement[final_mask == 0] # for not bounded
        #print(perelement_row, perelement_row.shape)
        log_p = self.log_prob(z[final_mask > 0] - thetha_b[final_mask > 0])
        log_p[perelement_row == 0] = 0 #bound 안된애는 다 0
        ldj[final_mask > 0] = -1 * sum_except_batch(log_p)

        e = torch.randn(size=z[final_mask > 0].shape, device=z.device) * size
        log_q = self.noise_log_prob(e, size=size)
        #print(log_q.shape)
        log_q[perelement_row > 0] = 0
        ldj[final_mask > 0] = ldj[final_mask > 0] + sum_except_batch(log_q)

        e[perelement_row > 0] = 0
        #print(e)
        #raise ValueError
        x = thetha_b
        x[final_mask > 0] = x[final_mask > 0] + e
        x = torch.clamp(x, min=self.lb, max=self.ub)

        #x[s0_rowmask > 0 ] = x[s0_rowmask > 0] + e1
        #x[s2_rowmask > 0] = x[s2_rowmask > 0] - e2
        #x[s0mask] = x[s0mask] + e1
        #x[s2mask] = x[s2mask] + e2
        #e1[s0mask[s0_rowmask>0]]
        #s0_rowmask
        #z[s0_rowmask > 0] = z[s0_rowmask > 0] + e1
        #z[s2_rowmask > 0] = z[s2_rowmask > 0] - e2
        #e1 = torch.randn(size=z[s0_rowmask >0].shape, device=z.device) * size
        #e2 = torch.randn(size=z[s2mask].shape, device=z.device) * size

        #ldj = torch.zeros(size=(z.shape[0],), device=z.device)
        #final_mask = s0_rowmask + s2_rowmask # Number of Bound over per batch
        #ldj[final_mask > 0] = -1 * self.log_prob(z[final_mask > 0]-x[final_mask > 0])



        #ldj[final_mask > 0] = ldj[final_mask > 0] + self.noise_log_prob(z[final_mask > 0]-x[final_mask > 0], size=size)

        #x[s0mask] = x[s0mask]
        #x[s0mask] = x[s0mask] + torch.abs(torch.randn(size=x[s0mask].shape, device=z.device)) * size
        #x[s2mask] = x[s2mask] - torch.abs(torch.randn(size=x[s2mask].shape, device=z.device)) * size
        #z[s0mask] = self.lb - z[s0mask]
        #print(x)

        #s0_rowmask = torch.sum(s0mask, dim=1) # Number of Bound over per batch
        #s2_rowmask = torch.sum(s2mask, dim=1)
        #x[s0mask] = x[s0mask] + torch.abs(torch.randn(size=x[s0mask].shape, device=z.device)) * size
        #x[s2mask] = x[s2mask] - torch.abs(torch.randn(size=x[s2mask].shape, device=z.device)) * size
        #final_mask = s0_rowmask + s2_rowmask # Number of Bound over per batch
        #ldj = torch.zeros(size=(z.shape[0],), device=z.device)
        #ldj[final_mask > 0] = self.log_prob(z[final_mask > 0]-x[final_mask > 0])
        #ldj[s0_rowmask > 0] = ldj[s0_rowmask > 0] -1 * self.noise_log_prob(e1, size)
        #ldj[s2_rowmask > 0] = ldj[s2_rowmask > 0] - 1 * self.noise_log_prob(e2, size)

        #ldj_born = torch.ones(size=(z.shape[0],), device=z.device) * (0.5 * math.log(2* math.pi * sig**2) + 0.5 - math.log(2))
        #ldj = ldj_born * final_mask
        #raise ValueError
        return x, ldj

    def log_prob(self, x):
        #masking = (x == 0.01)
        #print(x)
        sig = 1#math.log(math.pi / 2)
        log_scaling = math.log(2)
        log_base =    0.5 * math.log(2 / math.pi) - math.log(sig) #- log_scaling
        log_inner =   (- 0.5 * x**2)/sig**2

        log_probs = log_base+log_inner

        #log_probs[masking] = 0
        #print(log_probs)
        #print(sum_except_batch(log_probs))
        #log_probs[x < self.lb] = 0
        #log_probs[x > self.ub] = 0
        return log_probs#um_except_batch(log_probs)

    def noise_log_prob(self, x, size):
        #masking = (x < 0.01)
        #print(x)
        sig = size#0.5#math.log(math.pi / 2)
        log_scaling = math.log(2)
        log_base =    0.5 * math.log(1 / math.pi) - math.log(sig) - 0.5 * log_scaling
        log_inner =   (- 0.5 * x**2)/sig**2

        log_probs = log_base+log_inner
        #log_probs[masking] = 0
        #print(log_probs)
        #print(sum_except_batch(log_probs))
        #log_probs[x < self.lb] = 0
        #log_probs[x > self.ub] = 0
        return log_probs#sum_except_batch(log_probs)



class BoundSurjection_Max_sample(Surjection):
    '''
    An absolute value layer.
    Uses a fixed inverse which flips the sign with probability 0.5.
    This enforces symmetry across all axes.
    '''

    stochastic_forward = False
    def __init__(self, a = -1.5, b = 1.5):
        super(BoundSurjection_Max_sample, self).__init__()
        self.ub = b
        self.lb = a

    def forward(self, z):
        size = 0.1

        s0mask = (z < self.lb)
        s2mask = (z > self.ub)
        s0_rowmask = torch.sum(s0mask, dim=1) # Number of Bound over per batch
        s2_rowmask = torch.sum(s2mask, dim=1)
        #z[s0_rowmask > 0] = z[s0_rowmask > 0] + torch.randn(size=z[s0_rowmask>0].shape, device=z.device) * size
        #z[s2_rowmask > 0] = z[s2_rowmask > 0] - torch.randn(size=z[s2_rowmask>0].shape, device=z.device) * size
        x = torch.clamp(z, min=self.lb, max=self.ub)

        #x[s0mask] = x[s0mask] + torch.abs(torch.randn(size=x[s0mask].shape, device=z.device)) * size
        #x[s2mask] = x[s2mask] - torch.abs(torch.randn(size=x[s2mask].shape, device=z.device)) * size
        #z[s0mask] = self.lb - z[s0mask]
        #print(x)

        s0_rowmask = torch.sum(s0mask, dim=1) # Number of Bound over per batch
        s2_rowmask = torch.sum(s2mask, dim=1)
        #x[s0mask] = x[s0mask] + torch.abs(torch.randn(size=x[s0mask].shape, device=z.device)) * size
        #x[s2mask] = x[s2mask] - torch.abs(torch.randn(size=x[s2mask].shape, device=z.device)) * size
        final_mask = s0_rowmask + s2_rowmask # Number of Bound over per batch
        ldj = torch.zeros(size=(z.shape[0],), device=z.device)
        #x = x[final_mask==0]
        #ldj = ldj[final_mask==0]
        #ldj[final_mask > 0] = -1 * self.log_prob(z[final_mask > 0]-x[final_mask > 0])


        #ldj_born = torch.ones(size=(z.shape[0],), device=z.device) * (0.5 * math.log(2* math.pi * sig**2) + 0.5 - math.log(2))
        #ldj = ldj_born * final_mask
        #raise ValueError
        return x, ldj

    def log_prob(self, x):
        #masking = (x < 0.01)
        #print(x)
        sig = 2#math.log(math.pi / 2)
        log_scaling = math.log(2)
        log_base =    0.5 * math.log(2 / math.pi) - math.log(sig)
        log_inner =   (- 0.5 * x**2)/sig**2

        log_probs = log_base+log_inner
        #log_probs[masking] = 0
        #print(log_probs)
        #print(sum_except_batch(log_probs))
        #log_probs[x < self.lb] = 0
        #log_probs[x > self.ub] = 0
        return sum_except_batch(log_probs)

class BoundSurjection_Q(Surjection):
    '''
    An absolute value layer.
    Uses a fixed inverse which flips the sign with probability 0.5.
    This enforces symmetry across all axes.
    '''

    stochastic_forward = False
    def __init__(self, a = -1.5, b = 1.5):
        super(BoundSurjection_Q, self).__init__()
        self.ub = b
        self.lb = a

    def forward(self, z, ldj, model):
        s0mask = (z < 0)
        s2mask = ~s0mask

        s2z = 2 * self.ub - torch.masked_select(z, s2mask)#z[s2mask]
        s0z = 2 * self.lb - torch.masked_select(z, s0mask)#z[s0mask]
        z0 = z.clone()
        print(torch.masked_select(z, s2mask).shape)
        print(z.shape)
        print(s0z.shape)
        s0_p_ldj, _ = model.inversed(s0z)
        s2_p_ldj, _ = model.inversed(s2z)

        # Make P1
        final_p = torch.zeros(size=ldj.shape)
        # s0 part
        p01 = ldj[s0mask]
        final_p[s0mask] = torch.exp(p01) / (torch.exp(p01) + torch.exp(s0_p_ldj))

        p02 = ldj[s2mask]
        final_p[s2mask] = torch.exp(p02) / (torch.exp(p02) + torch.exp(s2_p_ldj))

        entropy = -1*(final_p * torch.log(final_p) + (1-final_p) * torch.log(1-final_p + 1e-5))

        return entropy


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

    def bound_inverse(self, x):
        s1tos0mask = x < 0
        s1tos2mask = x > 0


class BoundSurjection_M(Surjection):
    '''
    An absolute value layer.
    Uses a fixed inverse which flips the sign with probability 0.5.
    This enforces symmetry across all axes.
    '''

    stochastic_forward = False
    def __init__(self, a = -2.8, b = 2.8):
        super(BoundSurjection_M, self).__init__()
        self.ub = b
        self.lb = a

    def forward(self, z, ldj_ls, model):
        z = torch.clamp(z, min=-2.99, max=2.99)
        s0mask = (z < self.lb)
        s2mask = (z > self.ub)

        s0_rowmask = torch.sum(s0mask, dim=1)
        s2_rowmask = torch.sum(s2mask, dim=1)
        final_mask = s0_rowmask + s2_rowmask
        z[final_mask > 0] = z[final_mask > 0] * (2.79 /2.99)

        z_1 = torch.chunk(z, len(ldj_ls), dim=0)[-1]
        ldj = torch.zeros(size=(z_1.shape[0],), device=z.device)
        s0mask = (z_1 < self.lb * 2.8 / 3.0)
        s2mask = (z_1 > self.ub * 2.8 / 3.0)
        s0_rowmask = torch.sum(s0mask, dim=1)
        s2_rowmask = torch.sum(s2mask, dim=1)
        final_mask = s0_rowmask + s2_rowmask
        ldj[final_mask > 0] = ldj[final_mask > 0] + 0.05
        """
        z_1 = torch.chunk(z, len(ldj_ls), dim=0)[-1]
        ldj = torch.zeros(size=(z_1.shape[0],), device=z.device)
        s0mask = (z_1 < self.lb * 2.8 / 3.0)
        s2mask = (z_1 > self.ub * 2.8 / 3.0)
        s0_rowmask = torch.sum(s0mask, dim=1)
        s2_rowmask = torch.sum(s2mask, dim=1)
        final_mask = s0_rowmask + s2_rowmask
        #print(s0_rowmask==1)
        indexed = final_mask > 0
        #print(indexed)

        z_transform = z_1[indexed] * 3.0/2.8

        s_p_ldj, _ = model.inversed(z_transform)
        p01 = ldj_ls[-1][indexed]
        #print(p01)
        #print(p01.shape)
        p02 = s_p_ldj
        p02 = torch.nan_to_num(p02, nan=1e-5)
        res = torch.exp(p02 - p01)
        #print(p02)
        final_p = 1 / (1 + torch.exp(res))
        final_p = final_p #* 0.95 + 0.015
        #print(final_p)
        isone = (final_p < 0.99)
        iszero = (final_p > 0.01)
        mask = isone + iszero
        entropy = -1*(final_p * torch.log(final_p) + (1-final_p) * torch.log(1-final_p))
        entropy[mask > 0] = 0.0
        #print(entropy.shape)
        print(entropy)
        #print(ldj[indexed].shape)
        ldj[indexed] = ldj[indexed] + entropy.detach()
        """
        """
        s0_rowmask = torch.sum(s0mask, dim=1)
        #print(z[s0_rowmask==1].shape)
        s2mask = ~s0mask

        #s2z = z[]2 * self.ub - torch.masked_select(z, s2mask)#z[s2mask]
        s0z = 2 * self.lb - torch.masked_select(z, s0mask)#z[s0mask]
        z0 = z.clone()
        print(torch.masked_select(z, s2mask).shape)
        print(z.shape)
        print(s0z.shape)
        s0_p_ldj, _ = model.inversed(s0z)
        s2_p_ldj, _ = model.inversed(s2z)

        # Make P1
        final_p = torch.zeros(size=ldj.shape)
        # s0 part
        p01 = ldj[s0mask]
        final_p[s0mask] = torch.exp(p01) / (torch.exp(p01) + torch.exp(s0_p_ldj))

        p02 = ldj[s2mask]
        final_p[s2mask] = torch.exp(p02) / (torch.exp(p02) + torch.exp(s2_p_ldj))

        entropy = -1*(final_p * torch.log(final_p) + (1-final_p) * torch.log(1-final_p + 1e-5))
        """
        #print(ldj)
        return z, ldj


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

    def bound_inverse(self, x):
        s1tos0mask = x < 0
        s1tos2mask = x > 0


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