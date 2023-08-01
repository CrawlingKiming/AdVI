import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform, ConditionalTransform

from survae.flows import Flow
from survae.distributions import StandardNormal, StandardUniform
from survae.transforms import AdditiveCouplingBijection, AffineCouplingBijection, ActNormBijection, Reverse, ConditionalAffineCouplingBijection, StochasticPermutation,Shuffle
from survae.transforms import Logit, SoftplusInverse, Sigmoid
from survae.nn.nets import MLP
from survae.nn.layers import ElementwiseParams, scale_fn
from survae.transforms import Sigmoid, Logit
from .layers import ShiftBijection, ScaleBijection, ActShiftBijection

from .diagnostic import tile
from survae import utils

class InverseFlow(Flow):
    """
    Base class for Flow.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    def __init__(self, base_dist, transforms):
        super(Flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)

    def log_prob(self, x):
        log_prob1 = torch.zeros(x.shape[0], device=x.device)
        log_prob2 = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            #temp = x
            x, ldj = transform(x)
            log_prob1 += ldj
        log_prob2 += self.base_dist.log_prob(x)
        return log_prob1, log_prob2, x

    def sample(self, num_samples):
        z = self.base_dist.sample(num_samples)
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

    def inversed(self, z):
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")

    def sample_within_context(self, context):
        z = context
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

class ParkInverseFlow(Flow):
    """
    This Flow is used for Park function 1

    """
    def __init__(self, base_dist, transforms1, transforms2):
        super(Flow, self).__init__()

        self.base_dist = base_dist
        self.transforms1 = nn.ModuleList(transforms1)
        self.transforms2 = nn.ModuleList(transforms2)

    def log_prob(self, x):
        """
        x1 : z1, gets sigmoid. This should fit Uniform distribution
        x2 : This will be generated given z1
        """
        x1, x2 = torch.split(x, (2, 2), dim = 1)

        log_prob1 = torch.zeros(x.shape[0], device=x.device)
        log_prob2 = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms1:
            x1, ldj = transform(x1)
            log_prob1 += ldj
        #x1, ldj = Sigmoid(x1)
        #log_prob1 += ldj
        # Skip log_prob2 of x1. It is merely uniform distribution
        for transform in self.transforms2:
            if isinstance(transform, ConditionalTransform):
                x2, ldj = transform(x2, context=x1)
            else :
                x2, ldj = transform(x2)
            log_prob1 += ldj
        x = torch.cat((x1, x2), dim=1)

        log_prob2 += self.base_dist.log_prob(x2)
        return log_prob1, log_prob2, x

    def sample(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


    def inversed(self, z):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")

class ConvenFlow(Flow):
    """
    This Flow is used for Park function 1

    """
    def __init__(self, base_dist, transforms, sigmoid=False):
        super(Flow, self).__init__()

        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.sigmoid = sigmoid
        #self.transforms2 = nn.ModuleList(transforms2)

    def log_prob(self, x):
        """
        x1 : z1, gets sigmoid. This should fit Uniform distribution
        x2 : This will be generated given z1
        """
        #x1, x2 = torch.split(x, (2, 2), dim = 1)

        log_prob1 = torch.zeros(x.shape[0], device=x.device)
        log_prob2 = torch.zeros(x.shape[0], device=x.device)
        log_prob1 += self.base_dist.log_prob(x)
        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob1 -= ldj
        #if self.sigmoid
        log_prob2 += self.base_dist.log_prob(x)
        return log_prob1, log_prob2, x

    def sample(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


    def inversed(self, z):
        log_prob1 = torch.zeros(z.shape[0], device=z.device)
        #log_prob2 = torch.zeros(z.shape[0], device=z.device)
        #log_prob1 += self.base_dist.log_prob(x)
        for transform in self.transforms:
            z, ldj = transform(z)
            log_prob1 += ldj
        #if self.sigmoid
        log_prob1 += self.base_dist.log_prob(z)
        return log_prob1, z

        #raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")

class SqFlow(Flow):
    """
    This Flow is used for Sequentially Annealed Posteriors

    ladder: used
    """

    def __init__(self, base_dist, transforms, ladd_step=3, sigmoid=False):
        super(Flow, self).__init__()

        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.sigmoid = sigmoid
        self.ladd_step = ladd_step
        #self.transforms2 = nn.ModuleList(transforms2)

    def log_prob(self, x, ladder):
        """
        log_prob 1 : ldj
        log_prob 2 : Stacks base distribution (pz)
        """
        #x1, x2 = torch.split(x, (2, 2), dim = 1)

        log_prob1_ls = []
        log_prob2_ls = []
        x_ls = []
        j = 0

        log_prob1 = torch.zeros(x.shape[0], device=x.device)
        log_prob2 = torch.zeros(x.shape[0], device=x.device)
        log_prob1 += self.base_dist.log_prob(x)

        ladder = [la * self.ladd_step for la in ladder]

        for transform in self.transforms:

            x, ldj = transform(x)
            #print(ldj)
            log_prob1 -= ldj
            log_prob2 = self.base_dist.log_prob(x)
            j += 1
            #print(x[0])
            if j in ladder:
                #print(j)
                log_prob1_ls.append(log_prob1)
                log_prob2_ls.append(log_prob2)
                x_ls.append(x)
                x = x.clone()#.detach()
                log_prob1 = log_prob1.clone()#.detach()

        return log_prob1_ls, log_prob2_ls, x_ls

    def sample(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")

    def inversed(self, z):
        log_prob1 = torch.zeros(z.shape[0], device=z.device)
        # log_prob2 = torch.zeros(z.shape[0], device=z.device)
        # log_prob1 += self.base_dist.log_prob(z)
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        log_prob1 += self.base_dist.log_prob(z)
        for transform in self.transforms:
            z, ldj = transform(z)
            log_prob1 -= ldj
        return log_prob1, z

    def unnorm_loglike(self, z):
        log_prob1 = torch.zeros(z.shape[0], device=z.device)
        for transform in self.transforms:
            z, ldj = transform(z)
            log_prob1 += ldj
        log_prob1 += self.base_dist.log_prob(z)
        return torch.sum(log_prob1)


    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


class Sq_Context_Flow(Flow):
    """
    This Flow is used for seqeuntial temperature annealing.
    """

    def __init__(self, base_dist, transforms, sigmoid=False):
        super(Flow, self).__init__()

        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.sigmoid = sigmoid
        #self.transforms2 = nn.ModuleList(transforms2)

    def log_prob(self, x, ladder):
        """
        x1 : z1, gets sigmoid. This should fit Uniform distribution
        x2 : This will be generated given z1
        """
        #x1, x2 = torch.split(x, (2, 2), dim = 1)

        log_prob1_ls = []
        log_prob2_ls = []
        x_ls = []
        j = 0

        log_prob1 = torch.zeros(x.shape[0], device=x.device)
        log_prob2 = torch.zeros(x.shape[0], device=x.device)

        ladder = [la * 3 for la in ladder]

        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                x, ldj = transform(x)
            else :
                x, ldj = transform(x)
            log_prob1 -= ldj
            log_prob2 = self.base_dist.log_prob(x)
            j += 1
            if j in ladder:
                log_prob1_ls.append(log_prob1)
                log_prob2_ls.append(log_prob2)
                x_ls.append(x)
                x = x.clone()#.detach()
                log_prob1 = log_prob1.clone()#.detach()

        return log_prob1_ls, log_prob2_ls, x_ls

    def sample(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


    def inversed(self, z):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


class MSIR_Flow(Flow):
    """
    For MSIR Flow
    """

    def __init__(self, base_dist, transforms):
        super(Flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        #assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)

    def log_prob(self, x):
        """
        x1 : z1, gets sigmoid. This should fit Uniform distribution
        x2 : This will be generated given z1
        """
        #x1, x2 = torch.split(x, (2, 2), dim = 1)

        log_prob1 = torch.zeros(x.shape[0], device=x.device)
        log_prob2 = torch.zeros(x.shape[0], device=x.device)
        #log_prob1 += self.base_dist.log_prob(x)
        for transform in self.transforms:
            #print(x)
            x, ldj = transform(x)
            log_prob1 += ldj

        log_prob2 += self.base_dist.log_prob(x)
        return log_prob1, log_prob2, x

    def sample(self, num_samples):
        z = self.base_dist.sample(num_samples)
        #z1, z2 = torch.split(z, (2,2), dim = 1)

        for transform in reversed(self.transforms):
            z = transform.inverse(z)

        return z

    def inversed(self, z):


        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        #z = torch.clamp(z, min=0.01)
        return z

class GaussianModel(Flow):

    """
    For Gaussian Coupla Approximation
    """

    def __init__(self, base_dist, dim, transforms):
        """
        transforms: YJ transforms (if needed)

        """
        super(Flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        # if isinstance(transforms, Transform): transforms = [transforms]
        # assert isinstance(transforms, Iterable)
        # assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist  # Should be independent Gaussian

        self.lower_tri = nn.Parameter(data=torch.rand(int(dim * (dim-1)/2),))
        self.tril_ind = torch.tril_indices(dim, dim, -1)  # Stores param
        self.mu = nn.Parameter(data=torch.rand(dim))

        self.transforms = nn.ModuleList(transforms)  # Should be YJ transformation
        self.lower_bound = any(transform.lower_bound for transform in transforms)


    def log_prob(self, x):
        """
        log_prob1 : final log probability of the generated samples
        log_prob2 : log p_{z} values
        """
        #print(x.shape)
        lower_fake_param = torch.zeros(x.shape[1], x.shape[1], device=x.device) + torch.eye(x.shape[1], device=x.device)
        lower_fake_param[self.tril_ind[0, :], self.tril_ind[1, :]] = self.lower_tri

        lower_fake_param = lower_fake_param[None, :, :].repeat(x.shape[0], 1, 1)
        Bz = torch.matmul(lower_fake_param, x[:, :, None])

        x = self.mu + Bz[:, :, 0]

        #print(x.shape)

        final_dist = MultivariateNormal(loc=self.mu, scale_tril=lower_fake_param[0])  # Much more stable than standard version
        log_prob1 = final_dist.log_prob(x)

        #print(log_prob1.shape)
        #raise ValueError
        if self.transforms:
            for transform in self.transforms:
                x, ldj = transform(x)
                log_prob1 += ldj

        # No need to evaluate this value
        log_prob2 = torch.zeros(x.shape[0], device=x.device)

        return log_prob1, log_prob2, x

    def sample(self, num_samples):
        raise NotImplementedError
        #z = self.base_dist.sample(num_samples)
        #z1, z2 = torch.split(z, (2,2), dim = 1)

        #for transform in reversed(self.transforms):
        #    z = transform.inverse(z)

        #return z

    def inversed(self, z):
        raise NotImplementedError

        #for transform in reversed(self.transforms):
        #    z = transform.inverse(z)
        #z = torch.clamp(z, min=0.01)
        #return z

import torch
import torch.nn as nn
import torch.nn.functional as F
from survae.nn.layers import LambdaLayer
from survae.nn.layers import act_module



class MLP(nn.Sequential):
    def __init__(self, input_size, output_size, hidden_units, activation='relu', in_lambda=None, out_lambda=None):
        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(act_module(activation))
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(hidden_units[-1], output_size))
        if out_lambda: layers.append(LambdaLayer(out_lambda))

        super(MLP, self).__init__(*layers)

class NDMLP(nn.Sequential):
    def __init__(self, input_size, output_size, hidden_units, activation='relu', in_lambda=None, out_lambda=None):
        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(act_module(activation))
            #layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(hidden_units[-1], output_size))
        if out_lambda: layers.append(LambdaLayer(out_lambda))

        super(NDMLP, self).__init__(*layers)

class ND_2MLP(nn.Sequential):
    def __init__(self, input_size, output_size, hidden_units, activation='relu', in_lambda=None, out_lambda=None):
        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(act_module(activation))
            layers.append(nn.Dropout(p=0.05))
        layers.append(nn.Linear(hidden_units[-1], output_size))
        if out_lambda: layers.append(LambdaLayer(out_lambda))

        super(ND_2MLP, self).__init__(*layers)

class ND_Spline_2MLP(nn.Sequential):
    def __init__(self, input_size, output_size, hidden_units, activation='relu', in_lambda=None, out_lambda=None):
        layers = []
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(act_module(activation))
            #layers.append(nn.Dropout(p=0.05))
        layers.append(nn.Linear(hidden_units[-1], output_size))
        if out_lambda: layers.append(LambdaLayer(out_lambda))
        super(ND_Spline_2MLP, self).__init__(*layers)

hidden_units = [200, 200, 200, 200]
activation = "relu"
param_scale_fn = "exp"

def Build_Transform(D = 4, P = 2, num_flows = 8, affine = True):
    transforms = []
    #transforms.append(ActNormBijection(D))![](results/MSIR/generated_sample_epoch0_interval218.png)![](results/MSIR/generated_sample_epoch0_interval219.png)![](results/MSIR/generated_sample_epoch0_interval220.png)

    for _ in range(num_flows):
        net = nn.Sequential(MLP(D//2, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        if affine: transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(param_scale_fn)))
        else:           transforms.append(AdditiveCouplingBijection(net))
        transforms.append(Reverse(D))
    transforms.pop()
    return transforms

from .layers import Shuffle_with_Order, Order_Shuffle, create_3dim_sort, create_5dim_sort, create_7dim_sort, create_11dim_sort

def Build_Shuffle_Order_Transform(D = 10, P = 2, num_flows = 8, steps =4, affine = True):
    Shuffle_list = []
    #idx = create_idx(dim_size=D, num= num_flows, per=steps)
    #print(len(idx))
    for j in range(num_flows):
        Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j+1, step=steps))

    transforms = []

    for flow_idx in range(num_flows):
        net = nn.Sequential(ND_2MLP(D//2 + D%2, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        if affine: transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(param_scale_fn)))
        else:           transforms.append(AdditiveCouplingBijection(net))
        transforms.append(Shuffle_list[flow_idx])
        #transforms.append(BoundSurjection())

    return transforms

def Build_Shuffle_Order_3_Transform(D = 3, P = 2, num_flows = 6, steps =6, affine = True):
    Shuffle_list = []
    idx = create_3dim_sort()
    #print(len(idx))
    for j in range(num_flows):
        #Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j+1, step=steps))
        #print(idx[j][0].shape)
        Shuffle_list.append(Order_Shuffle(order=idx[j%10][0], dim=1))

    transforms = []

    for flow_idx in range(num_flows):
        net = nn.Sequential(MLP(D//2 + D%2, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        if affine: transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(param_scale_fn)))
        else:           transforms.append(AdditiveCouplingBijection(net))
        transforms.append(Shuffle_list[flow_idx])
        #transforms.append(BoundSurjection())

    return transforms


def Build_Shuffle_Order_5_Transform(D = 3, P = 2, num_flows = 6, steps =10, affine = True):
    Shuffle_list = []
    idx = create_5dim_sort()
    #print(len(idx))
    for j in range(num_flows):
        #Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j+1, step=steps))
        #print(idx[j][0].shape)
        Shuffle_list.append(Order_Shuffle(order=idx[j%10][0], dim=1))

    transforms = []

    for flow_idx in range(num_flows):
        net = nn.Sequential(ND_2MLP(D//2 + D%2, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        if affine: transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(param_scale_fn)))
        else:           transforms.append(AdditiveCouplingBijection(net))
        transforms.append(Shuffle_list[flow_idx])
        #transforms.append(BoundSurjection())

    return transforms

def Build_Shuffle_Order_7_Transform(D = 7, P = 2, num_flows = 6, steps =10, affine = True):
    Shuffle_list = []
    idx = create_7dim_sort()
    #print(len(idx))
    for j in range(num_flows):
        #Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j+1, step=steps))
        #print(idx[j][0].shape)
        Shuffle_list.append(Order_Shuffle(order=idx[j%10][0], dim=1))

    transforms = []

    for flow_idx in range(num_flows):
        net = nn.Sequential(ND_2MLP(D//2 + D%2, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        if affine: transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(param_scale_fn)))
        else:           transforms.append(AdditiveCouplingBijection(net))
        transforms.append(Shuffle_list[flow_idx])
        #transforms.append(BoundSurjection())

    return transforms

def Build_Shuffle_Transform(D = 4, P = 2, num_flows = 8, affine = True):
    transforms = []
    #transforms.append(ActNormBijection(D))
    #transforms.append(ShiftBijection(shift=torch.tensor([[0.0, 1.0, 0.2, 0.4]], requires_grad=True)))

    for _ in range(num_flows):
        net = nn.Sequential(MLP(D//2 + D%2, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        if affine: transforms.append(AffineCouplingBijection(net, scale_fn=scale_fn(param_scale_fn)))
        else:           transforms.append(AdditiveCouplingBijection(net))

        #transforms.append(ActNormBijection(D))
        transforms.append(Shuffle(D, dim=1))
       #transforms.append(StochasticPermutation(dim=1))
        #transforms.append(Reverse(D))

    #transforms.pop()


    return transforms


from survae.transforms import LogisticMixtureCouplingBijection
def Build_Plus_Transform(D = 4, P = 3, num_flows = 8, affine = True):
    transforms = []
    num_mixtures = 3
    P *= num_mixtures
    #print(P)
    for _ in range(num_flows):
        net = nn.Sequential(MLP(D//2, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        if affine: transforms.append(LogisticMixtureCouplingBijection(net, num_mixtures=num_mixtures))
        else:           transforms.append(AdditiveCouplingBijection(net))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        transforms.append(StochasticPermutation(dim=1))
    transforms.pop()
    return transforms

from survae.transforms import CubicSplineCouplingBijection, Sigmoid

def Build_Cubic_Transform(D = 4, P = 2, num_flows = 8):
    transforms = []
    P = 22
    for _ in range(num_flows):
        transforms.append(Sigmoid())
        net = nn.Sequential(ND_Spline_2MLP(D//2, D//2 *P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        transforms.append(CubicSplineCouplingBijection(net, num_bins=10))

        transforms.append(ActNormBijection(D, data_dep_init=False))
        #transforms.append(ActNormBijection(D, data_dep_init=False))
        transforms.append(Shuffle(D, dim=1))
    #transforms.append(Logit())
        #transforms.append(StochasticPermutation(dim=1))
    #transforms.pop()
    #transforms.append(ActNormBijection(D, data_dep_init=False))

    return transforms

from survae.transforms import AffineAutoregressiveBijection, AutoregressiveBijection
import torch


from torch import nn
from torch.nn import functional as F, init


def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(self,
                 in_degrees,
                 out_features,
                 autoregressive_features,
                 random_mask,
                 is_output,
                 bias=True):
        super().__init__(
            in_features=len(in_degrees),
            out_features=out_features,
            bias=bias)
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', degrees)

    @classmethod
    def _get_mask_and_degrees(cls,
                              in_degrees,
                              out_features,
                              autoregressive_features,
                              random_mask,
                              is_output):
        if is_output:
            out_degrees = tile(
                _get_input_degrees(autoregressive_features),
                out_features // autoregressive_features
            )
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long)
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.
    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        if context_features is not None:
            raise NotImplementedError()

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if context is not None:
            raise NotImplementedError()

        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 zero_initialization=True):
        if random_mask:
            raise ValueError('Masked residual block can\'t be used with random masks.')
        super().__init__()
        features = len(in_degrees)

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])

        # Masked linear.
        linear_0 = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        linear_1 = MaskedLinear(
            in_degrees=linear_0.degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError('In a masked residual block, the output degrees can\'t be'
                               ' less than the corresponding input degrees.')

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-1e-3, b=1e-3)
            init.uniform_(self.linear_layers[-1].bias, a=-1e-3, b=1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(
                torch.cat((temps, self.context_layer(context)), dim=1),
                dim=1
            )
        return inputs + temps



from torch import nn
from torch.nn import functional as F, init


def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(self,
                 in_degrees,
                 out_features,
                 autoregressive_features,
                 random_mask,
                 is_output,
                 bias=True,
                 zero_initialization = True):
        super().__init__(
            in_features=len(in_degrees),
            out_features=out_features,
            bias=bias)
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output)
        self.register_buffer('mask', mask)
        self.register_buffer('degrees', degrees)

        if zero_initialization:
            #init_weights(self.linear_layers)
            #for layers in self.linear_layers:
            init.zeros_(self.weight)
            init.zeros_(self.bias)

    @classmethod
    def _get_mask_and_degrees(cls,
                              in_degrees,
                              out_features,
                              autoregressive_features,
                              random_mask,
                              is_output):
        if is_output:
            out_degrees = tile(
                _get_input_degrees(autoregressive_features),
                out_features // autoregressive_features
            )
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long)
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.
    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        if context_features is not None:
            raise NotImplementedError()

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if context is not None:
            raise NotImplementedError()

        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(self,
                 in_degrees,
                 autoregressive_features,
                 context_features=None,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 zero_initialization=True):
        if random_mask:
            raise ValueError('Masked residual block can\'t be used with random masks.')
        super().__init__()
        features = len(in_degrees)

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])

        # Masked linear.
        linear_0 = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        linear_1 = MaskedLinear(
            in_degrees=linear_0.degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError('In a masked residual block, the output degrees can\'t be'
                               ' less than the corresponding input degrees.')

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
        if zero_initialization:
            #init_weights(self.linear_layers)
            for layers in self.linear_layers:
                init.zeros_(layers.weight)
                init.zeros_(layers.bias)
            # init.uniform_(self.linear_layers[-1].weight, a=-1e-6, b=1e-6)
            # init.uniform_(self.linear_layers[-1].bias, a=-1e-6, b=1e-6)
            # nn.init.zeros_(self.linear_layers[-1].weight)
            # nn.init.zeros_(self.linear_layers[-2].weight)
            # nn.init.zeros_(self.linear_layers[-1].bias)
    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(
                torch.cat((temps, self.context_layer(context)), dim=1),
                dim=1
            )
        return inputs + temps


class MADE(nn.Module):
    """Implementation of MADE.
    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 output_multiplier=1,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        if use_residual_blocks and random_mask:
            raise ValueError('Residual blocks can\'t be used with random masks.')
        super().__init__()

        # Initial layer.
        self.initial_layer = MaskedLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)

        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True
        )

    def forward(self, inputs, context=None):
        outputs = self.initial_layer(inputs)
        if context is not None:
            outputs += self.context_layer(context)
        for block in self.blocks:
            outputs = block(outputs, context)
        outputs = self.final_layer(outputs)
        return outputs

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0.00)

class MaskedAffineAutoregressiveTransform(AutoregressiveBijection):
    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        self.features = features
        net = MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        init_weights(net)
        super(MaskedAffineAutoregressiveTransform, self).__init__(autoregressive_net=net, autoregressive_order='ltr')

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)

        scale = torch.sigmoid(unconstrained_scale + 1e-2) + 0.5 #+ 1e-3
        #print(scale, shift)
        #raise ValueError
        #scale = torch.exp(unconstrained_scale)
        #print(scale)
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = utils.sum_except_batch(log_scale)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 2.) + 1e-3
        #scale = torch.exp(unconstrained_scale)
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -utils.sum_except_batch(log_scale)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        return unconstrained_scale, shift

    def forward(self, x):
        #print(self.autoregressive_net)
        elementwise_params = self.autoregressive_net(x)
        z, ldj = self._elementwise_forward(x, elementwise_params)
        return z, ldj

def Build_Auto_Transform(D = 4, P = 2, num_flows = 8):
    steps = 2
    Shuffle_list = []
    for j in range(num_flows):
        Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j + 1, step=steps))

    transforms = []
    #transforms += [ScaleBijection(scale=torch.tensor([[1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3,
    #                                                   1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3]]))]

    for flow_idx in range(num_flows):
        transforms.append(MaskedAffineAutoregressiveTransform(D, 3, num_blocks=1,
                 use_residual_blocks=True, random_mask=False))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        transforms.append(Shuffle_list[flow_idx])
        #transforms.append(Reverse(D))


    return transforms





import torch
from survae.distributions import ConditionalDistribution
from survae.transforms import RationalQuadraticSplineCouplingBijection, CouplingBijection, splines

class RationalQuadraticSplineCouplingBijection(CouplingBijection):

    def __init__(self, coupling_net, num_bins, split_dim=1, num_condition=None):
        super(RationalQuadraticSplineCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 3 * self.num_bins + 1

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = elementwise_params[..., 2*self.num_bins:]
        try:
            z, ldj_elementwise = splines.unconstrained_rational_quadratic_spline(x,
                                                               unnormalized_widths=unnormalized_widths,
                                                               unnormalized_heights=unnormalized_heights,
                                                               unnormalized_derivatives=unnormalized_derivatives,
                                                               inverse=False, tail_bound=2)
        except RuntimeError as e:
            print(x)

        ldj = utils.sum_except_batch(ldj_elementwise)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = elementwise_params[..., 2*self.num_bins:]
        x, _ = splines.rational_quadratic_spline(z,
                                                 unnormalized_widths=unnormalized_widths,
                                                 unnormalized_heights=unnormalized_heights,
                                                 unnormalized_derivatives=unnormalized_derivatives,
                                                 inverse=True)
        return x


def Build_Spline_Transform(D = 4, num_bin = 10, num_flows = 8):
    steps = 2
    Shuffle_list = []
    for j in range(num_flows):
        Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j + 1, step=steps))

    transforms = []
    P = num_bin * 3 + 1


    #transforms.append(Sigmoid())
    #transforms += [ShiftBijection(shift=torch.tensor([[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]))]
    #transforms += [ScaleBijection(scale=torch.tensor([[1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3,
    #                                                   1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3]]))]

    for flow_idx in range(num_flows):
        transforms.append(ActNormBijection(D, data_dep_init=False))
        net = nn.Sequential(ND_2MLP(D//2, D//2 *P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        net.apply(init_weights)
        transforms.append(RationalQuadraticSplineCouplingBijection(coupling_net=net, num_bins=num_bin))
        transforms.append(Shuffle_list[flow_idx])
    return transforms


def Build_Spline_Order_5_Transform(D = 3, P = 2, num_flows = 6, num_bin=16):
    Shuffle_list = []
    idx = create_5dim_sort()
    #print(len(idx))
    for j in range(num_flows):
        #Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j+1, step=steps))
        #print(idx[j][0].shape)
        Shuffle_list.append(Order_Shuffle(order=idx[j%10][0], dim=1))

    transforms = []
    P = num_bin * 3 + 1
    for flow_idx in range(num_flows):
        net = nn.Sequential(ND_2MLP(D//2 + D%2, D//2 * P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        #net.apply(init_weights)
        transforms.append(RationalQuadraticSplineCouplingBijection(coupling_net=net, num_bins=num_bin))

        transforms.append(Shuffle_list[flow_idx])

    return transforms

from .autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform

def Build_Spline_Order_11_Transform(D = 3, P = 2, num_flows = 6, num_bin=16):
    Shuffle_list = []
    idx = create_11dim_sort()
    #print(len(idx))
    for j in range(num_flows):
        #Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j+1, step=steps))
        Shuffle_list.append(Order_Shuffle(order=idx[j%4][0], dim=1))

    transforms = []
    P = num_bin * 3 + 1

    #transforms.append(Sigmoid())
    #transforms += [ShiftBijection(shift=torch.tensor([[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]))]
    #transforms += [ScaleBijection(scale=torch.tensor([[1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3,
    #                                                   1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3]]))]

    for flow_idx in range(num_flows):
        transforms.append(ActNormBijection(D, data_dep_init=False))
        #net.apply(init_weights)
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=D, hidden_features=256, context_features=None,
            num_bins=num_bin,
            tails='linear', use_residual_blocks=True
        ))
        #transforms.append(RationalQuadraticSplineCouplingBijection(coupling_net=net, num_bins=num_bin))
        transforms.append(Shuffle_list[flow_idx])

    return transforms

def Build_Cond_Order_Transform(D=10, P=2, W=118, num_flows=5, steps=2, affine=True):
    transforms=[]
    hidden_units = [200]
    Shuffle_list = []

    for j in range(num_flows):
        Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j+1, step=steps))

    for flow_idx in range(num_flows):
        transforms.append(Sigmoid())
        net = nn.Sequential(NDMLP(D//2 + D%2 + W, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        if affine: transforms.append(ConditionalAffineCouplingBijection(net, scale_fn=scale_fn(param_scale_fn)))
        else:           transforms.append(AdditiveCouplingBijection(net))
        transforms.append(Logit())
        transforms.append(Shuffle_list[flow_idx])

    return transforms


if __name__ == "__main__":
    P = 30
    D = 10
    get_in = torch.ones(size=(1, D//2))
    net = nn.Sequential(ND_2MLP(D // 2, D // 2 * P,
                                hidden_units=hidden_units,
                                activation=activation),
                        ElementwiseParams(P))
    net.apply(init_weights)
    res = torch.exp(net(get_in))/torch.sum(torch.exp(net(get_in)))
    # print(torch.exp(net(get_in)))