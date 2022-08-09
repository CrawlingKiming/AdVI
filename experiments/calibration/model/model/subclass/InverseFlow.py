import torch
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform, ConditionalTransform
from survae.flows import Flow

import torch.nn as nn
from survae.flows import Flow
from survae.distributions import StandardNormal, StandardUniform
from survae.transforms import AdditiveCouplingBijection, AffineCouplingBijection, ActNormBijection, Reverse, ConditionalAffineCouplingBijection, StochasticPermutation,Shuffle
from survae.transforms import Logit, SoftplusInverse, Sigmoid
from survae.nn.nets import MLP
from survae.nn.layers import ElementwiseParams, scale_fn
from .layers import ShiftBijection, ScaleBijection, ActShiftBijection



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
        log_prob1 += self.base_dist.log_prob(x)

        ladder = [la * 3 for la in ladder]

        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob1 -= ldj
            log_prob2 = self.base_dist.log_prob(x)
            j += 1
            #print(x[0])
            if j in ladder:
                #print(j)
                log_prob1_ls.append(log_prob1)
                log_prob2_ls.append(log_prob2)
                x_ls.append(x)
                x = x.clone().detach()
                log_prob1 = log_prob1.clone().detach()
        #print(x_ls)
                #print(j)

        #print(len(x_ls))

        #log_prob2 = self.base_dist.log_prob(x)
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
                x = x.clone().detach()
                log_prob1 = log_prob1.clone().detach()

        return log_prob1_ls, log_prob2_ls, x_ls

    def sample(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


    def inversed(self, z):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")


class ParkFlow(Flow):
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
        #assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)

    def log_prob(self, x):
        x1, x2 = torch.split(x, (2, 2), dim = 1)
        log_prob1 = torch.zeros(x2.shape[0], device=x.device)
        log_prob2 = torch.zeros(x2.shape[0], device=x.device)
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                x2, ldj = transform(x2, context=x1)
            #temp = x
            else :
                x2, ldj = transform(x2)
            log_prob1 += ldj

        x = torch.cat((x1, x2), dim=1)
        log_prob2 += self.base_dist.log_prob(x)

        return log_prob1, log_prob2, x

    def sample(self, num_samples):
        z1 = torch.rand(size = (num_samples,2))
        z2 = self.base_dist.sample(num_samples)
        #z1, z2 = torch.split(z, (2,2), dim = 1)

        for transform in reversed(self.transforms):

            if isinstance(transform, ConditionalTransform):
                z2 = transform.inverse(z2, context=z1)
            #temp = x
            else :
                z2 = transform.inverse(z2)
                #z2 = transform.inverse(z2)
        z = torch.cat((z1, z2), dim = 1)
        return z

    def inversed(self, z):

        z1, z2 = torch.split(z, (2, 2), dim=1)

        for transform in reversed(self.transforms):

            if isinstance(transform, ConditionalTransform):
                z2 = transform.inverse(z2, context=z1)
            #temp = x
            else :
                z2 = transform.inverse(z2)
                #z2 = transform.inverse(z2)
        z = torch.cat((z1, z2), dim = 1)
        return z

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")

    def sample_within_context(self, context):
        z = context
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z


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
            layers.append(nn.Dropout(p=0.0))
        layers.append(nn.Linear(hidden_units[-1], output_size))
        if out_lambda: layers.append(LambdaLayer(out_lambda))

        super(ND_2MLP, self).__init__(*layers)
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

from .layers import Shuffle_with_Order, create_idx, Order_Shuffle, create_3dim_sort, create_5dim_sort, create_7dim_sort
def Build_Shuffle_Order_Transform(D = 10, P = 2, num_flows = 8, steps =4, affine = True):
    Shuffle_list = []
    #idx = create_idx(dim_size=D, num= num_flows, per=steps)
    #print(len(idx))
    for j in range(num_flows):
        Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j+1, step=steps))

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

from survae.transforms import AffineAutoregressiveBijection
def Build_Auto_Transform(D = 4, P = 2, num_flows = 8, affine = True):
    transforms = []

    for _ in range(num_flows):
        net = nn.Sequential(MLP(D, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        if affine: transforms.append(AffineAutoregressiveBijection(net))
        else:           transforms.append(AffineAutoregressiveBijection(net))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        #transforms.append(Shuffle(D, dim=1))

        #transforms.append(StochasticPermutation(dim=1))
    #transforms.pop()
    #transforms.append(ActNormBijection(D, data_dep_init=False))

    return transforms

import torch
from survae.distributions import ConditionalDistribution
from survae.transforms.surjections import Surjection





def Build_Cond_Order_Transform(D=10, P=2, W=118, num_flows=5, steps=2, affine=True):
    transforms=[]
    hidden_units = [200]
    Shuffle_list = []

    for j in range(num_flows):
        Shuffle_list.append(Shuffle_with_Order(dim_size=D, idx=j+1, step=steps))

    for flow_idx in range(num_flows):
        net = nn.Sequential(NDMLP(D//2 + D%2 + W, P,
                                hidden_units=hidden_units,
                                activation=activation),
                            ElementwiseParams(P))
        transforms.append(ActNormBijection(D, data_dep_init=False))
        if affine: transforms.append(ConditionalAffineCouplingBijection(net, scale_fn=scale_fn(param_scale_fn)))
        else:           transforms.append(AdditiveCouplingBijection(net))

        transforms.append(Shuffle_list[flow_idx])

    return transforms