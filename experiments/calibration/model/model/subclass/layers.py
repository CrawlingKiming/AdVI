import math
import torch
import torch.nn.functional as F
from survae.transforms import Surjection, Bijection, StochasticTransform
from survae.utils import sum_except_batch

from survae.transforms import Permute



class ElementAbsSurjection(Surjection):
    stochastic_forward = False

    def __init__(self, classifier, element=0):
        super(ElementAbsSurjection, self).__init__()
        self.classifier = classifier
        self.element = element

    def forward(self, x):
        s = (x[:, self.element].sign()+1)/2
        z = x
        z[:, self.element] = x[:, self.element].abs()
        logit_pi = self.classifier(z)
        ldj = sum_except_batch(-F.binary_cross_entropy_with_logits(logit_pi, s, reduction='none'))
        return z, ldj

    def inverse(self, z):
        logit_pi = self.classifier(z)
        s = torch.bernoulli(torch.sigmoid(logit_pi))
        x = z
        x[:, self.element] = (2*s-1)*x[:, self.element]
        return x


class ScaleBijection(Bijection):

    def __init__(self, scale):
        super(ScaleBijection, self).__init__()
        self.register_buffer('scale', scale)

    @property
    def log_scale(self):
        return torch.log(torch.abs(self.scale)).sum()

    def forward(self, x):
        z = x * self.scale
        ldj = x.new_ones(x.shape[0]) * self.log_scale
        return z, ldj

    def inverse(self, z):
        x = z / self.scale
        return x


class Exp(Bijection):

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        z = torch.exp(x)
        ldj = sum_except_batch(x)
        return z, ldj

    def inverse(self, z):
        x = z / self.scale
        return x

class ShiftBijection(Bijection):

    def __init__(self, shift):
        super(ShiftBijection, self).__init__()
        self.register_buffer('shift', shift)

    def forward(self, x):
        z = x + self.shift
        #print(self.shift)
        ldj = x.new_zeros(x.shape[0])
        return z, ldj

    def inverse(self, z):
        x = z - self.shift
        return x

class ActShiftBijection(Bijection):

    def __init__(self, shift):
        super(ActShiftBijection, self).__init__()
        self.shift = shift
        #self.register_buffer('shift', shift)

    def forward(self, x):
        z = x + self.shift
        ldj = x.new_zeros(x.shape[0])
        return z, ldj

    def inverse(self, z):
        x = z - self.shift
        return x

class Shuffle_with_Order(Permute):

    def __init__(self, dim_size, idx, step = 1, dim=1):
        order = torch.linspace(0, dim_size-1,steps=dim_size, dtype=torch.int64)
        order = torch.roll(order, idx*step)
        #print(order.shape)
        super(Shuffle_with_Order, self).__init__(order , dim)

class Order_Shuffle(Permute):
    def __init__(self, order, dim=1):
        #print(order.shape)
        super(Order_Shuffle, self).__init__(order, dim)

class GaussianModel(Bijection):

    """
    For Gaussian Coupla Approximation
    """

    def __init__(self, dim):
        """
        transforms: YJ transforms (if needed)

        """
        super(Bijection, self).__init__()
        #assert isinstance(base_dist, Distribution)
        # if isinstance(transforms, Transform): transforms = [transforms]
        # assert isinstance(transforms, Iterable)
        # assert all(isinstance(transform, Transform) for transform in transforms)
        #self.base_dist = base_dist  # Should be independent Gaussian

        self.lower_tri = torch.nn.Parameter(data=torch.rand(int(dim * (dim-1)/2),))
        self.tril_ind = torch.tril_indices(dim, dim, -1)  # Stores param
        self.mu = torch.nn.Parameter(data=torch.rand(dim))

        lower_fake_param = torch.zeros(x.shape[1], x.shape[1], device=x.device) + torch.eye(x.shape[1], device=x.device)
        lower_fake_param[self.tril_ind[0, :], self.tril_ind[1, :]] = self.lower_tri

        lower_fake_param = lower_fake_param[None, :, :].repeat(x.shape[0], 1, 1)

        self.final_dist = MultivariateNormal(loc=self.mu,
                                        scale_tril=lower_fake_param[0])  # Much more stable than standard version

        #self.transforms = nn.ModuleList(transforms)  # Should be YJ transformation
        #self.lower_bound = any(transform.lower_bound for transform in transforms)


    def log_prob(self, x):
        """
        log_prob1 : final log probability of the generated samples
        log_prob2 : log p_{z} values
        """
        #print(x.shape)
        Bz = torch.matmul(lower_fake_param, x[:, :, None])

        x = self.mu + Bz[:, :, 0]

        #print(x.shape)

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

def create_idx(dim_size, num, per):
    assert num % per == 0
    shuffle_list = []
    for i in range(num):
        shuffle_list.append(torch.randperm(dim_size).view(1,-1))
    #print(shuffle_list[0])
    for j in range(num // per):
        #print(j)
        temp1 = torch.linspace(start=0, end=dim_size-1, steps=dim_size, dtype=torch.int64).view(1, -1)
        for i in range(per*(j), per*(j+1)-1):
            temp1 = torch.index_select(input=temp1, dim=1, index=shuffle_list[i][0])
        shuffle_list[per*(j+1)-1] = torch.argsort(temp1, dim=1)
    return shuffle_list

def compute_idx(dim_size, num, per):
    assert num % per == 0
    shuffle_list = []
    for i in range(num):
        shuffle_list.append(torch.randperm(dim_size).view(1,-1))
    temp_list = []
    temp1 = torch.linspace(start=0, end=dim_size - 1, steps=dim_size, dtype=torch.int64).view(1, -1)
    for j in range(num):
        temp1 = torch.index_select(input=temp1, dim=1, index=shuffle_list[j][0])
        temp_list.append(temp1)
    return shuffle_list, temp_list


def create_3dim_sort():
    shuffle_list=[torch.tensor([[0, 1, 2]]), torch.tensor([[2, 0, 1]]), torch.tensor([[1, 2, 0]]), torch.tensor([[1, 2, 0]]),
                  torch.tensor([[1, 2, 0]]), torch.tensor([[2, 0, 1]]), torch.tensor([[0, 2, 1]]), torch.tensor([[2, 0, 1]]),
                  torch.tensor([[2, 1, 0]]), torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 2]])]
    return shuffle_list

def create_5dim_sort():
    shuffle_list = [torch.tensor([[4, 3, 1, 0, 2]]), torch.tensor([[0, 2, 4, 3, 1]]), torch.tensor([[3, 1, 2, 0, 4]]),
                    torch.tensor([[4, 1, 3, 0, 2]]), torch.tensor([[2, 0, 4, 1, 3]]), torch.tensor([[2, 4, 1, 0, 3]]),
                    torch.tensor([[3, 1, 0, 2, 4]]), torch.tensor([[0, 1, 4, 2, 3]]), torch.tensor([[0, 3, 2, 4, 1]]),
                    torch.tensor([[4, 2, 1, 3, 0]])]
    return shuffle_list

def create_7dim_sort():
    shuffle_list = [torch.tensor([[1, 6, 0, 2, 4, 3, 5]]), torch.tensor([[5, 4, 6, 1, 2, 0, 3]]), torch.tensor([[4, 5, 6, 2, 3, 0, 1]]),
                    torch.tensor([[6, 5, 3, 2, 1, 4, 0]]), torch.tensor([[4, 3, 0, 2, 5, 1, 6]]), torch.tensor([[0, 5, 3, 6, 4, 1, 2]]),
                    torch.tensor([[6, 3, 4, 0, 2, 5, 1]]), torch.tensor([[1, 3, 2, 5, 6, 4, 0]]), torch.tensor([[2, 3, 0, 4, 6, 1, 5]]),
                    torch.tensor([[2, 5, 1, 3, 4, 6, 0]])]
    return shuffle_list

def create_11dim_sort():
    shuffle_list = [torch.tensor([[8, 3, 6, 2, 7, 0, 1, 5, 9, 4, 10]]), torch.tensor([[6, 1, 10, 7, 2, 3, 5, 4, 9, 0, 8]]),
                    torch.tensor([[0, 4, 3, 6, 1, 10, 2, 5, 8, 7, 9]]), torch.tensor([[3, 0, 7, 4, 8, 2, 1, 9, 10, 5, 6]])]
    return shuffle_list

if __name__ == "__main__":

    Shuffle_list = []
    dim_size = 11
    s, t = compute_idx(11,4,1)
    print(s)
    print(t)

    """
    for j in range(10):
        Shuffle_list.append(Shuffle_with_Order(dim_size=7, idx=j+1, step=2))

    t=torch.linspace(0, 3 - 1, steps=3, dtype=torch.int64)
    print(t.shape)
    sort = torch.tensor([[0,1,2,3,4,5,6]])
    sort2 = torch.tensor([[0,1,2]])
    sorted_res = torch.index_select(sort, dim=1, index=sort2[0])
    #for shuffle in Shuffle_list:
    #    print(shuffle(sort))
    #print(sorted_res)
    #print(sorted_res[:,sort])
    #sorted_res[:,sorted_res] = sorted_res[:,sort]
    #print(sorted_res)
    """
    #Build Ordered Permutation Shuffle
    """
    shuffle_list = create_idx(7, 10, 10)
    print(shuffle_list)
    for idx in shuffle_list:
        #print(idx)
        #print(idx[0].shape)
        sort = torch.index_select(input=sort, index=idx[0], dim=1)
        print(sort)
    """

