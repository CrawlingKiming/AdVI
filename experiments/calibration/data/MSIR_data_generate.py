import numpy as np
import torch
from torch.distributions import log_normal, multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal

from survae.distributions import StandardNormal, StandardUniform

def data_sim(datasize = 3):
    #datasize = 1e3
    data_list = []
    SMALL = 1e-5
    for i in range(datasize):
        """        x1 = log_normal.LogNormal(torch.tensor([0.0]), torch.tensor([0.25])).sample()

        x2 = torch.normal(torch.tensor([2.0]), torch.tensor([0.7]))
        x3 = log_normal.LogNormal(torch.tensor([-1.0]), torch.tensor([0.1])).sample()
        #x3 = torch.normal(torch.tensor([0.0]), torch.tensor([1.0])) * (x2)
        x4 = torch.normal(torch.tensor([0.0]), torch.tensor([0.6]))
        """
        #mean = torch.Tensor([[0.0, 0.0]])
        #cov = torch.Tensor([[0.5, 0.0], [0.0, 0.5]])
        x1 = torch.normal(torch.tensor([0.5]), torch.tensor([0.2]))
        #x1 = torch.rand(1) + SMALL
        x2 =  torch.normal(torch.tensor([np.pi + 1.6]), torch.tensor([1.2]))
        #x3 = torch.rand(1) + SMALL
        x3 = torch.normal(torch.tensor([0.117]), torch.tensor([0.06]))
        x4 = torch.normal(torch.tensor([0.6]), torch.tensor([0.2]))

        y1 = x1
        #y2 = x2 *2 *np.pi + 2
        y2 = x2
        y3 = x3
        #y4 = x4 * 1.1
        y4 = x4
        #y = torch.cat((x1,x2,x3,x4))
        y = torch.cat((y1, y2, y3, y4))

        #print(y.shape)
        #print(y1.shape)
        #raise ValueError
        data_list.append(y.detach().cpu().numpy())

        #print(i)


    return np.asarray(data_list)

import matplotlib.pyplot as plt

training = True

if training :

    robotics = data_sim(datasize=8 * 1000)
    print(np.max(robotics[:, -2]), np.min(robotics[:, -2]))
    print(np.max(robotics[:, -1]), np.min(robotics[:, -1]))
    fig, axs = plt.subplots(2,2)
    fig.suptitle("MSIR")

    axs[0,0].hist(robotics[:,0], bins=128)
    axs[0,0].set_title("Y1")
    axs[0,1].hist(robotics[:,1], bins=128)
    axs[0,1].set_title("Y2 ")
    axs[1,0].hist(robotics[:,2], bins=256)
    axs[1,0].set_title("Y3")
    axs[1,1].hist(robotics[:,3], bins=128)
    axs[1,1].set_title("Y4")
    plt.show()


    np.save("MSIR_unif_sample",robotics)

else :
    robotics = data_sim(datasize=500)
    np.save("park_plain_sample_test", robotics)
#print(a)



