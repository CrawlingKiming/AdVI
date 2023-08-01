import numpy as np
import torch

class Fourier():
    def __init__(self, T=66, N=7):
        """Calculate the Fourier Basis & Coeff"""
        self.basis = self.fourierbasis(T=T, N=N) # 14, 66

    def fourierbasis(self, T=66, N=7):
        t = torch.linspace(0, T - 1, steps=T)

        coeff = []
        for n in range(1, N + 1):
            an = torch.cos(2 * torch.pi * n * t / T)
            bn = torch.sin(2 * torch.pi * n * t / T)
            c = torch.stack((an, bn))
            coeff.append(c)
        return torch.cat(coeff) / np.sqrt(T // 2)

    def coeff(self, data):
        basis = self.basis.to(data.device) # 14 66
        b_hat = torch.linalg.inv(basis @ basis.T) @ basis @ (data.T)
        return b_hat

if __name__ == "__main__":
    test_data = np.genfromtxt("C:/Users/user/PycharmProjects/AdvVI/experiments/calibration/data/mRNA/mRNA_data.csv")
    test_data = torch.tensor(data=test_data[[0], :], dtype=torch.float32)
    fou = Fourier()
    B = 5
    print(fou.basis.shape)
    test_data = test_data.repeat(B, 1)
    b_hat = fou.coeff(test_data) # 14, B
    b_hat = b_hat.reshape(7, 2, B)
    print(b_hat)
    lambd = torch.sum(torch.square(b_hat), 1)
    lambd = lambd[:, 0]
    print(lambd.shape)
    # np.save("C:/Users/user/PycharmProjects/AdvVI/experiments/calibration/data/mRNA/mRNA_sk", lambd)
