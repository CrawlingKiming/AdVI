import numpy as np
import matplotlib.pyplot as plt

burden_estimate = np.load("SIR.npy")
print("Data Shape :", burden_estimate.shape)
SIR_list = []

for i in range(100):
    np.random.seed(i)
    rand = np.random.randn(18,3) * 0.1
    rand[[0], :] = 0
    #rand[-1, :] = 0
    obs_SIR = burden_estimate + rand
    SIR_list.append(obs_SIR)
    np.save(arr=obs_SIR, file="SIR_{}".format(i))

#SIR_0 = np.load("SIR_0.npy")
#SIR_49 = np.load("SIR_49.npy")
SIR = np.asarray(SIR_list)
SIR_0 = np.percentile(SIR, 97.5, axis=0)
SIR_95 = np.percentile(SIR, 2.5, axis=0)
fig, ax = plt.subplots(2, 1)
ax[0].plot(SIR_0, "--", color = "red")
ax[0].plot(SIR_95, "--", color = "red")
ax[0].plot(burden_estimate, color="green")
fig.delaxes(ax[1])
plt.show()

print(SIR_0)