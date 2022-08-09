import numpy as np
import pandas as pd

#data = np.load("MSIR.npy")

for i in range(100):
    data = np.load("SIR_{}.npy".format(i))

    print(data.shape)
    df = pd.DataFrame(data)

    filepath = './TestCSV/SIR_{}.csv'.format(i+1)
    #filepath = '.T/MSIR_test1.xlsx'
    # filepath = 'MSIR_46.csv'
    pd.DataFrame(data).to_csv(filepath, header=False, index=False)

