import numpy as np
import torch

dataset_choices = {'MSIR_full', "MSIR", 'SIR', 'SEIR', 'mRNA'}


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='MSIR_full', choices=dataset_choices)

def get_data_id(args):
    return '{}_'.format(args.dataset)

def get_data_shape(data_name):

    if (data_name == "SIR"):
        ds = 3

    if (data_name == "MSIR_full") or (data_name == "MSIR"):
        ds = 10

    if (data_name == "SEIR"):
        ds = 5

    if (data_name == "mRNA"):
        ds = 11

    return ds


def get_data(args, model_num=0):
    assert args.dataset in dataset_choices

    # Dataset
    data_shape = get_data_shape(args.dataset)

    if (args.dataset == 'MSIR') or (args.dataset =="MSIR_full"):
        #print("M")
        print("Current MSIR Data : {}".format(model_num), end='\n')
        test_data = np.load("../data/MSIR_{}.npy".format(model_num))
        test_data = torch.tensor(test_data, dtype=torch.float)

        # Get Test observed value, Y and Plot
        truey = torch.tensor(test_data, dtype=torch.float, device=args.device)
        truex = truey[[0], :, :]

    elif (args.dataset == 'SIR'):
        #print("S")
        print("Current SIR Data : {}".format(model_num), end='\n')
        test_data = np.load("../data/SIR/SIR_{}.npy".format(model_num))
        test_data = torch.tensor(test_data, dtype=torch.float)

        truex = test_data

    if args.dataset == 'SEIR':
        test_data = np.genfromtxt("../data/Measles_data_time.csv", delimiter=',')
        test_data = torch.tensor(test_data[1:], dtype=torch.float)

        # Get Test observed value, Y and Plot
        truex = torch.tensor(test_data, dtype=torch.float, device=args.device)

    if args.dataset == 'mRNA':
        test_data = np.genfromtxt("../data/mRNA/mRNA_data.csv")
        test_data = torch.tensor(data=test_data[[0], :], dtype=torch.float32)

        sk = np.load("../data/mRNA/mRNA_sk.npy")
        sk = torch.tensor(data=sk, dtype=torch.float32)
        truex = (test_data, sk)

    return truex, data_shape


if __name__ == "__main__":
    test_data = np.genfromtxt("/data/Measles_data_time.csv", delimiter=',')
    print(test_data.shape)