import numpy as np
import torch

dataset_choices = {'MSIR_full', "MSIR", 'SIR', 'SEIR', 'mRNA', 'WRF'}


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
        ds = 6
    
    if (data_name == "WRF"):
        ds = 10 #9

    return ds


def get_data(args, model_num=0):
    assert args.dataset in dataset_choices

    # Dataset
    data_shape = get_data_shape(args.dataset)

    if (args.dataset == 'MSIR') or (args.dataset =="MSIR_full"):
        #print("M")
        print("Current MSIR Data : {}".format(model_num), end='\n')
        #test_data = np.load("../data/MSIR_{}.npy".format(model_num))
        #test_data = np.genfromtxt("../data/TestCSV/MSIR6_{}.csv".format(model_num+1), delimiter=',')
        test_data = np.load("../data/MSIR_{}.npy".format(model_num))
        test_data = torch.tensor(test_data, dtype=torch.float, device=args.device)
        #test_data = test_data[None, :, :]
       
        # Get Test observed value, Y and Plot
        #truey = torch.tensor(test_data, dtype=torch.float, device=args.device)
        truex = test_data[[0], :, :]

    elif (args.dataset == 'SIR'):
        #print("S")
        print("Current SIR Data : {}".format(model_num), end='\n')
        obs_I = torch.tensor([[1,1,3,7,6,10,13,13,14,14,17,10,6,6,4,3,1,1,1,1,0]], dtype=torch.float, device=args.device)
        obs_R = torch.tensor([[0,0, 0, 0, 5, 7, 8, 13, 13, 16, 16, 24, 30, 31,33, 34, 36, 36, 36,36, 37]], dtype=torch.float, device=args.device)
        obs_I = obs_I[:,:, None]
        obs_R = obs_R[:,:, None]
        #test_data = np.load("../data/SIR/SIR_{}.npy".format(model_num))
        #test_data = torch.tensor(test_data, dtype=torch.float)

        truex = torch.cat((obs_I, obs_R), 2)
        #print(truex.shape)
        #raise ValueError

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
    
    if args.dataset == "WRF":
        obs0 = np.genfromtxt("../data/WRFdata/obs0.csv", delimiter=',')
        obs0 = obs0[1:,1]
        cov_theta_est = np.genfromtxt("../data/WRFdata/cov_theta_est.csv", delimiter=',')
        cov_theta_est = cov_theta_est[1:,1:]
        mat_Y_em = np.genfromtxt("../data/WRFdata/mat_Y_em.csv", delimiter=',')
        mat_Y_em = mat_Y_em[1:,1:]
        mat_Y_mean = np.genfromtxt("../data/WRFdata/mat_Y_mean.csv", delimiter=',')
        mat_Y_mean = mat_Y_mean[1:,1:]
        mle_est = np.genfromtxt("../data/WRFdata/mle_est.csv", delimiter=',')
        mle_est = mle_est[1:,1:]
        param_par_rescale = np.genfromtxt("../data/WRFdata/param_par_rescale.csv", delimiter=',')
        param_par_rescale =param_par_rescale[1:,1:]
        param_par = np.genfromtxt("../data/WRFdata/param_par.csv", delimiter=',')
        
        obs0 = torch.tensor(obs0, dtype=torch.float, device=args.device)
        cov_theta_est= torch.tensor(cov_theta_est, dtype=torch.float, device=args.device)
        mat_Y_em= torch.tensor(mat_Y_em, dtype=torch.float, device=args.device)
        mat_Y_mean= torch.tensor(mat_Y_mean, dtype=torch.float, device=args.device)
        mle_est = torch.tensor(mle_est , dtype=torch.float, device=args.device)
        param_par_rescale = torch.tensor(param_par_rescale , dtype=torch.float, device=args.device)
        param_par = torch.tensor(param_par , dtype=torch.float, device=args.device)
        truex = (obs0, cov_theta_est, mat_Y_em, mat_Y_mean, mle_est, param_par_rescale, param_par)

    return truex, data_shape


if __name__ == "__main__":
    test_data = np.genfromtxt("../data/WRFdata/cov_theta_est.csv") #np.genfromtxt("/data/Measles_data_time.csv", delimiter=',')
    print(test_data.shape)