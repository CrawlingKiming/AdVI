from .MSIR_full_AIS import build_MSIR_model, get_MSIR_loss,get_Hessian, get_MSIR_HMC_loss
#from .SEIR_AIS import build_SEIR_model, get_SEIR_loss
from .SIR_AIS import build_SIR_model, get_SIR_loss
#from .mRNA_full_AIS import build_mRNA_model, get_mRNA_loss
from .WRF_full_AIS import build_WRF_model, get_WRF_loss
def add_model_args(parser):

    # Flow params
    parser.add_argument('--num_flows', type=int, default=20)

    # Bound constriants
    parser.add_argument('--bound_surjection', type=eval, default=True)
    parser.add_argument('--AIS', type=eval, default=True)
    parser.add_argument('--temp', type=float, default=6)

    # Baseline Models
    parser.add_argument('--gaussian', type=eval, default=False)

    # Train params
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-3)

def get_model_id(args):
    if args.gaussian:
        mid = 'Gaussion_Coupla_{}'.format(args.gaussian)
    else:
        mid = 'Bound_surjection_{}_AIS_{}'.format(args.bound_surjection, args.AIS)
    return mid 

def get_model(args, data_shape):

    if (args.dataset == "MSIR_full") or (args.dataset == "MSIR"):
        #print("MSIR model")
        mycan, model = build_MSIR_model(args, data_shape)

    if (args.dataset == "SIR"):
        #print("SIR model")
        mycan, model = build_SIR_model(args, data_shape)

    #if (args.dataset == "SEIR"):
    #    mycan, model = build_SEIR_model(args, data_shape)

    #if (args.dataset == "mRNA"):
    #    mycan, model = build_mRNA_model(args, data_shape)

    if (args.dataset == "WRF"):
        mycan, model = build_WRF_model(args, data_shape)

    return mycan, model

def get_loss(args):

    if (args.dataset == "MSIR_full") or (args.dataset == "MSIR"):
        loss_fn = get_MSIR_loss#, get_MSIR_HMC_loss

    if args.dataset == "SIR":
        loss_fn = get_SIR_loss

    #if args.dataset == "SEIR":
    #    loss_fn = get_SEIR_loss

    #if args.dataset == "mRNA":
    #    loss_fn = get_mRNA_loss

    if args.dataset == "WRF":
        loss_fn = get_WRF_loss
    return loss_fn