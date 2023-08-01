import torch
import argparse
import numpy as np
from utils import set_seeds

from datetime import datetime
import os
# Exp
from VI_run import mRNA_Experiment_Writing, add_exp_args
from torch.optim import Adam, Adamax
import time
import matplotlib.pyplot as plt
# Data
from about_data.data import get_data, get_data_id, add_data_args

# Model
from model.model import get_model, get_model_id, add_model_args, get_loss


## Setup ##
parser = argparse.ArgumentParser()
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)

args = parser.parse_args()

## Specify data ##
truex, data_shape = get_data(args)
data_id = get_data_id(args)

## Specify model ##
loss_fn = get_loss(args)
model_id = get_model_id(args)

## Training ##
exp_writer = mRNA_Experiment_Writing(args, data_id, model_id)
real_batch = args.batch_size

assert args.dataset == "mRNA"
assert args.model_num == 1

for model_num in range(args.model_num):
    args.batch_size = real_batch
    torch.cuda.empty_cache()
    start = time.time()
    mycan, model = get_model(args, data_shape=data_shape)
    truex, data_shape = get_data(args, model_num=model_num)

    model = model.to(torch.float)
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.995 ** epoch)
    # groundtruth = truex

    try:
        for itr in range(args.iteration):
            # if itr> 100 :
            #    args.batch_size = 256
            loss, nll, _ = loss_fn(can_model=mycan, model=model, observation=truex, args=args, itr=itr)
            if itr != 0:
                loss.backward()
                max_norm = 4e-2
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            end = time.time()
            runtime = end - start
            exp_writer.loss_store(loss=loss, nll=nll, itr=itr, time=runtime, model_num=model_num)

            if itr % args.eval_every == 0:

                samples = loss_fn(can_model=mycan, model=model, observation=truex, args=args, eval=True, itr=itr)

                exp_writer.forward_img_store(samples, truex, model_num, itr)

                params_ls, _, _ = mycan(num_samples=5000, model=model)
                exp_writer.param_ls_img_store(params_ls, model_num=model_num, itr=itr)
                now = datetime.now()

                dt_string = now.strftime("%d.%m.%Y %H.%M.%S")
                torch.save({
                    'model_num': model_num,
                    'itr': itr,
                    # 'global_step': args.step,
                    'state_dict1': model.state_dict()},
                    os.path.join("./results/mRNA/model",
                                 '{}_checkpoint_date{}_itr{}_modelnum{}.pt'.format(args.bound_surjection,
                                                                                               dt_string, itr,
                                                                                               model_num)))
                print('')

    except NotImplementedError as e:
    #except Exception as e:
        print('')
        print("Error_{} occured".format(e))
        continue