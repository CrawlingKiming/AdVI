import os
import pickle
import torch
from utils.exp_utils import get_args_table

class BaseExperiment(object):

    def __init__(self, model, optimizer, scheduler_iter, scheduler_epoch,
                 log_path, eval_every, check_every):

        # Objects
        self.model = model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch

        # Paths
        self.log_path = log_path
        self.check_path = os.path.join(log_path, 'check')

        # Intervals
        self.eval_every = eval_every
        self.check_every = check_every

    def train_fn(self, epoch):
        raise NotImplementedError()

    def eval_fn(self, epoch):
        raise NotImplementedError()

    def log_fn(self, epoch, train_dict, eval_dict):
        raise NotImplementedError()


    def create_folders(self):

        # Create log folder
        os.makedirs(self.log_path)
        print("Storing logs in:", self.log_path)

        # Create check folder
        if self.check_every is not None:
            os.makedirs(self.check_path)
            print("Storing checkpoints in:", self.check_path)

    def save_args(self, args):

        # Save args
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(os.path.join(self.log_path,'args_table.txt'), "w") as f:
            f.write(str(args_table))

    def save_metrics(self):

        # Save metrics
        with open(os.path.join(self.log_path,'metrics_train.pickle'), 'wb') as f:
            pickle.dump(self.train_metrics, f)
        with open(os.path.join(self.log_path,'metrics_eval.pickle'), 'wb') as f:
            pickle.dump(self.eval_metrics, f)


    def checkpoint_save(self):
        checkpoint = {'current_epoch': self.current_epoch,
                      'train_metrics': self.train_metrics,
                      'eval_metrics': self.eval_metrics,
                      'eval_epochs': self.eval_epochs,
                      'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                      'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None}
        torch.save(checkpoint, os.path.join(self.check_path, 'checkpoint.pt'))

    def checkpoint_load(self, check_path):
        checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pt'))
        self.current_epoch = checkpoint['current_epoch']
        self.train_metrics = checkpoint['train_metrics']
        self.eval_metrics = checkpoint['eval_metrics']
        self.eval_epochs = checkpoint['eval_epochs']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler_iter: self.scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if self.scheduler_epoch: self.scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])

    def run(self, epochs):
        raise NotImplementedError()