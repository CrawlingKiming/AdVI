import os
from os import walk

def read_file(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
    dir_path = os.path.join(dir_path, 'results')
    #print(dir_path)
    if args.resume:
        if args.dataset == "MSIR":
            rel_path = 'MSIR' + '/check_model'
            log_path = os.path.join(dir_path, rel_path)
            filenames = next(walk(log_path), (None, None, []))[2]
        if args.dataset == "mRNA":
            rel_path = 'mRNA' + '/check_model'
            log_path = os.path.join(dir_path, rel_path)
            filenames = next(walk(log_path), (None, None, []))[2]
    else:

        rel_path = args.dataset + '\eval_model'#rel_path,
        log_path = os.path.join(dir_path, args.dataset,"eval_model")
        log_path = os.path.join(log_path, args.setting)

        filenames = next(walk(log_path), (None, None, []))[2]
        filenames.sort()
    #print(log_path, filenames)
    return log_path, filenames

def read_boot(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
    dir_path = os.path.join(dir_path, 'results')

    if args.resume:
        rel_path = os.path.join('MSIR', 'boot_model')
        log_path = os.path.join(dir_path, rel_path)
        filenames = next(walk(log_path), (None, None, []))[2]
    else:
        raise NotImplementedError

    return log_path, filenames

def read_samples(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
    dir_path = os.path.join(dir_path, 'results')

    rel_path = 'MSIR' + '/eval_model/MCMC'
    log_path = os.path.join(dir_path, rel_path)
    filenames = next(walk(log_path), (None, None, []))[2]

    return log_path, filenames
