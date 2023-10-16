# -*- coding: utf-8 -*-
import sys
import argparse
import random
import numpy as np
import torch
import time

import conf

import os

import wandb


def get_path():
    path = 'my_log/'

    # information about used data type
    path += conf.args.dataset + '/'

    # information about used model type
    path += conf.args.method + '/'

    """
    # information about domain(condition) of training data
    if conf.args.src == ['rest']:
        path += 'src_rest' + '/'
    elif conf.args.src == ['all']:
        path += 'src_all' + '/'
    elif conf.args.src is not None and len(conf.args.src) >= 1:
        path += 'src_' + '_'.join(conf.args.src) + '/'

    # if conf.args.tgt:
    #     path += 'tgt_' + conf.args.tgt + '/'
    path += 'tgt_' + 'multi-domain' + '/'
    """
    path += 'src'
    if conf.args.tgts:
        for tgt in conf.args.tgts:
            path += '_' + tgt.split('-') [0]
    path += '/'
    path += 'tgt_all/'

    path += conf.args.log_prefix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path
    result_path = path + '/'

    print('Path:{}'.format(path))
    return result_path, checkpoint_path, log_path


def main():
    ######################################################################
    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    ################### Hyper parameters #################

    if 'cifar100' in conf.args.dataset:
        opt = conf.CIFAR100Opt
    elif 'cifar10' in conf.args.dataset:
        opt = conf.CIFAR10Opt

    conf.args.opt = opt
    if conf.args.lr:
        opt['learning_rate'] = conf.args.lr
    if conf.args.weight_decay:
        opt['weight_decay'] = conf.args.weight_decay
    if conf.args.batch_size:
        opt['batch_size'] = conf.args.batch_size

    model = None


    if conf.args.model == "resnet18":
        from models import ResNet
        model = ResNet.ResNet18()
    elif 'wideresnet' in conf.args.model:
        from models.define_model import define_model
        model = define_model(conf.args)

    # import modules after setting the seed
    from data_loader import data_loader as data_loader
    from learner.dnn import DNN
    from learner.dnn_memoryBN import DNNmemoryBN
    
    result_path, checkpoint_path, log_path = get_path()

    ########## Dataset loading ############################

    if conf.args.method == 'Src':
        learner_method = DNN
    elif conf.args.method == 'Tgt_memoryBN':
        learner_method = DNNmemoryBN
    else:
        raise NotImplementedError

    print('##############Target Data Loading...##############')
    target_data_loader = data_loader.multi_domain_data_loader(conf.args.dataset, conf.args.tgts,
                                                        conf.args.opt['file_path'],
                                                        batch_size=conf.args.opt['batch_size'],
                                                        valid_split=0,
                                                        test_split=0, is_src=False,
                                                        num_source=conf.args.num_source)
    # learner = learner_method(model, source_dataloader=target_data_loader, target_dataloader=None, write_path=log_path)
    if conf.args.method == 'Src':
        learner = learner_method(model, source_dataloader=target_data_loader,
                             target_dataloader=None, write_path=log_path,
                             )
    else:
        learner = learner_method(conf.args.bn_memory_size,
                                conf.args.batch_renorm,
                                conf.args.add_eps_numer,
                                conf.args.pred_module_type, 
                                model, source_dataloader=target_data_loader,
                                target_dataloader=None, write_path=log_path,
                                )
        
    #################### Wandb init #######################
    if conf.args.wandb:
        wandb.init(
            project='RobustTTA',
            name=conf.args.wandb_name,
            config={
                "learning_rate": conf.args.lr,
                "epochs": conf.args.epoch
            }
        )

    #################### Training #########################

    since = time.time()


    # make dir if doesn't exist
    if not os.path.exists(result_path):
        oldumask = os.umask(0)
        os.makedirs(result_path, 0o777)
        os.umask(oldumask)
    if not os.path.exists(checkpoint_path):
        oldumask = os.umask(0)
        os.makedirs(checkpoint_path, 0o777)
        os.umask(oldumask)
    script = ' '.join(sys.argv[1:])

    if conf.args.online == False:

        start_epoch = 1
        best_acc = -9999
        best_epoch = -1

        for epoch in range(start_epoch, conf.args.epoch + 1):
            learner.train_multi(epoch, domain_step=conf.args.domain_step, num_domain=len(conf.args.tgts))


        learner.save_checkpoint(epoch=0, epoch_acc=-1, best_acc=best_acc,
                                checkpoint_path=checkpoint_path + 'cp_last.pth.tar')
        learner.dump_eval_online_result(is_train_offline=True) # eval with final model

        target_acc = learner.json['target_accuracy']

        time_elapsed = time.time() - since
        print('Completion time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))
        print('Target val Acc: {:4f}'.format(target_acc))
    else:
        raise NotImplemented

    if conf.args.remove_cp:
        best_path = checkpoint_path + 'cp_best.pth.tar'
        last_path = checkpoint_path + 'cp_last.pth.tar'
        try:
            os.remove(best_path)
            os.remove(last_path)
        except Exception as e:
            pass
            # print(e)
            
    if conf.args.wandb:
        wandb.finish()


def parse_arguments(argv):
    """Command line parse."""

    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    ###MANDATORY###

    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset to be used, in [ichar, icsr, dsa, hhar, opportunity, gait, pamap2]')

    parser.add_argument('--model', type=str, default='HHAR_model',
                        help='Which model to use')

    parser.add_argument('--method', type=str, default='',
                        help='specify the method name')

    parser.add_argument('--src', nargs='*', default=None,
                        help='Specify source domains; not passing an arg load default src domains specified in conf.py')
    parser.add_argument('--tgts', nargs='+', type=str, default=None,
                        help='specific target domain; give "src" if you test under src domain')
    parser.add_argument('--gpu_idx', type=int, default=0, help='which gpu to use')

    ###Optional###
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate to overwrite conf.py')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='weight_decay to overwrite conf.py')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size to overwrite conf.py')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--epoch', type=int, default=1,
                        help='How many epochs do you want to use for train')
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Load checkpoint and train from checkpoint in path?')
    parser.add_argument('--train_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for train')
    parser.add_argument('--valid_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for valid')
    parser.add_argument('--test_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for test')
    parser.add_argument('--nsample', type=int, default=99999,
                        help='How many samples do you want use for train')
    parser.add_argument('--log_prefix', type=str, default='',
                        help='Prefix of log file path')
    parser.add_argument('--remove_cp', action='store_true',
                        help='Remove checkpoints after evaluation')
    parser.add_argument('--data_gen', action='store_true',
                        help='generate training data with source for training estimator')

    parser.add_argument('--num_source', type=int, default=100,
                        help='number of available sources')

    #### Distribution ####
    parser.add_argument('--tgt_train_dist', type=int, default=0,
                        help='0: real selection'
                             '1: random selection'
                             '4: dirichlet distribution'
                        )
    parser.add_argument('--dirichlet_beta', type=float, default=0.1,
                        help='the concentration parameter of the Dirichlet distribution for heterogeneous partition.')

    parser.add_argument('--online', action='store_true', help='training via online learning?')
    parser.add_argument('--update_every_x', type=int, default=1, help='number of target samples used for every update')
    parser.add_argument('--memory_size', type=int, default=1,
                        help='number of previously trained data to be used for training')
    parser.add_argument('--memory_type', type=str, default='FIFO', help='FIFO, PBRS')

    parser.add_argument('--validation', action='store_true', help='Use validation data instead of test data for hyperparameter tuning')
    parser.add_argument('--iabn', action='store_true', help='replace bn with iabn layer')

    #MemoryBN
    parser.add_argument('--bn_memory_size', default=128, type=int)
    parser.add_argument('--prior', default=None, type=float)
    parser.add_argument('--batch_renorm', action='store_true')
    parser.add_argument('--add_eps_numer', action='store_true')

    parser.add_argument('--dynamic-weight', action='store_true')

    parser.add_argument('--binary-select', action='store_true')
    parser.add_argument('--std-threshold', type=float, default=1.0)

    parser.add_argument('--pred_module_type', type=int, default=0)

    parser.add_argument('--aug_type', type=int, default=0)

    parser.add_argument('--dummy', action='store_true', default=False, help='do nothing')

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_name', type=str, default='temp')

    parser.add_argument('--domain_step', type=int, default=1)

    parser.add_argument('--use_src_stat', action='store_true')
    parser.add_argument('--use_in_stat', action='store_true')
    parser.add_argument('--channel_wise', action='store_true')

    return parser.parse_args()


def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print('Command:', end='\t')
    print(" ".join(sys.argv))
    conf.args = parse_arguments(sys.argv[1:])
    set_seed()
    main()

