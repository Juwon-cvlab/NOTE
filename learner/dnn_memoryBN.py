import conf
from .dnn import DNN
from torch.utils.data import DataLoader

from utils.loss_functions import *

from models.norm_layer import adapt_memory_bn, WeightPredictionModule

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

class DNNmemoryBN(DNN):
    def __init__(self, memory_size, batch_renorm, add_eps_numer, pred_module_type, *args, **kwargs):
        super(DNNmemoryBN, self).__init__(*args, **kwargs)

        self.net = adapt_memory_bn(model=self.net,
                                memory_size=memory_size,
                                use_prior=None,
                                batch_renorm=batch_renorm,
                                add_eps_numer=add_eps_numer,
                                use_dynamic_weight=False, 
                                use_binary_select=False, 
                                std_threshold=None,
                                pred_module_type=pred_module_type
                                )

        for param in self.net.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                # TENT: force use of batch stats in train and eval modes: https://github.com/DequanWang/tent/blob/master/tent.py
                module.track_running_stats = False

                module.src_running_mean = module.running_mean
                module.src_running_var = module.running_var

                module.running_mean = None
                module.running_var = None
            elif isinstance(module, WeightPredictionModule):
                for param in module.parameters():
                    param.requires_grad = True
        
        # re-init criterions, optimizers, scheduler
        if conf.args.method == 'Src' or conf.args.method == 'Tgt_memoryBN':
            if conf.args.dataset in ['cifar10', 'cifar100', 'harth', 'reallifehar', 'extrasensory']:
                self.optimizer = torch.optim.SGD(
                                  self.net.parameters(),
                                  conf.args.opt['learning_rate'],
                                  momentum=conf.args.opt['momentum'],
                                  weight_decay=conf.args.opt['weight_decay'],
                                  nesterov=True)

                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=conf.args.epoch * len(self.source_dataloader['train']))
            elif conf.args.dataset in ['tinyimagenet']:
                    self.optimizer = torch.optim.SGD(
                        self.net.parameters(),
                        conf.args.opt['learning_rate'],
                        momentum=conf.args.opt['momentum'],
                        weight_decay=conf.args.opt['weight_decay'],
                        nesterov=True)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                            weight_decay=conf.args.opt['weight_decay'])
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=conf.args.opt['learning_rate'],
                                    weight_decay=conf.args.opt['weight_decay'])


