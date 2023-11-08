
import conf
from .dnn import DNN
from torch.utils.data import DataLoader

from utils import memory

from utils.loss_functions import *
from sklearn.metrics import f1_score

from models.norm_layer import BatchNormWithMemory

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
from utils.iabn import *

class TENT_PBRS2(DNN):

    def __init__(self, *args, **kwargs):
        super(TENT_PBRS2, self).__init__(*args, **kwargs)

        # turn on grad for BN params only

        for param in self.net.parameters():  # initially turn off requires_grad for all
                param.requires_grad = False
        for module in self.net.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = conf.args.bn_momentum
                else:
                    # With below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False

                    module.src_running_mean = module.running_mean
                    module.src_running_var = module.running_var

                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        self.fifo = memory.FIFO(capacity=conf.args.update_every_x) # required for evaluation


    def train_online(self, current_num_sample):
        """
        Train the model
        """

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if not hasattr(self, 'previous_train_loss'):
            self.previous_train_loss = 0

        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED
        
        # Add a sample
        feats, cls, dls = self.target_train_set
        current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1], dls[current_num_sample - 1]
        self.fifo.add_instance(current_sample) #for batch-based inferece

        if conf.args.use_learned_stats: #batch-free inference
            # self.evaluation_online(current_num_sample, '', [[current_sample[0]], [current_sample[1]], [current_sample[2]]])
            self._train_online(current_num_sample, '', current_sample, self.mem.get_memory())

        if current_num_sample % conf.args.update_every_x != 0:  # train only when enough samples are collected
            if not (current_num_sample == len(self.target_train_set[
                                                  0]) and conf.args.update_every_x >= current_num_sample):  # update with entire data

                # self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=self.previous_train_loss)
                return SKIPPED

        if not conf.args.use_learned_stats: #batch-based inference
            # self.evaluation_online(current_num_sample, '', self.fifo.get_memory())
            output_logits = self._train_online(current_num_sample, '', self.fifo.get_memory(), self.mem.get_memory())

        # current test samples
        feats, cls, dls = self.fifo.get_memory()
        num_samples = len(feats)
        
        # not computed output_logits
        if conf.args.use_learned_stats:
            output_logits = self.net(feats)
        pseudo_cls = output_logits.max(1, keepdim=False)[1].to('cpu')

        for sample_idx in range(num_samples):
            if conf.args.memory_type in ['FIFO', 'Reservioir']:
                sample = feats[sample_idx], cls[sample_idx], dls[sample_idx]
                self.mem.add_instance(sample)

            elif conf.args.memory_type in ['PBRS']:
                sample = [feats[sample_idx], pseudo_cls[sample_idx], dls[sample_idx], cls[sample_idx], 0]
                self.mem.add_instance(sample)

        """
        # setup models
        self.net.train()

        if len(feats) == 1:  # avoid BN error
            self.net.eval()

        feats, _, _ = self.mem.get_memory()
        feats = torch.stack(feats)
        dataset = torch.utils.data.TensorDataset(feats)
        # data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
        data_loader = DataLoader(dataset, batch_size=200,
                                 shuffle=True, drop_last=False, pin_memory=False)

        entropy_loss = HLoss(temp_factor=conf.args.temperature)

        for e in range(conf.args.epoch):

            for batch_idx, (feats,) in enumerate(data_loader):
                feats = feats.to(device)
                preds_of_data = self.net(feats) # update bn stats

                if conf.args.no_optim:
                    pass # no optimization
                else:
                    loss = entropy_loss(preds_of_data)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)
        """

        return TRAINED

    def _train_online(self, epoch, condition, current_samples, memory_samples):
        entropy_loss = HLoss()

        features1, cl_labels1, do_labels1 = current_samples
        features2, cl_labels2, do_labels2 = memory_samples

        current_sample_num = len(features1)

        feats, cls, dls = torch.stack(features1 + features2), torch.stack(cl_labels1 + cl_labels2), torch.stack(do_labels1 + do_labels2)
        feats, cls, dls = feats.to(device), cls.to(device), dls.to(device)

        preds_of_data = self.net(feats)
        loss = entropy_loss(preds_of_data)

        if conf.args.no_optim:
            pass
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Only count current_samples
        preds_of_data = preds_of_data[:current_sample_num]

        y_pred = preds_of_data.max(1, keepdim=False)[1]

        try:
            true_cls_list = self.json['gt']
            pred_cls_list = self.json['pred']
            accuracy_list = self.json['accuracy']
            f1_macro_list = self.json['f1_macro']
            distance_l2_list = self.json['distance_l2']
        except KeyError:
            true_cls_list = []
            pred_cls_list = []
            accuracy_list = []
            f1_macro_list = []
            distance_l2_list = []

        # append values to lists
        true_cls_list += [int(c) for c in cl_labels1]
        pred_cls_list += [int(c) for c in y_pred.tolist()]
        cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred) / float(
            len(true_cls_list)) * 100
        accuracy_list.append(cumul_accuracy)
        f1_macro_list.append(f1_score(true_cls_list, pred_cls_list,
                                      average='macro'))

        self.occurred_class = [0 for i in range(conf.args.opt['num_class'])]

        # epoch: 1~len(self.target_train_set[0])
        progress_checkpoint = [int(i * (len(self.target_train_set[0]) / 100.0)) for i in range(1, 101)]
        for i in range(epoch + 1 - len(current_samples[0]), epoch + 1):  # consider a batch input
            if i in progress_checkpoint:
                print(
                    f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')

        # update self.json file
        self.json = {
            'gt': true_cls_list,
            'pred': pred_cls_list,
            'accuracy': accuracy_list,
            'f1_macro': f1_macro_list,
            'distance_l2': distance_l2_list,
        }

        return preds_of_data
