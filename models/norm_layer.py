import torch
import torch.nn as nn
from torch.nn import functional as F

import conf

def adapt_alpha_bn(model, alpha):
    return AlphaBN.adapt_model(model.net, alpha)
def adapt_ema_bn(model, use_prior, momentum, batch_renorm, add_eps_numer):
    return emaBN.adapt_model(model, use_prior,  momentum, batch_renorm, add_eps_numer)
def adapt_memory_bn(model, memory_size, use_prior, batch_renorm, add_eps_numer, use_dynamic_weight, use_binary_select, std_threshold, pred_module_type, push_last):
    return BatchNormWithMemory.adapt_model(model, memory_size, use_prior, batch_renorm, add_eps_numer,
                                           use_dynamic_weight, use_binary_select, std_threshold, pred_module_type, push_last)

def calculate_weighted_stat(mu1, var1, mu2, var2, weight, weight_var=None):
    mean = mu1 * weight + (1 - weight) * mu2

    if conf.args.add_correction_term2:
        variance = weight * var1 + (1 - weight) * var2 + weight * (1 - weight) * (mu1 - mu2) ** 2
    else:
        if weight_var is None:
            variance = weight * var1 + (1 - weight) * var2
        else:
            variance = weight_var * var1 + (1 - weight_var) * var2
    return mean, variance

class WeightPredictionModule(nn.Module):
    def __init__(self, out_channel, channel, reduction_ratio=4, combined=False, channel_wise=False):
        super().__init__()

        self.reduction_ratio = reduction_ratio
        self.combined = combined

        if combined:
            self.fc1 = nn.Sequential(
                nn.Linear(channel, channel // reduction_ratio, bias=False),
                nn.ReLU(inplace=True)
            )
            self.fc2_1 = nn.Sequential(
                nn.Linear(channel // reduction_ratio, out_channel if channel_wise else 1, bias=False),
                nn.Sigmoid()
            )
            self.fc2_2 = nn.Sequential(
                nn.Linear(channel // reduction_ratio, out_channel if channel_wise else 1, bias=False),
                nn.Sigmoid()
            )

        else:
            self.fc1 = nn.Sequential(
                nn.Linear(channel, channel // reduction_ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction_ratio, out_channel if channel_wise else 1, bias=False),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        y = self.fc1(x)

        if self.combined:
            y1 = self.fc2_1(y)
            y2 = self.fc2_2(y)

            return y1, y2
        else:
            return y
        

class WeightPredictionModule2(nn.Module):
    def __init__(self, out_channel, channel, reduction_ratio=4, channel_wise=False):
        super().__init__()

        self.reduction_ratio = reduction_ratio
        self.channel_wise = channel_wise

        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc2_1 = nn.Sequential(
            nn.Linear(channel // reduction_ratio, out_channel * 3 if channel_wise else 3, bias=True),
        )
        self.fc2_2 = nn.Sequential(
            nn.Linear(channel // reduction_ratio, out_channel * 3 if channel_wise else 3, bias=True),
        )

        self.softmax = nn.Softmax(dim=0)

        
    def forward(self, x):
        y = self.fc1(x)

        y1 = self.fc2_1(y)
        y2 = self.fc2_2(y)

        if self.channel_wise:
            y1 = y1.view(3, -1)
            y2 = y2.view(3, -1)

        y1 = self.softmax(y1)
        y2 = self.softmax(y2)

        y1 = y1.view(-1)
        y2 = y2.view(-1)

        out_channel = y1.shape[0]

        alpha_mu, beta_mu, gamma_mu = y1[:out_channel//3], y1[out_channel//3:out_channel//3 * 2], y1[out_channel//3 * 2:]
        alpha_var, beta_var, gamma_var = y2[:out_channel//3], y2[out_channel//3:out_channel//3 * 2], y2[out_channel//3 * 2:]

        return alpha_mu, beta_mu, gamma_mu, alpha_var, beta_var, gamma_var

class AlphaBN(nn.Module):
    @ staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is not None:
            for name, child in parent.named_children():
                #FIXME: this may cause error if it will be combined with Tent or optimization method
                # child.requires_grad_(False)

                if isinstance(child, nn.BatchNorm2d):
                    module = AlphaBN(child, alpha)
                    replace_mods.append((parent, name, module))
                else:
                    replace_mods.extend(AlphaBN.find_bns(child, alpha))
        return replace_mods
    
    @staticmethod
    def adapt_model(model, alpha):
        replace_mods = AlphaBN.find_bns(model, alpha)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model
    
    def __init__(self, layer, alpha):
        super().__init__()

        self.layer = layer
        self.alpha = alpha

        self.norm = nn.BatchNorm2d(self.layer.num_features, eps=self.layer.eps, affine=False, momentum=1.0, device=self.layer.weight.device)

    def forward(self, input):
        # compute in-batch statistics
        self.norm.train() # to update in-batch statistics in BN_train mode
        self.norm(input)

        mixed_mu  = self.alpha * self.layer.src_running_mean + (1 - self.alpha) * self.norm.running_mean
        mixed_var = self.alpha * self.layer.src_running_var  + (1 - self.alpha) * self.norm.running_var

        return F.batch_norm(
            input,
            mixed_mu,
            mixed_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps
        )


class emaBN(nn.Module):
    @ staticmethod
    def find_bns(parent, use_prior, momentum, batch_renorm, add_eps_numer):
        replace_mods = []
        if parent is not None:
            for name, child in parent.named_children():
                #FIXME: this may cause error if it will be combined with Tent or optimization method
                # child.requires_grad_(False)

                if isinstance(child, nn.BatchNorm2d):
                    module = emaBN(child, use_prior, momentum, batch_renorm, add_eps_numer)
                    replace_mods.append((parent, name, module))
                else:
                    replace_mods.extend(emaBN.find_bns(child, use_prior, momentum, batch_renorm, add_eps_numer))
        return replace_mods
    
    @staticmethod
    def adapt_model(model, use_prior, momentum, batch_renorm, add_eps_numer):
        replace_mods = emaBN.find_bns(model, use_prior, momentum, batch_renorm, add_eps_numer)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model
    
    def __init__(self, layer, use_prior, momentum, batch_renorm, add_eps_numer):
        super().__init__()

        self.layer = layer

        self.use_prior = use_prior
        self.momentum = momentum
        self.batch_renorm = batch_renorm
        self.add_eps_numer = add_eps_numer

        ema_mu = torch.randn(size=(self.layer.num_features,), device=self.layer.weight.device)
        ema_var = torch.randn(size=(self.layer.num_features,), device=self.layer.weight.device)

        self.register_buffer('ema_mu', ema_mu)
        self.register_buffer('ema_var', ema_var)

        self.first = True

    def forward(self, input):
        # TODO: use mean_and_var or batch_norm(trainaing=True, momentum=1.0)???
        batch_mu = input.mean([0, 2, 3])
        batch_var = input.var([0, 2, 3], unbiased=False)
        # batch_num = 1

        # save mu and variance in memory
        if self.first:
            self.ema_mu[:] = batch_mu.detach()
            self.ema_var[:] = batch_var.detach()

            self.first = False
        else:
            self.ema_mu[:] = self.ema_mu * self.momentum + batch_mu.detach() * (1 - self.momentum)
            self.ema_var[:] = self.ema_var * self.momentum + batch_mu.detach() * (1 - self.momentum)

        if self.use_prior:
            test_mean, test_var = calculate_weighted_stat(self.layer.src_running_mean, self.layer.src_running_var,
                                                          self.ema_mu, self.ema_var, 
                                                          self.use_prior)
        else:
            test_mean = self.ema_mu
            test_var = self.ema_var

        if self.batch_renorm:
            input = BatchRenorm(input, test_mean, test_var, self.layer.eps, self.add_eps_numer)
            input = self.layer.weight[None, :, None, None] * input + self.layer.bias[None, :, None, None]

            return input
        else:
            input = (input - test_mean[None, :, None, None]) / (torch.sqrt(test_var[None, :, None, None] + self.layer.eps))
            input = input * self.layer.weight[None, :, None, None] + self.layer.bias[None, :, None, None]
            return input

class BatchNormWithMemory(nn.Module):
    @staticmethod
    def find_bns(parent, memory_size, use_prior, batch_renorm=False, add_eps_numer=False,
                 use_dynamic_weight=False, use_binary_select=False, std_threshold=1.0, pred_module_type=0, push_last=False):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            # child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = BatchNormWithMemory(child, memory_size, use_prior, batch_renorm, add_eps_numer,
                                             use_dynamic_weight, use_binary_select, std_threshold, pred_module_type, push_last)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BatchNormWithMemory.find_bns(child, memory_size, use_prior, batch_renorm, add_eps_numer,
                                                                 use_dynamic_weight, use_binary_select, std_threshold, pred_module_type, push_last))
    
        return replace_mods

    @staticmethod
    def adapt_model(model, memory_size, use_prior=None, batch_renorm=False, add_eps_numer=False,
                    use_dynamic_weight=False, use_binary_select=False, std_threshold=1.0, pred_module_type=0, push_last=False):
        replace_mods = BatchNormWithMemory.find_bns(model, memory_size, use_prior, batch_renorm=batch_renorm, add_eps_numer=add_eps_numer,
                                                    use_dynamic_weight=use_dynamic_weight, use_binary_select=use_binary_select, std_threshold=std_threshold,
                                                    pred_module_type=pred_module_type, push_last=push_last)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)

        """
        for bn_idx, (_, _, child) in enumerate(replace_mods):
            subsequent_bn_layers = [bn for _, _, bn in replace_mods[bn_idx+1:]]
            setattr(child, 'remaining_bns', subsequent_bn_layers)
        """
        
        for _, (_, _, child) in enumerate(replace_mods):
            all_memorybn_layers = [bn for _, _, bn in replace_mods]
            setattr(child, 'remaining_bns', all_memorybn_layers)

        return model
    
    def reset_subsequent_bns(self):
        for memory_bn in self.remaining_bns:
            memory_bn.reset()

    def __init__(self, layer, memory_size, use_prior=None, batch_renorm=False, add_eps_numer=False,
                 use_dynamic_weight=False, use_binary_select=False, std_threshold=1.0, pred_module_type=0, push_last=False):
        super().__init__()

        self.layer = layer

        self.memory_size = memory_size
        self.use_prior = use_prior
        self.batch_renorm = batch_renorm

        batch_mu_memory = torch.randn(size=(memory_size, self.layer.num_features), device=self.layer.weight.device)
        batch_var_memory = torch.randn(size=(memory_size, self.layer.num_features), device=self.layer.weight.device)
        self.register_buffer('batch_mu_memory', batch_mu_memory)
        self.register_buffer('batch_var_memory', batch_var_memory)

        self.unbiased = False

        batch_pointer = torch.zeros(1, dtype=torch.int)
        self.register_buffer('batch_pointer', batch_pointer)

        batch_full = torch.zeros(1, dtype=torch.bool)
        self.register_buffer('batch_full', batch_full)

        self.add_eps_numer = add_eps_numer

        self.use_dynamic_weight = use_dynamic_weight
        self.push_last = push_last

        self.event_count1 = 0
        self.event_count2 = 0
        self.event_count3 = 0
        self.event_count4 = 0

        self.save_stat = True

        """
        self.use_binary_select = use_binary_select
        self.std_threshold = std_threshold
        self.pred_module_type = pred_module_type

        if pred_module_type == 1 or pred_module_type == 2:
            self.pred_module1 = WeightPredictionModule(self.layer.num_features, self.layer.num_features * (pred_module_type + 1), reduction_ratio=4)
            self.pred_module2 = WeightPredictionModule(self.layer.num_features, self.layer.num_features * (pred_module_type + 1), reduction_ratio=4)

            self.pred_module1.to(self.layer.weight.device)
            self.pred_module2.to(self.layer.weight.device)
        elif pred_module_type == 3 or pred_module_type == 4:
            self.pred_module = WeightPredictionModule(self.layer.num_features, self.layer.num_features * (pred_module_type - 1) * 2, reduction_ratio=4, combined=True)

            self.pred_module.to(self.layer.weight.device)
        
        if pred_module_type == 5 or pred_module_type == 6:
            self.pred_module1 = WeightPredictionModule(self.layer.num_features, self.layer.num_features * (pred_module_type - 3), reduction_ratio=4, channel_wise=True)
            self.pred_module2 = WeightPredictionModule(self.layer.num_features, self.layer.num_features * (pred_module_type - 3), reduction_ratio=4, channel_wise=True)

            self.pred_module1.to(self.layer.weight.device)
            self.pred_module2.to(self.layer.weight.device)
        elif pred_module_type == 7 or pred_module_type == 8:
            self.pred_module = WeightPredictionModule(self.layer.num_features, self.layer.num_features * (pred_module_type - 5) * 2, reduction_ratio=4, combined=True, channel_wise=True)

            self.pred_module.to(self.layer.weight.device)
        elif pred_module_type == 9:
            channel_mul = 1
            if conf.args.use_src_stat:
                channel_mul = channel_mul + 1
            if conf.args.use_in_stat:
                channel_mul = channel_mul + 1
            self.pred_module = WeightPredictionModule2(self.layer.num_features, self.layer.num_features * channel_mul * 2, reduction_ratio=4, channel_wise=conf.args.channel_wise)

            self.pred_module.to(self.layer.weight.device)
        """

    def set_save_stat(self, save_stat):
        self.save_stat = save_stat
        
    def reset(self):
        # self.pointer = 0
        # self.full = False

        self.batch_pointer[0] = 0
        self.batch_full[0] = False

    def get_stat_of_mem(self, std=False):
        mem_mean, mem_var = self.get_batch_mu_and_var()

        if std:
            std_of_mem_mean = mem_mean.std([0], unbiased=False)
            std_of_mem_var = mem_var.std([0], unbiased=False)
            
            return std_of_mem_mean, std_of_mem_var
        else:
            var_of_mem_mean = mem_mean.var([0], unbiased=False)
            var_of_mem_var = mem_var.var([0], unbiased=False)

            return var_of_mem_mean, var_of_mem_var
        
    def get_last_batch_mu_and_var(self):
        last_index = (self.batch_pointer.item() - 1) % self.memory_size

        last_mu = self.batch_mu_memory[last_index,:].view(1, -1)
        last_var = self.batch_var_memory[last_index,:].view(1, -1)

        return last_mu, last_var
    
    def get_mixed_mu_and_var(self, prior_value=None):
        
        # if self.use_instance:
        if False:
            if self.full:
                test_mean = self.mu_memory
                test_var = self.var_memory
            else:
                test_mean = self.mu_memory[:self.pointer, :]
                test_var = self.var_memory[:self.pointer, :]
        else:
            if self.batch_full:
                test_mean = self.batch_mu_memory
                test_var = self.batch_var_memory
            else:
                test_mean = self.batch_mu_memory[:self.batch_pointer.item(), :]
                test_var = self.batch_var_memory[:self.batch_pointer.item(), :]

        test_mean = torch.mean(test_mean, 0, keepdim=True)
        test_var = torch.mean(test_var, 0, keepdim=True)

        if prior_value is not None:
            prior = prior_value
        else:
            prior = self.use_prior

        if prior:
            test_mean = (
                prior * self.layer.src_running_mean
                + (1 - prior) * test_mean
            )
            test_var = (
                prior * self.layer.src_running_var
                + (1 - prior) * test_var
            )

        return test_mean, test_var

    def get_batch_mu_and_var(self):
        if self.batch_full:
            return self.batch_mu_memory, self.batch_var_memory
        return self.batch_mu_memory[:self.batch_pointer.item()], self.batch_var_memory[:self.batch_pointer.item()]

    def get_aggreated_statistics(self):
        mem_mean, mem_var = self.get_batch_mu_and_var()

        if conf.args.add_correction_term1:
            first = torch.mean(mem_var, 0)
            second = torch.mean(mem_mean * mem_mean, 0)
            third = mem_mean.mean(0) * mem_mean.mean(0)

            test_var = first + second - third
            test_mean = torch.mean(mem_mean, 0)
        else:
            test_mean = torch.mean(mem_mean, 0)
            test_var = torch.mean(mem_var, 0)

        return test_mean, test_var

    def forward(self, input):
        # self._check_input_dim(input)
        
        # TODO: use mean_and_var or batch_norm(trainaing=True, momentum=1.0)???
        batch_mu = input.mean([0, 2, 3])
        batch_var = input.var([0, 2, 3], unbiased=False)
        # batch_num = 1

        if not self.push_last and self.save_stat:
            # save mu and variance in memory
            batch_start = self.batch_pointer.item()
            batch_end = self.batch_pointer.item() + 1
            batch_idxs_replace = torch.arange(batch_start, batch_end).to(input.device) % self.memory_size

            self.batch_mu_memory[batch_idxs_replace, :] = batch_mu.detach()
            self.batch_var_memory[batch_idxs_replace, :] = batch_var.detach()

            self.batch_pointer[0] = batch_end % self.memory_size

            if batch_end >= self.memory_size:
                self.batch_full[0] = True

        # compute test mu and variance from in-memory elements
        if self.batch_full:
            test_mean = self.batch_mu_memory
            test_var = self.batch_var_memory
        elif self.batch_pointer == 0:
            test_mean = batch_mu.unsqueeze(0)
            test_var = batch_var.unsqueeze(0)
        else:
            test_mean = self.batch_mu_memory[:self.batch_pointer.item(), :]
            test_var = self.batch_var_memory[:self.batch_pointer.item(), :]

        # test_mean = torch.mean(test_mean, 0)
        # test_var = torch.mean(test_var, 0)
        if conf.args.add_correction_term1:
            first = torch.mean(test_var, 0)
            second = torch.mean(test_mean * test_mean, 0)
            third = test_mean.mean(0) * test_mean.mean(0)
            test_var = first + second - third

            test_mean = torch.mean(test_mean, 0)
        else:
            test_mean = torch.mean(test_mean, 0)
            test_var = torch.mean(test_var, 0)

        if self.batch_full or self.batch_pointer > 10:
            self.detect_domain_shift(batch_mu, batch_var, test_mean, test_var)

        if self.use_dynamic_weight:
            prior_mu, prior_var = self.dynamic_weight(batch_mu, batch_var, test_mean, test_var)
            test_mean, test_var = calculate_weighted_stat(self.layer.src_running_mean, self.layer.src_running_var,
                                                         test_mean, test_var, prior_mu, prior_var)
        else:
            ### just for calculating distance and prior
            # self.dynamic_weight(batch_mu, batch_var, test_mean, test_var)
            if self.use_prior:
                test_mean, test_var = calculate_weighted_stat(self.layer.src_running_mean, self.layer.src_running_var,
                                                              test_mean, test_var,
                                                              self.use_prior)
        
        if self.push_last and self.save_stat:
            # save mu and variance in memory
            batch_start = self.batch_pointer.item()
            batch_end = self.batch_pointer.item() + 1
            batch_idxs_replace = torch.arange(batch_start, batch_end).to(input.device) % self.memory_size

            self.batch_mu_memory[batch_idxs_replace, :] = batch_mu.detach()
            self.batch_var_memory[batch_idxs_replace, :] = batch_var.detach()

            self.batch_pointer[0] = batch_end % self.memory_size

            if batch_end >= self.memory_size:
                self.batch_full[0] = True
        
        if self.batch_renorm:
            input = BatchRenorm(input, test_mean, test_var, self.layer.eps, self.add_eps_numer)
            input = self.layer.weight[None, :, None, None] * input + self.layer.bias[None, :, None, None]

            return input
        else:
            input = (input - test_mean[None, :, None, None]) / (torch.sqrt(test_var[None, :, None, None] + self.layer.eps))
            input = input * self.layer.weight[None, :, None, None] + self.layer.bias[None, :, None, None]
            return input
    
    def detect_domain_shift(self, test_mu, test_var, global_mu, global_var):
        mem_mean, mem_var = self.get_batch_mu_and_var()

        dist_mean_m2avg = torch.cdist(mem_mean, global_mu.unsqueeze(0)).squeeze()
        dist_var_m2avg = torch.cdist(mem_var, global_var.unsqueeze(0)).squeeze()

        var_dist_mean_m2avg, mean_dist_mean_m2avg = torch.var_mean(dist_mean_m2avg)
        var_dist_var_m2avg, mean_dist_var_m2avg = torch.var_mean(dist_var_m2avg)

        dist_mean_b2avg = torch.cdist(test_mu.unsqueeze(0), global_mu.unsqueeze(0)).squeeze()
        dist_var_b2avg = torch.cdist(test_var.unsqueeze(0), global_var.unsqueeze(0)).squeeze()

        condition1 =  (mean_dist_mean_m2avg - dist_mean_b2avg) / torch.sqrt(var_dist_mean_m2avg) > conf.args.threshold
        condition2 =  (mean_dist_mean_m2avg - dist_mean_b2avg) / torch.sqrt(var_dist_mean_m2avg) < -1 * conf.args.threshold

        condition3 =  (mean_dist_var_m2avg - dist_var_b2avg) / torch.sqrt(var_dist_var_m2avg) > conf.args.threshold
        condition4 =  (mean_dist_var_m2avg - dist_var_b2avg) / torch.sqrt(var_dist_var_m2avg) < -1 * conf.args.threshold

        if condition1 or condition2 or condition3 or condition4:
            self.reset_subsequent_bns()

            if condition1:
                self.event_count1 = self.event_count1 + 1

            if condition2:
                self.event_count2 = self.event_count2 + 1

            if condition3:
                self.event_count3 = self.event_count3 + 1

            if condition4:
                self.event_count4 = self.event_count4 + 1

            print("domain_shift is detected ", self.event_count1, self.event_count2, self.event_count3, self.event_count4)

    def dynamic_weight(self, test_mu, test_var, mem_mean, mem_var):
        if self.push_last and not self.batch_full and self.batch_pointer == 0: # no item,
            return 0.5, 0.5
        
        # TODO: layer-wise interpolation
        test2src_mu = torch.cdist(test_mu.unsqueeze(0), self.layer.src_running_mean.unsqueeze(0)).squeeze()
        test2src_var = torch.cdist(test_var.unsqueeze(0), self.layer.src_running_var.unsqueeze(0)).squeeze()

        test2mem_mu = torch.cdist(test_mu.unsqueeze(0), mem_mean.unsqueeze(0)).squeeze()
        test2mem_var = torch.cdist(test_var.unsqueeze(0), mem_var.unsqueeze(0)).squeeze()

        # TODO: same weight or separated weight?
        """
        prior_mu = 1 - (test2src_mu / (test2src_mu + test2mem_mu))
        prior_var = 1 - (test2src_var / (test2src_var + test2mem_var))

        prior_mu = prior_mu.detach()
        prior_var = prior_var.detach()

        self.last_prior_mu = prior_mu
        self.last_prior_var = prior_var

        return prior_mu, prior_var
        """

        """
        test2src_total = test2src_mu + test2src_var
        test2mem_total = test2mem_mu + test2mem_var

        self.last_t2src_total = test2src_total.detach()
        self.last_t2src_mu = test2src_mu.detach()
        self.last_t2src_var = test2src_var.detach()

        self.last_t2mem_total = test2mem_total.detach()
        self.last_t2mem_mu = test2mem_mu.detach()
        self.last_t2mem_var = test2mem_var.detach()

        prior = 1 - test2src_total / (test2src_total + test2mem_total)
        prior = prior.detach()

        if conf.args.gamma != 1.0:
            prior = prior ** conf.args.gamma

        prior_mu = 1 - test2src_mu / (test2src_mu + test2mem_mu)
        prior_mu = prior_mu.detach()

        prior_var = 1 - test2src_var / (test2src_var + test2mem_var)
        prior_var = prior_var.detach()

        self.last_prior = prior
        self.last_prior_mu = prior_mu
        self.last_prior_var = prior_var
        """

        test2src = test2src_mu + test2src_var
        test2mem = test2mem_mu + test2mem_var

        prior = 1 - test2src / (test2src + test2mem)
        if conf.args.gamma != 1.0:
            prior = prior ** conf.args.gamma
        prior = prior.detach()

        self.last_prior_mu = prior
        self.last_prior_var = prior
        
        if not self.push_last and not self.batch_full and self.batch_pointer == 1: # just only one item,
            return 0.5, 0.5

        return prior, prior

    def binary_selection(self, a_mu, a_var, b_mu, b_var, std_threshold=1.0):
        std_of_mem_mu, std_of_mem_var = self.get_stat_of_mem(std=True)

        mem_mean, mem_var = self.get_batch_mu_and_var()
        mem_mean = torch.mean(mem_mean, 0)
        mem_var = torch.mean(mem_var, 0)

        out_upper_mu = mem_mean + std_of_mem_mu * std_threshold
        out_upper_var = mem_var + std_of_mem_var * std_threshold

        out_lower_mu = mem_mean - std_of_mem_mu * std_threshold
        out_lower_var = mem_var - std_of_mem_var * std_threshold

        out_mu = (self.layer.src_running_mean >= out_upper_mu) | (out_lower_mu >= self.layer.src_running_mean)
        out_var = (self.layer.src_running_var >= out_upper_var) | (out_lower_var >= self.layer.src_running_var)

        # FIXME: channel-wise selection
        output_mu = torch.where(out_mu, a_mu, b_mu)
        output_var = torch.where(out_var, a_var, b_var)

        self.last_prior_mu = out_mu
        self.last_prior_var = out_var

        return output_mu, output_var

def BatchRenorm(batch, new_mu, new_var, eps, add_eps_numer=False):
    # TODO: need to contain eps?
    batch_mu = batch.mean([0, 2, 3])
    batch_var = batch.var([0, 2, 3], unbiased=False)

    if add_eps_numer:
        r = torch.sqrt(batch_var.detach() + eps) / torch.sqrt(new_var + eps)
    else:
        r = torch.sqrt(batch_var.detach()) / torch.sqrt(new_var + eps)
    d = (batch_mu.detach() - new_mu) / torch.sqrt(new_var + eps)

    output = (batch - batch_mu[None, :, None, None]) / torch.sqrt(batch_var[None, :, None, None] + eps) * r[None, :, None, None] + d[None, :, None, None]

    return output
