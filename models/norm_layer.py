import torch
import torch.nn as nn
from torch.nn import functional as F

def adapt_alpha_bn(model, alpha):
    return AlphaBN.adapt_model(model.net, alpha)
def adapt_memory_bn(model, memory_size, use_prior, batch_renorm, add_eps_numer, use_dynamic_weight, use_binary_select, std_threshold):
    return BatchNormWithMemory.adapt_model(model.net, memory_size, use_prior, batch_renorm, add_eps_numer,
                                           use_dynamic_weight, use_binary_select, std_threshold)


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

class BatchNormWithMemory(nn.Module):
    @staticmethod
    def find_bns(parent, memory_size, use_prior, batch_renorm=False, add_eps_numer=False,
                 use_dynamic_weight=False, use_binary_select=False, std_threshold=1.0):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            # child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = BatchNormWithMemory(child, memory_size, use_prior, batch_renorm, add_eps_numer,
                                             use_dynamic_weight, use_binary_select, std_threshold)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BatchNormWithMemory.find_bns(child, memory_size, use_prior, batch_renorm, add_eps_numer,
                                                                 use_dynamic_weight, use_binary_select, std_threshold))
    
        return replace_mods

    @staticmethod
    def adapt_model(model, memory_size, use_prior=None, batch_renorm=False, add_eps_numer=False,
                    use_dynamic_weight=False, use_binary_select=False, std_threshold=1.0):
        replace_mods = BatchNormWithMemory.find_bns(model, memory_size, use_prior, batch_renorm=batch_renorm, add_eps_numer=add_eps_numer,
                                                    use_dynamic_weight=use_dynamic_weight, use_binary_select=use_binary_select, std_threshold=std_threshold)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, memory_size, use_prior=None, batch_renorm=False, add_eps_numer=False,
                 use_dynamic_weight=False, use_binary_select=False, std_threshold=1.0, pred_module_type=0):
        super().__init__()

        self.layer = layer

        self.memory_size = memory_size
        self.use_prior = use_prior
        self.batch_renorm = batch_renorm

        self.batch_mu_memory = torch.randn(size=(memory_size, self.layer.num_features), device=self.layer.weight.device)
        self.batch_var_memory = torch.randn(size=(memory_size, self.layer.num_features), device=self.layer.weight.device)

        self.unbiased = False

        self.batch_pointer = 0
        self.batch_full = False

        self.add_eps_numer = add_eps_numer

        self.use_dynamic_weight = use_dynamic_weight
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
        
    def reset(self):
        self.pointer = 0
        self.full = False

        self.batch_pointer = 0
        self.batch_full = False

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
        last_index = (self.batch_pointer - 1) % self.memory_size

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
                test_mean = self.batch_mu_memory[:self.batch_pointer, :]
                test_var = self.batch_var_memory[:self.batch_pointer, :]

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
        return self.batch_mu_memory[:self.batch_pointer], self.batch_var_memory[:self.batch_pointer]

    def forward(self, input):
        # self._check_input_dim(input)
        
        # TODO: use mean_and_var or batch_norm(trainaing=True, momentum=1.0)???
        batch_mu = input.mean([0, 2, 3])
        batch_var = input.var([0, 2, 3], unbiased=False)
        # batch_num = 1

        # save mu and variance in memory
        batch_start = self.batch_pointer
        batch_end = self.batch_pointer + 1
        batch_idxs_replace = torch.arange(batch_start, batch_end).to(input.device) % self.memory_size

        self.batch_mu_memory[batch_idxs_replace, :] = batch_mu.detach()
        self.batch_var_memory[batch_idxs_replace, :] = batch_var.detach()

        self.batch_pointer = batch_end % self.memory_size

        if batch_end >= self.memory_size:
            self.batch_full = True

        # compute test mu and variance from in-memory elements
        if self.batch_full:
            test_mean = self.batch_mu_memory
            test_var = self.batch_var_memory
        else:
            test_mean = self.batch_mu_memory[:self.batch_pointer, :]
            test_var = self.batch_var_memory[:self.batch_pointer, :]

        test_mean = torch.mean(test_mean, 0)
        test_var = torch.mean(test_var, 0)

        if self.use_dynamic_weight:
            prior_mu, prior_var = self.dynamic_weight(batch_mu, batch_var)
            test_mean = (
                prior_mu * self.layer.src_running_mean
                + (1 - prior_mu) * test_mean
            )
            test_var = (
                prior_var * self.layer.src_running_var
                + (1 - prior_var) * test_var
            )
        elif self.use_binary_select:
            test_mean, test_var = self.binary_selection(self.layer.src_running_mean,
                                                        self.layer.src_running_var,
                                                        test_mean, test_var, self.std_threshold)
        elif self.pred_module_type == 1 or self.pred_module_type == 2:
            if self.pred_module_type == 1:
                module_input1 = torch.cat((test_mean, self.layer.src_running_mean))
                module_input2 = torch.cat((test_var, self.layer.src_running_var))
            elif self.pred_module_type == 2:
                module_input1 = torch.cat((batch_mu, test_mean, self.layer.src_running_mean))
                module_input2 = torch.cat((batch_var, test_var, self.layer.src_running_var))
            else:
                raise NotImplemented

            alpha_mu = self.pred_module1(module_input1)
            alpha_var = self.pred_module2(module_input2)

            self.last_prior_mu = alpha_mu.detach()
            self.last_prior_var = alpha_var.detach()

            test_mean = (
                alpha_mu * self.layer.src_running_mean
                + (1 - alpha_mu) * test_mean
            )
            test_var = (
                alpha_var * self.layer.src_running_var
                + (1 - alpha_var) * test_var
            )
        elif self.pred_module_type == 3 or self.pred_module_type == 4:
            if self.pred_module_type == 3:
                module_input = torch.cat((test_mean, self.layer.src_running_mean, test_var, self.layer.src_running_var))
            elif self.pred_module_type == 4:
                module_input = torch.cat((batch_mu, test_mean, self.layer.src_running_mean, batch_var, test_var, self.layer.src_running_var))
            else:
                raise NotImplemented

            alpha_mu, alpha_var = self.pred_module(module_input)

            self.last_prior_mu = alpha_mu.detach()
            self.last_prior_var = alpha_var.detach()

            test_mean = (
                alpha_mu * self.layer.src_running_mean
                + (1 - alpha_mu) * test_mean
            )
            test_var = (
                alpha_var * self.layer.src_running_var
                + (1 - alpha_var) * test_var
            )
        else:
            if self.use_prior:
                test_mean = (
                    self.use_prior * self.layer.src_running_mean
                    + (1 - self.use_prior) * test_mean
                )
                test_var = (
                    self.use_prior * self.layer.src_running_var
                    + (1 - self.use_prior) * test_var
                )

        # input = (input - batch_mu[None, :, None, None]) / (torch.sqrt(batch_var[None, :, None, None] + self.eps))

        # if self.affine:
        #     input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        if self.batch_renorm:
            input = BatchRenorm(input, test_mean, test_var, self.layer.eps, self.add_eps_numer)
            input = self.layer.weight[None, :, None, None] * input + self.layer.bias[None, :, None, None]

            return input
        else:
            return F.batch_norm(
                input,
                test_mean,
                test_var,
                self.layer.weight,
                self.layer.bias,
                False,
                0,
                self.layer.eps
            )

    def dynamic_weight(self, test_mu, test_var):
        # TODO: layer-wise interpolation
        test2src_mu = torch.cdist(test_mu.unsqueeze(0), self.layer.src_running_mean.unsqueeze(0)).squeeze()
        test2src_var = torch.cdist(test_var.unsqueeze(0), self.layer.src_running_var.unsqueeze(0)).squeeze()

        mem_mean, mem_var = self.get_batch_mu_and_var()
        mem_mean = torch.mean(mem_mean, 0)
        mem_var = torch.mean(mem_var, 0)

        test2mem_mu = torch.cdist(test_mu.unsqueeze(0), mem_mean.unsqueeze(0)).squeeze()
        test2mem_var = torch.cdist(test_var.unsqueeze(0), mem_var.unsqueeze(0)).squeeze()

        # TODO: same weight or separated weight?
        prior_mu = 1 - (test2src_mu / (test2src_mu + test2mem_mu))
        prior_var = 1 - (test2src_var / (test2src_var + test2mem_var))

        prior_mu = prior_mu.detach()
        prior_var = prior_var.detach()

        self.last_prior_mu = prior_mu
        self.last_prior_var = prior_var

        return prior_mu, prior_var

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
