import torch
import torch.nn as nn
from torch.nn import functional as F

def adapt_alpha_bn(model, alpha):
    return AlphaBN.adapt_model(model.net, alpha)
def adapt_memory_bn(model, memory_size, use_prior, batch_renorm):
    return BatchNormWithMemory.adapt_model(model.net, memory_size, use_prior, batch_renorm)

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
    def find_bns(parent, memory_size, use_prior, batch_renorm=False, add_eps_numer=False):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            # child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = BatchNormWithMemory(child, memory_size, use_prior, batch_renorm, add_eps_numer)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BatchNormWithMemory.find_bns(child, memory_size, use_prior, batch_renorm, add_eps_numer))
    
        return replace_mods

    @staticmethod
    def adapt_model(model, memory_size, use_prior=None, batch_renorm=False, add_eps_numer=False):
        replace_mods = BatchNormWithMemory.find_bns(model, memory_size, use_prior, batch_renorm=batch_renorm, add_eps_numer=add_eps_numer)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, memory_size, use_prior=None, batch_renorm=False, add_eps_numer=False):
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

    def reset(self):
        self.pointer = 0
        self.full = False

        self.batch_pointer = 0
        self.batch_full = False

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
