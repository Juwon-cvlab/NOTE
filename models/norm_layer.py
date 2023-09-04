import torch
import torch.nn as nn
from torch.nn import functional as F

def adapt_alpha_bn(model, alpha):
    return AlphaBN.adapt_model(model.net, alpha)

class AlphaBN(nn.Module):
    @ staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is not None:
            for name, child in parent.named_children():
                #FIXME: this may cause error if it will be combined with Tent or optimization method
                child.requires_grad_(False)

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
