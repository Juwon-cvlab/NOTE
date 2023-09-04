from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

def define_model(args):
    if args.model == 'wideresnet28-10':
        model = load_model('Strandard', model_dir='./ckpt', dataset='cifar10', threat_model=ThreatModel.corruptions)
    elif args.model == 'wideresnet40-2':
        model = load_model('Hendrycks2020AugMix_WRN', model_dir='./ckpt', dataset='cifar10', threat_model=ThreatModel.corruptions)
    return model
