from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreadModel

def define_model(args):
    if args.model == 'wideresnet28-10':
        model = load_model('Strandard', dataset='cifar10', threat_model=ThreatModel.corruptions)
    elif args.model == 'wideresnet40-2':
        model = load_model('Hendrycks2020AugMix_WRN', dataset='cifar10', threat_model=ThreatModel.corruptions)
    return model