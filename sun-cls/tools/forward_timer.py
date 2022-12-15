import sys
import argparse
import logging

import torch

from timm.models import create_model
from timm.utils import setup_default_logging

from hooks import ModuleForwardTimerHook

_logger = logging.getLogger('Forward timer')


def parse_args():
    parser = argparse.ArgumentParser(description="Check forward pass time of modules")
    parser.add_argument('model', type=str, help='Model architecture')
    parser.add_argument('--modules', nargs='+', type=str, required=False,
                        help='List of module to check forward time')
    parser.add_argument('--shape', nargs='+', type=int, metavar='N N N N',
                        default=[32, 3, 224, 224], help='Feature maps shape as input')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device')
    parser.add_argument('--print-model', action='store_true',
                        help='Print all model layers')
    return parser.parse_args()

def main():
    setup_default_logging()
    args = parse_args()

    device = torch.device(args.device)
    # create model
    model = create_model(
        args.model,
        num_classes=None,
        in_chans=3,
        pretrained=False
    )
    model = model.to(device)
    # inspect model's layers
    if args.print_model:
        print(model)
        sys.exit()
    if len(args.modules) == 0:
        _logger.warning("No modules to get timer")
        sys.exit()
    # warmup
    featmaps = torch.rand((args.shape)).to(device)
    model(featmaps)
    # register layers and its name for hook
    modules = []
    module_names = []
    for module in args.modules:
        modules.append(eval(module))
        module_names.append(module)
    hook = ModuleForwardTimerHook(modules, module_names)
    with torch.no_grad():
        _logger.info(f"Forward timer:")
        out = model(featmaps)
        hook.assign_timer()
        for k, v in hook.timer_dict.items():
            print(f"- {k}: {v:.6f}s")


if __name__ == '__main__':
    main()