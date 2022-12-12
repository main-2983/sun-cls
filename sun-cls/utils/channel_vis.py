import os
import sys
import argparse
import logging
from PIL import Image
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch

from timm.models import create_model, apply_test_time_pool
from timm.data import resolve_data_config
from timm.data.transforms_factory import transforms_imagenet_eval
from timm.utils import setup_default_logging

from hooks import IOHook

_logger = logging.getLogger('Featmap vis')

def parse_args():
    parser = argparse.ArgumentParser(description='Feature maps visualization')
    parser.add_argument('data', type=str,
                        help='path to image')
    parser.add_argument('--model', '-m', type=str,
                        help='model arch')
    parser.add_argument('--print-model', action='store_true',
                        help='Print all layers of model')
    parser.add_argument('--layer', default=None, type=str,
                        help='Model layer to all channel activation map')
    parser.add_argument('--before', action='store_true',
                        help='Visualize input feature maps instead of output')
    parser.add_argument('--num-chans', default=None, type=int,
                        help='Number of channels to visualize, default is all')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--img-size', default=None, type=int, metavar='N',
                        help='Input dim, use model default if empty')
    parser.add_argument('--input-size', default=None, type=int,
                        metavar='N N N', help='Input of all image dim')
    parser.add_argument('--crop-pct', default=None, type=float, metavar='N',
                        help='Input image crop pct')
    parser.add_argument('--crop-mode', default=None, type=str, metavar='N',
                        help='Input image crop mode (squash, border, center). Model default if None.')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number classes in dataset')
    parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                        help='enable test time pool')
    parser.add_argument('--device', default='cuda', type=str,
                        help="Device (accelerator) to use.")
    parser.add_argument('--save-path', default=None, type=str,
                        help='Path to save visualize results')
    parser.add_argument('--view', action='store_true',
                        help='View visualization')
    return parser.parse_args()


def main():
    setup_default_logging()
    args = parse_args()
    args.pretrained = True or not args.checkpoint

    device = torch.device(args.device)

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
    )
    # inspect model's layers
    if args.print_model:
        print(model)
        sys.exit()

    # get layer for hook
    layer = eval(args.layer)
    # register hook
    hook = IOHook(layer)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    _logger.info(
        f'Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    model.eval()

    img_size = data_config['input_size'][1:]
    transform = transforms_imagenet_eval(
        img_size=img_size,
        crop_pct=data_config['crop_pct']
    )

    with torch.no_grad():
        image = Image.open(args.data)
        inp_tensor = transform(image)[None] # expand for batch dim
        inp_tensor.to(device)
        model(inp_tensor)
        if not args.before:
            activations = hook.output
        else:
            activations = hook.input[0]
        s = "input" if args.before else "output"
        _logger.info(f"Activation map at layer {str(layer)} has shape: {activations.shape}")
        # visualize
        activations = activations[0].cpu().numpy()
        c, h, w = activations.shape
        nrows, ncols = args.num_chans or int(np.sqrt(c)), args.num_chans or int(np.sqrt(c))
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
        for i in tqdm(range(nrows)):
            for j in range(ncols):
                axes[i, j].imshow(activations[i+j, :, :])
                axes[i, j].axis('off')
        if args.save_path is not None:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            plt.savefig(f"{args.save_path}/vis_{layer}_{s}.png")
        if args.view:
            plt.show()


if __name__ == '__main__':
    main()