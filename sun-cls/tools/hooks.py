import time
from typing import Union, List
from collections import Iterable

import torch.nn as nn


class IOHook:
    """
    This hook attached to a nn.Module, it supports both backward and forward hooks
    This hook provides access to feature maps from a nn.Module, it can be either
    the input feature maps to that nn.Module, or output feature maps from nn.Module
    Args:
        module (nn.Module): a nn.Module object to attach the hook
        backward (bool): get feature maps in backward phase
    """
    def __init__(self, module: nn.Module, backward=False):
        self.backward = backward
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class LayerForwardTimerHook:
    def __init__(self, module: nn.Module):
        self.pre_forward = module.register_forward_pre_hook(self.set_start_timer)
        self.after_forward = module.register_forward_hook(self.set_end_timer)

        self.eval = 0

    def set_start_timer(self, module, input):
        self.start = time.time()

    def set_end_timer(self, module, input, output):
        self.end = time.time()
        self.eval += (self.end - self.start)

    def close(self):
        self.pre_forward.remove()
        self.after_forward.remove()


class ModuleForwardTimerHook:
    def __init__(self, module: Union[List, nn.Module]):
        self.length = self._get_length(module)

    def _get_length(self, module: Union[List, nn.Module]):
        length = 0
        # make all module becomes nn.ModuleList
        if not isinstance(module, Iterable):
            module = [module]
        if isinstance(module, list):
            module = nn.ModuleList(module)
        # count only non-ModuleList and non-Sequential module
        for m in module.modules():
            if type(m) != nn.ModuleList and type(m) != nn.Sequential:
                length += 1

        return length
