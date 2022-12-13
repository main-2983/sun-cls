import time

import torch.nn as nn


class IOHook:
    """
    This hook attached to a nn.Module, it supports both backward and forward hooks.
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
    """
    This hook attached to a nn.Module to measure the forward time of that module.
    Call self.eval to get the execution time after perform a forward pass to a module.
    Args:
         module (nn.Module): a module to attach to hook
    """
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
