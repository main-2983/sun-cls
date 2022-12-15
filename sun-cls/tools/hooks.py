import time
from typing import List, Union

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
    This hook attached to a nn.Module.
    This hook is used to measure forward time of a nn.Module
    In order to view time after a forward pass, access self.eval
    Args:
        module (nn.Module): a nn.Module object to attach the hook

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


class ModuleForwardTimerHook:
    """
    This hook is an extended for LayerForwardTimerHook.
    It supports multiple module while LayerForwardTimerHook only supports one.
    Forward time for each module is stored in self.timer_dict
        with key being name of that module
        and value being forward time
    Args:
        module (List[nn.Module]): List nn.Module objects to attach the hook
        module_name (List[str]): List of nn.Module names
    """
    def __init__(self,
                 module: List[nn.Module],
                 module_name: List[str]):
        self.length = len(module)
        self.timer_dict = dict()
        self.hook_list = []

        for m, n in zip(module, module_name):
            self.hook_list.append(
                LayerForwardTimerHook(m)
            )
            self.timer_dict[n] = 0

    def assign_timer(self):
        for m_name, hook in zip(list(self.timer_dict.keys()),
                                self.hook_list):
            self.timer_dict[m_name] = hook.eval

    def close(self):
        for h in self.hook_list:
            h.close()