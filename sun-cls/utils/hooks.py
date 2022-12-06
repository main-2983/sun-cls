import torch.nn as nn


class IOHook:
    def __init__(self, module: nn.Module, backward=False):
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

