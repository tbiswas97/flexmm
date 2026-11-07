import torch

class PyramidNet(torch.nn.ModuleList):
    def __init__(self, modules):
        """
        In the constructor we instantiate two nn.Linear modules and assign them
        as member variables.
        """
        super(PyramidNet, self).__init__()
        if modules is not None:
            self += modules

    def forward(self, input):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        for module in self:
            out = module(input)
        return out
