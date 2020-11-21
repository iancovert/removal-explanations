import torch
import torch.nn as nn


class MaskLayer1d(nn.Module):
    '''
    Masking for 1d inputs.

    Args:
      append: whether to append the mask along channels dim.
      value: replacement value for held out features.
    '''
    def __init__(self, append=True, value=0):
        super().__init__()
        self.append = append
        self.value = value

    def forward(self, input_tuple):
        x, S = input_tuple
        x = x * S + self.value * (1 - S)
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x


class MaskLayer2d(nn.Module):
    '''
    Masking for 2d inputs.

    Args:
      append: whether to append the mask along channels dim.
      value: replacement value for held out features.
    '''
    def __init__(self, append=True, value=0):
        super().__init__()
        self.append = append
        self.value = value

    def forward(self, input_tuple):
        '''
        Apply mask to input.

        Args:
          input_tuple: tuple of input x and mask S.
        '''
        x, S = input_tuple
        x = x * S + self.value * (1 - S)
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x
