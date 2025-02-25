from flax import linen as nn
from typing import Sequence

class NN_pf(nn.Module):
    input_size: int
    hidden_sizes: Sequence[int]  # A sequence (list or tuple) of hidden layer sizes
    output_size: int

    @nn.compact
    def __call__(self, x): 
        for size in self.hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.elu(x)
        x = nn.Dense(features=self.output_size)(x)
        return x