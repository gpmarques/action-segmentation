import math
import torch
import numpy as np


class PositionalEncoding:

    def __call__(self, data: np.ndarray) -> np.ndarray:
        d_model = data.shape[1]
        length = data.shape[0]

        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        den_exp = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * den_exp)
        pe[:, 1::2] = torch.cos(position.float() * den_exp)

        return data + pe.numpy()
