import torch
import torch.nn as nn
import numpy as np

# Positional Encoding adopted from https://arxiv.org/pdf/2110.05357.pdf
# Parameters are the same as in the paper.


class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model=36, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1
        )  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe


class Time2VecEncoding(nn.Module):
    def __init__(self, in_feats=1, out_feats=512) -> None:
        super(Time2VecEncoding, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.non_periodic = nn.Linear(in_feats, 1, bias=True)
        self.periodic = nn.Linear(in_feats, out_feats - 1, bias=True)

    def forward(self, timestamps):
        non_periodic_encoding = self.non_periodic(timestamps)
        periodic_encoding = torch.sin(self.periodic(timestamps))

        return torch.cat([non_periodic_encoding, periodic_encoding], dim=-1)


def timestamp_encoding(timestamps, d_model):
    """
    Encode the timestamps into a periodic signal using sine and cosine functions.

    Args:
        timestamps (Tensor): A tensor of shape (n_patients, n_measurements, 1) containing the timestamps.

    Returns:
        (Tensor, Tensor): A tuple of shape (n_patients, n_measurements, d_model) containing the encoded timestamps.
    """
    n_patients, n_measurements, _ = timestamps.shape
    # Normalize the timestamps
    timestamps = timestamps - timestamps.min()
    timestamps = timestamps / timestamps.max()
    # Encode the timestamps using Positional Encoding
    pe = PositionalEncodingTF(d_model=d_model)
    timestamps = pe(timestamps)
    return timestamps.squeeze(2)
