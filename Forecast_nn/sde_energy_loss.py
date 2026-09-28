import torch
import torch.nn as nn


class SDEEnergyLoss(nn.Module):
    """Masked MSE plus the quadratic SDE-energy penalty from empirical a(x), b(x)."""
    def __init__(self, mask, bin_edges, a, b, sde_lambda=1e-4, eps=1e-6):
        super().__init__()
        self.sde_lambda = sde_lambda
        self.eps = eps
        self.register_buffer('mask', torch.as_tensor(mask, dtype=torch.bool)[None, None, None])
        self.register_buffer('bin_edges', torch.as_tensor(bin_edges, dtype=torch.float32))
        self.register_buffer('a', torch.as_tensor(a, dtype=torch.float32))
        self.register_buffer('b', torch.as_tensor(b, dtype=torch.float32))

    def forward(self, truth, pred, last_input):
        # Keep the loss itself in float32 even when the network forward pass uses AMP/float16.
        truth, pred, last_input = truth.float(), pred.float(), last_input.float()
        valid = self.mask.expand_as(pred)
        mse = (pred - truth).square().masked_select(valid).mean()
        prev = torch.cat([last_input[:, None], pred[:, :-1]], dim=1)
        delta = pred - prev
        terms = []
        for c in range(pred.size(2)):
            idx = torch.bucketize(prev[:, :, c].contiguous(), self.bin_edges[c, 1:-1])
            a = self.a[c][idx]
            b = self.b[c][idx]
            terms.append(0.5 * (delta[:, :, c] - a).square() / (b + self.eps))
        sde = torch.stack(terms, dim=2).masked_select(valid).mean()
        return mse + self.sde_lambda * sde, mse.detach(), sde.detach()
