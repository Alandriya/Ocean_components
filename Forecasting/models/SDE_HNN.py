import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- SDE Block for spatiotemporal data -----
class SDEHNN(nn.Module):
    def __init__(self, output_length, in_channels=1, hidden_channels=32, step_size=0.5, num_steps=3, dropout_p=0.1):
        super().__init__()
        self.dt = step_size
        self.num_steps = num_steps
        self.dropout_p = dropout_p
        self.output_length = output_length

        # Drift network (deterministic dynamics → mean)
        self.drift_net = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        )

        # Diffusion network (stochastic dynamics → variance)
        self.diff_net = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.Softplus()   # ensures positive variance scaling
        )

        self.dropout = nn.Dropout2d(p=dropout_p)
        self.encoder = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.decoder_mu = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        self.decoder_sigma2 = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Softplus()
        )

    def forward(self, x_seq):
        """
        x_seq: (batch, seq_len_in, 1, 81, 91)
        Returns predictions for next seq_len_out frames.
        """
        B, T_in, C, H, W = x_seq.shape
        h = torch.zeros(B, 32, H, W, device=x_seq.device)

        # Encode temporal input sequence into hidden representation
        for t in range(T_in):
            frame = x_seq[:, t]
            h = F.relu(self.encoder(frame)) + h  # simple temporal accumulation

        # Integrate dynamics using Euler–Maruyama SDE
        preds_mu, preds_sigma2 = [], []
        for _ in range(self.output_length):  # predict S future frames
            for _ in range(self.num_steps):
                f_t = self.drift_net(h)
                g_t = self.diff_net(h)
                g_t = self.dropout(g_t)
                epsilon = torch.randn_like(h)
                h = h + f_t * self.dt + g_t * torch.sqrt(torch.tensor(self.dt)) * epsilon

            mu = self.decoder_mu(h)
            sigma2 = self.decoder_sigma2(h)
            preds_mu.append(mu)
            preds_sigma2.append(sigma2)

        mu_seq = torch.stack(preds_mu, dim=1)        # (B, S, 1, H, W)
        sigma2_seq = torch.stack(preds_sigma2, dim=1)
        return mu_seq, sigma2_seq
