"""Conditional DDPM and conditional GAN for joint ocean-map forecasting.

No pretrained weights, no future covariates. Uses only historical fields,
land/sea mask and normalized spatial coordinates. Shapes:
    history [B,T,C,H,W], future samples [B,K,C,H,W].
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _group(c):
    return max(g for g in (8, 4, 2, 1) if c % g == 0)


class ResBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.a = nn.Conv2d(inc, outc, 3, padding=1)
        self.b = nn.Conv2d(outc, outc, 3, padding=1)
        self.norm1 = nn.GroupNorm(_group(outc), outc)
        self.norm2 = nn.GroupNorm(_group(outc), outc)
        self.shortcut = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(self.a(x)))
        h = self.norm2(self.b(h))
        return F.silu(h + self.shortcut(x))


class HistoryEncoder(nn.Module):
    """Compress all T historical maps, retaining spatiotemporal information."""
    def __init__(self, channels, width=32):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv3d(channels, width, (3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.GroupNorm(_group(width), width), nn.SiLU(),
            nn.Conv3d(width, width, (3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.GroupNorm(_group(width), width), nn.SiLU(),
        )
        self.fuse = ResBlock(width + 3 * channels + 3, width)

    def forward(self, history, mask):
        b, t, c, h, w = history.shape
        m = mask.to(device=history.device, dtype=history.dtype).reshape(1, 1, h, w)
        z = self.temporal(history.permute(0, 2, 1, 3, 4)).mean(dim=2)
        z = F.interpolate(z, size=(h, w), mode='bilinear', align_corners=False)
        yy = torch.linspace(-1, 1, h, device=history.device, dtype=history.dtype)
        xx = torch.linspace(-1, 1, w, device=history.device, dtype=history.dtype)
        coords = torch.stack(torch.meshgrid(yy, xx, indexing='ij'), 0).unsqueeze(0).expand(b, -1, -1, -1)
        context = torch.cat([
            z, history[:, -1], history.mean(1),
            (history[:, -1] - history[:, 0]) / max(t - 1, 1),
            m.expand(b, -1, -1, -1), coords,
        ], 1)
        return self.fuse(context) * m


class SpatialUNet(nn.Module):
    def __init__(self, future_channels, width=32, condition_width=32, with_time=True):
        super().__init__()
        self.with_time = with_time
        self.enc0 = ResBlock(future_channels + condition_width, width)
        self.enc1 = ResBlock(width, width * 2)
        self.enc2 = ResBlock(width * 2, width * 4)
        self.mid = ResBlock(width * 4, width * 4)
        self.dec1 = ResBlock(width * 4 + width * 2, width * 2)
        self.dec0 = ResBlock(width * 2 + width, width)
        self.output = nn.Conv2d(width, future_channels, 3, padding=1)
        if with_time:
            self.time_mlp = nn.Sequential(nn.Linear(width * 4, width * 4), nn.SiLU(), nn.Linear(width * 4, width * 4))

    def forward(self, noisy, condition, steps=None):
        h0 = self.enc0(torch.cat([noisy, condition], 1))
        h1 = self.enc1(F.avg_pool2d(h0, 2, ceil_mode=True))
        h2 = self.enc2(F.avg_pool2d(h1, 2, ceil_mode=True))
        if self.with_time:
            dim = h2.size(1)
            freq = torch.exp(torch.arange(0, dim // 2, device=h2.device, dtype=torch.float32)
                             * (-math.log(10000) / max(dim // 2 - 1, 1)))
            angles = steps.float()[:, None] * freq[None]
            embedding = torch.cat([angles.sin(), angles.cos()], -1)
            if embedding.size(1) < dim:
                embedding = F.pad(embedding, (0, dim - embedding.size(1)))
            h2 = h2 + self.time_mlp(embedding).to(h2.dtype)[:, :, None, None]
        h2 = self.mid(h2)
        u1 = F.interpolate(h2, size=h1.shape[-2:], mode='bilinear', align_corners=False)
        u1 = self.dec1(torch.cat([u1, h1], 1))
        u0 = F.interpolate(u1, size=h0.shape[-2:], mode='bilinear', align_corners=False)
        return self.output(self.dec0(torch.cat([u0, h0], 1)))


class ConditionalDDPM(nn.Module):
    def __init__(self, channels, in_len=30, out_len=3, width=32, diffusion_steps=100):
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError('diffusion_steps must be >= 2')
        self.channels, self.out_len = int(channels), int(out_len)
        self.condition = HistoryEncoder(channels, width)
        self.denoiser = SpatialUNet(channels * out_len, width, width, with_time=True)
        # Cosine schedule from Improved DDPM (Nichol & Dhariwal 2021).
        times = torch.linspace(0, 1, diffusion_steps + 1, dtype=torch.float64)
        a_bar = torch.cos((times + .008) / 1.008 * math.pi / 2).square()
        a_bar = a_bar / a_bar[0]
        betas = (1 - a_bar[1:] / a_bar[:-1]).clamp(0.0001, 0.999).float()
        alphas = 1 - betas
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alphas.cumprod(0))

    def training_loss(self, history, target, mask):
        b, k, c, h, w = target.shape
        m = mask.to(history.device, history.dtype).reshape(1, 1, h, w)
        residual = (target - history[:, -1:]).reshape(b, k*c, h, w) * m
        steps = torch.randint(0, len(self.betas), (b,), device=history.device)
        a_bar = self.alpha_cumprod[steps].reshape(b, 1, 1, 1)
        noise = torch.randn_like(residual) * m
        noisy = a_bar.sqrt() * residual + (1 - a_bar).sqrt() * noise
        estimated = self.denoiser(noisy, self.condition(history, mask), steps)
        return ((estimated - noise).square() * m).sum() / (m.sum() * b * k*c).clamp_min(1)

    @torch.no_grad()
    def sample(self, history, mask, members=1):
        self.eval()
        b, _, c, h, w = history.shape
        m = mask.to(history.device, history.dtype).reshape(1, 1, h, w)
        condition = self.condition(history, mask)
        generated = []
        for _ in range(int(members)):
            z = torch.randn(b, c*self.out_len, h, w, device=history.device, dtype=history.dtype) * m
            for idx in range(len(self.betas)-1, -1, -1):
                t = torch.full((b,), idx, device=history.device, dtype=torch.long)
                noise_pred = self.denoiser(z, condition, t)
                alpha = self.alphas[idx]
                alpha_bar = self.alpha_cumprod[idx]
                # Posterior with a bounded normalized x0 estimate to avoid
                # catastrophic reverse-time excursions for early checkpoints.
                x0 = ((z - (1-alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()).clamp(-8, 8)
                if idx > 0:
                    prev_ab = self.alpha_cumprod[idx-1]
                    coef0 = self.betas[idx] * prev_ab.sqrt() / (1-alpha_bar)
                    coef_t = (1-prev_ab) * alpha.sqrt() / (1-alpha_bar)
                    z = coef0*x0 + coef_t*z
                    posterior_var = self.betas[idx] * (1-prev_ab) / (1-alpha_bar)
                    z += posterior_var.sqrt() * torch.randn_like(z) * m
                else:
                    z = x0
                z = z * m
            generated.append(history[:, -1:] + z.reshape(b, self.out_len, c, h, w))
        return torch.stack(generated, 1)


class PatchCritic(nn.Module):
    """Conditional patch discriminator on future residual and recent observations."""
    def __init__(self, channels, out_len, width=32):
        super().__init__()
        count = channels * (out_len + 3) + 1
        def block(a, b):
            return nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(a, b, 4, stride=2, padding=1)), nn.LeakyReLU(.2))
        self.net = nn.Sequential(block(count, width), block(width, width*2),
                                 block(width*2, width*4), nn.Conv2d(width*4, 1, 3, padding=1))

    def forward(self, history, future, mask):
        b, k, c, h, w = future.shape
        m = mask.to(history.device, history.dtype).reshape(1, 1, h, w)
        context = [history[:, -1], history.mean(1), history[:, -1] - history[:, -2]]
        data = (future - history[:, -1:]).reshape(b, k*c, h, w)
        return self.net(torch.cat([*(v*m for v in context), data*m, m.expand(b, -1, -1, -1)], 1))


class ConditionalGAN(nn.Module):
    def __init__(self, channels, in_len=30, out_len=3, width=32):
        super().__init__()
        self.channels, self.out_len = channels, out_len
        self.condition = HistoryEncoder(channels, width)
        self.generator = SpatialUNet(channels*out_len, width, width, with_time=False)
        self.critic = PatchCritic(channels, out_len, width)

    def generate(self, history, mask):
        b, _, c, h, w = history.shape
        m = mask.to(history.device, history.dtype).reshape(1, 1, h, w)
        noise = torch.randn((b, c*self.out_len, h, w), device=history.device, dtype=history.dtype) * m
        residual = self.generator(noise, self.condition(history, mask)) * m
        return history[:, -1:] + residual.reshape(b, self.out_len, c, h, w)

    @torch.no_grad()
    def sample(self, history, mask, members=1):
        self.eval()
        return torch.stack([self.generate(history, mask) for _ in range(int(members))], 1)
