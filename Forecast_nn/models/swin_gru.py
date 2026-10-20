"""Multi-scale CNN + temporal cross-attention + shifted-window attention/ConvGRU.

Forecast_nn interface:
    x:      [B, T_in, C, H, W] (C=1 or C=2)
    mask:   [H,W] / [B,H,W] / [B,1,H,W]
    result: [B, T_out, C, H, W]

* Uses only observed input frames (no known future forcings).
* Direct multi-horizon residual predictions relative to the last input frame.
* The temporal compressor sees all T_in encoded frames; ConvGRU then integrates
  `temporal_tokens` compressed features at the coarse grid.
* Shifted-window self attention mixes spatially within the recurrent block.
* Does not implement SDE conditioning or stochastic sampling.

The module is an original hybrid inspired by Swin Transformer/SwinLSTM, NOT an
exact reproduction of the SwinLSTM paper.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.shortcut = (nn.Identity() if in_channels == out_channels and stride == 1
                         else nn.Conv2d(in_channels, out_channels, 1, stride=stride))
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            _norm(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            _norm(out_channels),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return F.gelu(self.shortcut(x) + self.drop(self.main(x)))


class TemporalCompressor(nn.Module):
    """Learned time queries compress T input maps to K steps at each grid cell.

    Attention is along time only; no quadratic attention over H*W pixels.
    Uses PyTorch scaled_dot_product_attention (Flash attention when supported).
    """
    def __init__(self, channels, in_len, tokens=8, heads=8):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must be divisible by attention heads')
        self.heads = heads
        self.channels = channels
        self.tokens = tokens
        self.positions = nn.Parameter(torch.randn(1, in_len, channels) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, tokens, channels) * 0.02)
        self.kv_norm = nn.LayerNorm(channels)
        self.q_norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.proj = nn.Linear(channels, channels)
        self.post_norm = nn.LayerNorm(channels)

    def forward(self, x):
        # [B,T,C,H,W] -> [B*H*W,T,C]
        b, t, c, h, w = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        kv = self.kv(self.kv_norm(x + self.positions)).reshape(b*h*w, t, 2, self.heads, c//self.heads)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        q = self.q(self.q_norm(self.queries)).expand(b*h*w, -1, -1)
        q = q.reshape(b*h*w, self.tokens, self.heads, c//self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(b*h*w, self.tokens, c)
        y = self.post_norm(self.proj(y))
        return y.reshape(b, h, w, self.tokens, c).permute(0, 3, 4, 1, 2).contiguous()


def _partition(x, window):
    # x: [B,H,W,C] where H and W already multiples of window.
    b,h,w,c=x.shape
    return x.reshape(b,h//window,window,w//window,window,c).permute(0,1,3,2,4,5).reshape(-1, window*window,c)


def _unpartition(x, window, b, h, w, c):
    return x.reshape(b,h//window,w//window,window,window,c).permute(0,1,3,2,4,5).reshape(b,h,w,c)


def _shift_mask(h, w, window, shift, device):
    """Exclude artificial connections across cyclic shifted-window boundaries."""
    if not shift:
        return None
    ids = torch.zeros((1, h, w, 1), device=device, dtype=torch.int64)
    slices = (slice(0,-window),slice(-window,-shift),slice(-shift,None))
    n=0
    for hs in slices:
        for ws in slices:
            ids[:,hs,ws,:]=n
            n+=1
    blocks=_partition(ids,window).squeeze(-1)
    return blocks[:, :, None] != blocks[:, None, :]


class ShiftedWindowAttention(nn.Module):
    def __init__(self, channels, heads=8, window_size=4, shift=False, dropout=0.05):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must be divisible by heads')
        self.channels = channels
        self.heads = heads
        self.window = window_size
        self.shift = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.attn_drop = float(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(nn.Linear(channels, channels*4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(channels*4, channels))
        self.dropout = nn.Dropout(dropout)
        # Relative position bias distinguishes offsets within local windows.
        span = 2*window_size-1
        self.bias = nn.Parameter(torch.zeros(heads, span*span))
        pos=torch.stack(torch.meshgrid(torch.arange(window_size),torch.arange(window_size),indexing='ij'))
        pos=pos.reshape(2,-1)
        delta=pos[:,:,None]-pos[:,None,:]+window_size-1
        pos_index=delta[0]*span+delta[1]
        self.register_buffer('position_index',pos_index.reshape(-1).long(),persistent=False)
        nn.init.trunc_normal_(self.bias,std=0.02)

    def forward(self, x):
        b,c,h,w=x.shape
        win=self.window
        hp=math.ceil(h/win)*win
        wp=math.ceil(w/win)*win
        z=x.permute(0,2,3,1)
        z=self.norm1(z)
        if (hp,wp)!=(h,w):
            z=F.pad(z,(0,0,0,wp-w,0,hp-h))
        if self.shift:
            z=torch.roll(z,shifts=(-self.shift,-self.shift),dims=(1,2))
        windows=_partition(z,win)
        bw,n,_=windows.shape
        qkv=self.qkv(windows).reshape(bw,n,3,self.heads,c//self.heads).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        bias=self.bias[:,self.position_index].reshape(self.heads,n,n)
        additive=bias.unsqueeze(0).to(dtype=q.dtype).expand(bw,-1,-1,-1).clone()
        # Padded region should not participate as attention keys.
        if hp!=h or wp!=w:
            valid=torch.ones((1,h,w,1),device=x.device,dtype=torch.bool)
            valid=F.pad(valid,(0,0,0,wp-w,0,hp-h),value=False)
            if self.shift:
                valid=torch.roll(valid,shifts=(-self.shift,-self.shift),dims=(1,2))
            vwin=_partition(valid,win).squeeze(-1)
            invalid=(~vwin).repeat(b,1)
            additive=additive.masked_fill(invalid[:,None,None,:],-100.0)
        if self.shift:
            invalid=_shift_mask(hp,wp,win,self.shift,x.device)
            nwin=(hp//win)*(wp//win)
            invalid=invalid.unsqueeze(0).expand(b,-1,-1,-1).reshape(bw,n,n)
            additive=additive.masked_fill(invalid[:,None],-100.0)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=additive,
                dropout_p=self.attn_drop if self.training else 0.0)
        y=y.transpose(1,2).reshape(bw,n,c)
        y=self.proj(y)
        y=_unpartition(y,win,b,hp,wp,c)
        if self.shift:
            y=torch.roll(y,shifts=(self.shift,self.shift),dims=(1,2))
        y=y[:,:h,:w]
        x=x+self.dropout(y.permute(0,3,1,2))
        x=x+self.dropout(self.ff(self.norm2(x.permute(0,2,3,1))).permute(0,3,1,2))
        return x


class ConvGRUCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates=nn.Conv2d(channels*2,channels*2,3,padding=1)
        self.candidate=nn.Conv2d(channels*2,channels,3,padding=1)

    def forward(self, x, h):
        if h is None:
            h=torch.zeros_like(x)
        reset,update=self.gates(torch.cat([x,h],dim=1)).chunk(2,dim=1)
        reset=torch.sigmoid(reset)
        update=torch.sigmoid(update)
        candidate=torch.tanh(self.candidate(torch.cat([x,reset*h],dim=1)))
        return (1-update)*h+update*candidate


class SwinGRUForecaster(nn.Module):
    """Heavy deterministic multi-scale model; no external forcings or SDE."""
    def __init__(self, in_len, out_len, in_channels=1, base_channels=64,
                 temporal_tokens=8, attention_depth=2, heads=8,
                 window_size=4, dropout=0.05, residual=True,
                 use_mask_channel=True, use_coord_channels=True, mask_features=True):
        super().__init__()
        if in_len<1 or out_len<1 or temporal_tokens<1 or attention_depth<1:
            raise ValueError('All time lengths and attention_depth must be positive')
        if window_size<2:
            raise ValueError('window_size must be at least 2')
        self.in_len=int(in_len)
        self.out_len=int(out_len)
        self.in_channels=int(in_channels)
        self.residual=bool(residual)
        self.use_mask_channel=bool(use_mask_channel)
        self.use_coord_channels=bool(use_coord_channels)
        self.mask_features=bool(mask_features)
        base=int(base_channels)
        hidden=base*4
        stem_channels=in_channels+int(use_mask_channel)+2*int(use_coord_channels)
        self.stem=nn.Sequential(nn.Conv2d(stem_channels,base,3,padding=1,stride=2),
                                _norm(base),nn.GELU(),ConvBlock(base,base,dropout=dropout))
        self.down_mid=ConvBlock(base,base*2,stride=2,dropout=dropout)
        self.down_coarse=ConvBlock(base*2,hidden,stride=2,dropout=dropout)
        self.temporal=TemporalCompressor(hidden,in_len,temporal_tokens,heads)
        self.memory=ConvGRUCell(hidden)
        self.swin=nn.ModuleList([
            ShiftedWindowAttention(hidden,heads,window_size,shift=bool(i%2),dropout=dropout)
            for i in range(attention_depth)
        ])
        self.horizon_embedding=nn.Parameter(torch.randn(out_len,hidden,1,1)*0.02)
        self.decode_mid=ConvBlock(hidden+base*2,base*2,dropout=dropout)
        self.decode_fine=ConvBlock(base*3,base,dropout=dropout)
        self.output=nn.Sequential(ConvBlock(base+in_channels,base,dropout=dropout),
                                  nn.Conv2d(base,in_channels,1))
        # Initially exactly persistence, but the subsequent gradient steps can
        # learn independent nonzero horizon corrections.
        if residual:
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _mask(mask,b,h,w,x):
        if mask is None:
            return None
        if mask.ndim==2:
            mask=mask[None,None]
        elif mask.ndim==3:
            mask=mask[:,None]
        if mask.ndim!=4 or mask.size(1)!=1:
            raise ValueError('mask must have shape [H,W], [B,H,W] or [B,1,H,W]')
        if mask.shape[-2:]!=(h,w) or mask.size(0) not in (1,b):
            raise ValueError('mask size mismatch')
        return mask.to(device=x.device,dtype=x.dtype).expand(b,-1,-1,-1)

    def _mask_features(self,feat,mask):
        if mask is None or not self.mask_features:
            return feat
        return feat*F.interpolate(mask,size=feat.shape[-2:],mode='nearest')

    def forward(self,x,mask=None):
        if x.ndim!=5:
            raise ValueError('Input must be [B,T,C,H,W]')
        b,t,c,h,w=x.shape
        if t!=self.in_len or c!=self.in_channels:
            raise ValueError(f'Expected T={self.in_len}, C={self.in_channels}, got {tuple(x.shape)}')
        mask=self._mask(mask,b,h,w,x)
        if self.use_mask_channel and mask is None:
            raise ValueError('use_mask_channel=True needs mask')
        frames=[x]
        if self.use_mask_channel:
            frames.append(mask[:,None].expand(-1,t,-1,-1,-1))
        if self.use_coord_channels:
            yy=torch.linspace(-1,1,h,device=x.device,dtype=x.dtype)
            xx=torch.linspace(-1,1,w,device=x.device,dtype=x.dtype)
            yy,xx=torch.meshgrid(yy,xx,indexing='ij')
            coords=torch.stack((yy,xx),dim=0)[None,None].expand(b,t,-1,-1,-1)
            frames.append(coords)
        encoded=torch.cat(frames,dim=2).reshape(b*t,-1,h,w)
        p2=self.stem(encoded)
        p4=self.down_mid(p2)
        p8=self.down_coarse(p4)
        h2,w2=p2.shape[-2:]
        h4,w4=p4.shape[-2:]
        h8,w8=p8.shape[-2:]
        # Mask all history frames in the feature representations.
        if mask is not None and self.mask_features:
            m2=F.interpolate(mask,size=(h2,w2),mode='nearest')
            m4=F.interpolate(mask,size=(h4,w4),mode='nearest')
            m8=F.interpolate(mask,size=(h8,w8),mode='nearest')
            p2=p2*m2[:,None].expand(b,t,1,h2,w2).reshape(b*t,1,h2,w2)
            p4=p4*m4[:,None].expand(b,t,1,h4,w4).reshape(b*t,1,h4,w4)
            p8=p8*m8[:,None].expand(b,t,1,h8,w8).reshape(b*t,1,h8,w8)
        skip_fine=p2.reshape(b,t,-1,h2,w2)[:,-1]
        skip_mid=p4.reshape(b,t,-1,h4,w4)[:,-1]
        z=self.temporal(p8.reshape(b,t,-1,h8,w8))
        state=None
        for step in range(z.size(1)):
            state=self.memory(z[:,step],state)
            for block in self.swin:
                state=block(state)
                state=self._mask_features(state,mask)
        results=[]
        for horizon in range(self.out_len):
            d=state+self.horizon_embedding[horizon][None]
            d=F.interpolate(d,size=(h4,w4),mode='bilinear',align_corners=False)
            d=self._mask_features(self.decode_mid(torch.cat([d,skip_mid],dim=1)),mask)
            d=F.interpolate(d,size=(h2,w2),mode='bilinear',align_corners=False)
            d=self._mask_features(self.decode_fine(torch.cat([d,skip_fine],dim=1)),mask)
            d=F.interpolate(d,size=(h,w),mode='bilinear',align_corners=False)
            delta=self.output(torch.cat([d,x[:,-1]],dim=1))
            pred=x[:,-1]+delta if self.residual else delta
            if mask is not None:
                pred=pred*mask
            results.append(pred)
        return torch.stack(results,dim=1)
