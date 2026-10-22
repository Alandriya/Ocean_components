"""Synthetic tests for the deterministic Swin-GRU and 2-flux pipeline."""
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from Forecast_nn.config import cfg
from Forecast_nn.models import build_model, available_models
from Forecast_nn.data import make_field_reader, OceanSequenceDataset
from Forecast_nn.tests.test_pipeline_integration import fixture


def _small_model(in_len=5, out_len=3, channels=2):
    return build_model('swin_gru', in_len, out_len, channels, {
        'base_channels': 8, 'temporal_tokens': 3, 'attention_depth': 2,
        'heads': 4, 'window_size': 4, 'dropout': 0.0,
        'residual': True, 'use_mask_channel': True,
        'use_coord_channels': True, 'mask_features': True,
    })


def test_swin_gru_odd_shapes_mask_and_gradients():
    assert 'swin_gru' in available_models()
    torch.manual_seed(0)
    model = _small_model()
    x = torch.randn(2, 5, 2, 17, 19)
    mask = torch.ones(1, 1, 17, 19)
    mask[..., :3] = 0
    opt = torch.optim.AdamW(model.parameters(), lr=0.005)
    y = torch.randn(2, 3, 2, 17, 19)
    for step in range(2):
        pred = model(x, mask)
        assert pred.shape == (2, 3, 2, 17, 19)
        assert torch.allclose(pred[..., :3], torch.zeros_like(pred[..., :3]))
        loss = ((pred-y)*mask[:,None]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # After the zero-initialized output head learns a step, backprop reaches
    # temporal attention and spatial shifted-window attention.
    assert model.temporal.q.weight.grad.abs().sum().item() > 0
    assert model.swin[0].qkv.weight.grad.abs().sum().item() > 0
    assert model.memory.gates.weight.grad.abs().sum().item() > 0
    assert torch.isfinite(pred).all()


def test_joint_reader_and_pipeline():
    from Forecast_nn import train as trainmod, test as testmod
    from Forecast_nn.runtime import load_checkpoint

    with tempfile.TemporaryDirectory() as temp:
        fixture(temp)
        flux = np.load(Path(temp)/'flux.npy')
        other = (0.6*flux+1.5).astype('float32')
        np.save(Path(temp)/'latent.npy', other)
        cfg.data_files['latent'] = Path(temp)/'latent.npy'
        cfg.model_params['swin_gru'].update(base_channels=8, temporal_tokens=2,
                                             attention_depth=2, heads=4, window_size=4, dropout=0)
        source=[str(cfg.data_files['sensible']),str(cfg.data_files['latent'])]
        r=make_field_reader(source, 8)
        assert r.channels == 2 and r.length == 75
        z=r.read_indices([0,1]); assert z.shape == (2,2,21,23)
        assert np.isclose(z[0,1,0,0],other[0,0,0])

        sys.argv=['train','--model','swin_gru','--variable','joint',
                  '--end-date',cfg.data_last_date,'--epochs','1','--batch-size','8','--grad-accum','3','--grad-clip','1',
                  '--num-workers','0','--downsample','8','--patience','2','--no-amp']
        trainmod.main()
        outputs=list((Path(temp)/'results').rglob('best.pth'))
        assert len(outputs)==1
        ck=load_checkpoint(outputs[0])
        assert ck['variable']=='joint' and ck['best_epoch']==1
        assert len(ck['mean'])==2 and len(ck['psnr_ranges'])==2
        assert isinstance(ck['data_file'],list)
        assert ck['data_end_idx']==75
        sys.argv=['test','--model','swin_gru','--variable','joint',
                  '--batch-size','8','--num-workers','0','--downsample','8','--no-amp',
                  '--plot-samples','0']
        testmod.main()
        files=list((Path(temp)/'results').rglob('metrics_swin_gru_joint.json'))
        assert len(files)==1
        report=json.loads(files[0].read_text(encoding='utf-8'))
        assert 'psnr_db' in report['per_horizon'][0]
        assert len(report['per_channel']) == 2
        assert files[0].with_name(report['per_sample_csv']).is_file()
