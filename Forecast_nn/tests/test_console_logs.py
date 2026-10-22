"""Logs should be written in parallel to stdout/stderr without changing run IDs."""
import argparse
from pathlib import Path

import pytest

from Forecast_nn.logging_utils import log_file_path, run_tag_from_args, tee_output


def test_log_names(monkeypatch):
    from Forecast_nn.config import cfg
    monkeypatch.setattr(cfg, 'in_len', 30)
    monkeypatch.setattr(cfg, 'out_len', 3)
    args = argparse.Namespace(
        model='unet_convlstm', variable='sensible', downsample=2,
        base_channels=None, temporal_tokens=None, attention_depth=None,
        heads=None, window_size=None, sde_weight=None,
        sde_b_kind='diffusion', sde_loss_mode='first_step', run_id=''
    )
    run_tag = run_tag_from_args(args)
    assert run_tag == 'ds2_i30_o3_b16_d3'
    log_path = log_file_path('train', args.model, run_tag, args.variable)
    assert log_path.name == 'train_unet_convlstm_ds2_i30_o3_b16_d3.txt'
    assert log_path.parent.name == 'logs'
    assert log_file_path('test', args.model, run_tag).name == 'test_unet_convlstm_ds2_i30_o3_b16_d3.txt'
    assert log_file_path('train', args.model, run_tag, 'latent').name.endswith('_latent.txt')
    assert log_file_path('test', args.model, run_tag, 'joint', 'val').name.endswith('_joint_val.txt')
    args.run_id = 'try2'
    assert run_tag_from_args(args).endswith('_try2')


def test_stdout_stderr_append(tmp_path, capsys):
    path = tmp_path / 'logs' / 'train_example.txt'
    with tee_output(path):
        print('epoch 1, RMSE=1.23')
        import sys
        print('warning from model', file=sys.stderr)
    out = capsys.readouterr()
    assert 'epoch 1, RMSE=1.23' in out.out
    assert 'warning from model' in out.err
    text = path.read_text(encoding='utf-8')
    assert 'epoch 1, RMSE=1.23' in text
    assert 'warning from model' in text
    assert '[log] Finished:' in text
    assert '[log] Started:' in text
    assert '[log] Command:' in text
    assert '[log] File:' in text
    with tee_output(path):
        print('continued epoch 2')
    new_text = path.read_text(encoding='utf-8')
    assert 'epoch 1, RMSE=1.23' in new_text
    assert 'continued epoch 2' in new_text
    assert new_text.count('[log] Started:') == 2


def test_traceback_saved(tmp_path):
    path = tmp_path / 'logs' / 'test_error.txt'
    with pytest.raises(RuntimeError, match='synthetic error'):
        with tee_output(path):
            raise RuntimeError('synthetic error')
    output = path.read_text(encoding='utf-8')
    assert 'Traceback (most recent call last):' in output
    assert 'RuntimeError: synthetic error' in output
