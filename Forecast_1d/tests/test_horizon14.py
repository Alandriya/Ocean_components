"""Regression tests for the 14-day experiment and legacy artifact isolation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from Forecast_1d.data import PointDataset, split_dates
from Forecast_1d.experiment import horizon_root, DEFAULT_HORIZON
from Forecast_1d.models.baselines import persistence, input_mean
from Forecast_1d.models.sarima import fit_ar1, forecast_ar1
from Forecast_1d import train, test
from Forecast_1d.plotting import plot_results
from Forecast_1d.collect_results import build_parser as collect_parser


class Horizon14Tests(unittest.TestCase):
    """Regression tests covering Horizon14."""
    def test_defaults_and_daily_splits(self):
        """Verify that defaults and daily splits."""
        self.assertEqual(DEFAULT_HORIZON, 14)
        for parser in (train.build_parser(), test.build_parser(), collect_parser()):
            self.assertEqual(parser.parse_args([]).out_len, 14)
        self.assertEqual(plot_results.build_parser().parse_args(['--model', 'gru']).out_len, 14)
        split = split_dates(17470, in_len=60)
        self.assertEqual(split.train_end, 16375)
        self.assertEqual(len(split.origins('val')), 717)
        self.assertEqual(len(split.origins('test')), 352)
        for part in ('val', 'test'):
            origins = split.origins(part)
            start, end = split.slice_for(part)
            self.assertEqual(origins[0]+1, start)
            self.assertEqual(origins[-1]+14, end-1)

    def test_baseline_forecasts_do_not_use_future(self):
        """Verify that baseline forecasts do not use future."""
        rng = np.random.default_rng(11)
        data = rng.normal(size=(200, 10, 2))
        params = fit_ar1(data, 0, 100)
        origins = np.array([120])
        changed = data.copy()
        changed[121:] = 1e6
        for function in (lambda x: persistence(x, origins, 14),
                         lambda x: input_mean(x, origins, 14, 30),
                         lambda x: forecast_ar1(x, origins, 14, params)):
            np.testing.assert_array_equal(function(data), function(changed))
            self.assertEqual(function(data).shape, (1, 10, 2, 14))
        np.testing.assert_array_equal(params, fit_ar1(changed, 0, 100))

    def test_train_evaluate_plot_collect(self):
        """Verify that train evaluate plot collect."""
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / 'dataset'
            source.mkdir()
            results = folder / 'results'
            results.mkdir()
            old = results / 'gru' / 'P3' / 'best.pt'
            old.parent.mkdir(parents=True)
            old.write_bytes(b'legacy-checkpoint')
            rng = np.random.default_rng(22)
            values = rng.normal(size=(240,10,2)).astype(np.float32)
            np.save(source/'centers.npy', values)
            np.save(source/'dates.npy', np.datetime64('1979-01-01')+np.arange(240).astype('timedelta64[D]'))
            (source/'points.json').write_text(json.dumps({'points': [{'id':f'P{i}', 'region':f'Test {i}'} for i in range(1,11)]}))
            common = ['--dataset-dir',str(source),'--results-dir',str(results),'--in-len','60','--val-days','35','--test-days','35']
            train.main(common+['--model','sarima'])
            common += ['--points','P3','P5','P7','P9']
            for name in ('mlp','tcn','gru'):
                train.main(common+['--model',name,'--epochs','1','--batch-size','32','--hidden-size','8','--device','cpu'])
            root = horizon_root(results, 14)
            for name in ('mlp','tcn','gru'):
                for point in ('P3','P5','P7','P9'):
                    self.assertTrue((root/name/point/'best.pt').exists())
            self.assertEqual(old.read_bytes(),b'legacy-checkpoint')
            for part in ('val','test'):
                dest = test.main(common+['--model','level01','--split',part,'--device','cpu'])
                metrics = json.loads((dest/'metrics.json').read_text())
                self.assertEqual(len(metrics['models']),6)
                self.assertEqual(metrics['out_len'],14)
                for name in metrics['models']:
                    self.assertEqual(set(metrics['models'][name]),{str(i) for i in range(1,15)})
                    self.assertEqual(metrics['models'][name]['14']['joint']['n'],22*8)
                    with np.load(dest/f'{name}_predictions.npz',allow_pickle=False) as data:
                        self.assertEqual(data['prediction'].shape,(22,4,2,14))
                        np.testing.assert_array_equal(data['target'][...,13],values[data['origins']+14][:,[2,4,6,8],:])
                        self.assertTrue(np.isfinite(data['prediction']).all())
                # CPU collector must run even with site-packages disabled.
                subprocess.run([sys.executable,'-S','-m','Forecast_1d.collect_results','--results-dir',str(results),'--split',part,'--points','P3','P5','P7','P9'],check=True,capture_output=True,text=True)
                summary = root/'summary'/f'forecast_1d_{part}_P3_P5_P7_P9_h14.csv'
                with summary.open() as f:
                    rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),6*14*11)
                self.assertTrue(all(row['sarima_mse_skill'] != '' for row in rows))
            outputs = plot_results.main(['--dataset-dir',str(source),'--results-dir',str(results),'--model','gru','--points','P3','P5','P7','P9','--point','P3','--horizon','1','14','--show-baselines','--dpi','50'])
            self.assertEqual(len(outputs),2)
            self.assertTrue(all(p.is_file() for p in outputs))
            # Verify actual model inference against a future-only perturbation.
            data = PointDataset(source)
            split = data.get_splits(35,35,60,14)
            origins=split.origins('test')[:1]
            args=test.build_parser().parse_args(common+['--model','gru','--device','cpu'])
            fingerprint=data.fingerprint()
            predict=lambda model: test._forecast_neural(model,'P3',2,data,origins,root,split,args,torch.device('cpu'))
            original={model:predict(model) for model in ('mlp','tcn','gru')}
            data.values=values.copy()
            data.values[origins[0]+1:]=1e6
            # Bypass only the intentionally strict whole-dataset identity check.
            data.fingerprint=lambda:fingerprint
            for model in original:
                np.testing.assert_array_equal(original[model],predict(model))


if __name__=='__main__':
    unittest.main()
