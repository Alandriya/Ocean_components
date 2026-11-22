"""Synthetic end-to-end tests for leakage-aware SAR correction forecasting."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Forecast_1d import train, test, collect_results
from Forecast_1d.data import PointDataset
from Forecast_1d.experiment import target_root
from Forecast_1d.sar_residual import (
    fold_edges, fitted_sar, sar_prediction, zero_output_head,
    prepare_residual_training,
)


class SarResidualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        self.results = self.root / 'results'
        n = 260
        day = np.arange(n)
        rng = np.random.default_rng(420)
        values = np.zeros((n, 15, 2), dtype=np.float32)
        for p in range(15):
            for c in range(2):
                seasonal = (10+p)*np.sin(2*np.pi*day/365.25 + 0.5*c)
                anomaly = np.cumsum(rng.normal(0,1.0+c,n))*0.17
                values[:,p,c] = (10+p*3+c*40+seasonal+anomaly).astype(np.float32)
        dates = np.datetime64('2001-01-01','D') + day.astype('timedelta64[D]')
        np.save(self.dataset / 'centers.npy', values)
        np.save(self.dataset / 'dates.npy', dates)
        points = Path(__file__).resolve().parents[1]/'points.json'
        (self.dataset/'points.json').write_bytes(points.read_bytes())
        self.data = PointDataset(self.dataset)
        self.train_end = n - 40 - 40

    def training_args(self, model, extra=None):
        return train.build_parser().parse_args([
            '--model',model,'--target-mode','sar_residual',
            '--dataset-dir',str(self.dataset),'--results-dir',str(self.results),
            '--points','P3','--in-len','20','--out-len','14',
            '--val-days','40','--test-days','40','--epochs','2','--patience','2',
            '--scheduler','none','--batch-size','32','--device','cpu',
            '--monitor','sar',
            *(extra or []),
        ])

    def test_oof_fit_is_past_only_and_ignores_val_test(self):
        values = np.asarray(self.data.values[:,2,:]).copy()
        dates = self.data.dates
        edges = fold_edges(self.train_end,20,14,4)
        self.assertEqual(edges[-1],self.train_end)
        self.assertLess(edges[0],edges[1])
        self.assertGreater(edges[0], 20 + 14)
        climate1, phi1 = fitted_sar(values, dates, edges[0], 15)
        shifted = values.copy()
        shifted[edges[0]:] += 9999.0
        climate2, phi2 = fitted_sar(shifted, dates, edges[0], 15)
        np.testing.assert_array_equal(climate1, climate2)
        np.testing.assert_array_equal(phi1, phi2)
        for start, stop in zip(edges[:-1],edges[1:]):
            orig = np.arange(start-1, stop-14)
            self.assertTrue(np.all(orig + 1 >= start))
            self.assertTrue(np.all(orig + 14 < stop))
            self.assertTrue(start > np.max(np.arange(0,start-1)))

        origins = self.data.get_splits(40,40,20,14).origins('val')
        prepared1 = prepare_residual_training(values,dates,self.train_end,origins,20,14)
        modified = values.copy()
        modified[self.train_end:] += 3000
        prepared2 = prepare_residual_training(modified,dates,self.train_end,origins,20,14)
        np.testing.assert_array_equal(prepared1.x_train,prepared2.x_train)
        np.testing.assert_array_equal(prepared1.y_train,prepared2.y_train)
        np.testing.assert_array_equal(prepared1.residual_scale,prepared2.residual_scale)
        np.testing.assert_array_equal(prepared1.fold_edges,prepared2.fold_edges)
        self.assertEqual(prepared1.y_train.shape[1:],(14,2))
        self.assertTrue(np.isfinite(prepared1.y_train).all())
        self.assertTrue(np.isfinite(prepared1.y_val).all())

    def test_zero_head_starts_at_exact_sar_and_epoch_zero_is_candidate(self):
        import torch
        from Forecast_1d.models.neural import build_model, default_config
        for name in ('mlp','tcn','gru'):
            net = build_model(name,20,14,default_config(name,hidden_size=8))
            zero_output_head(net,name)
            net.eval()
            with torch.no_grad():
                y = net(torch.randn(5,20,2))
            self.assertEqual(tuple(y.shape),(5,14,2))
            self.assertTrue(bool(torch.all(y==0)))

        for name in ('mlp','tcn','gru'):
            train.run(self.training_args(name))
            checkpoint = self.results/'h14'/'sar_residual'/name/'P3'
            meta=json.loads((checkpoint/'train_metadata.json').read_text())
            self.assertEqual(meta['target_mode'],'sar_residual')
            self.assertEqual(meta['selection_metric'],'sar')
            self.assertEqual(meta['sar_residual']['reference'],'seasonal_ar1')
            self.assertEqual(meta['sar_residual']['oof_folds'],4)
            self.assertEqual(meta['sar_residual']['baseline_epoch_zero_candidate'],True)
            self.assertGreater(meta['sar_residual']['oof_train_origin_count'],10)
            self.assertGreaterEqual(meta['best_validation_sar_skill'],-1e-8)
            self.assertIn(meta['best_epoch'],range(0,3))

    def test_reconstruct_base_and_integrate_with_baseline(self):
        import torch
        train.run(self.training_args('mlp'))
        checkpoint = self.results/'h14'/'sar_residual'/'mlp'/'P3'/'best.pt'
        stored = torch.load(checkpoint,map_location='cpu',weights_only=True)
        stored['state_dict']={k:torch.zeros_like(v) for k,v in stored['state_dict'].items()}
        torch.save(stored,checkpoint)

        args = test.build_parser().parse_args([
            '--model','level0_new','--target-mode','sar_residual',
            '--dataset-dir',str(self.dataset),'--results-dir',str(self.results),
            '--points','P3','--in-len','20','--out-len','14',
            '--val-days','40','--test-days','40','--split','test','--device','cpu',
        ])
        dest=test.run(args)
        with np.load(dest/'seasonal_ar1_predictions.npz') as z:
            sar=z['prediction'].copy()
            origins=z['origins'].copy()
        args.model='mlp'
        test.run(args)
        with np.load(dest/'mlp_predictions.npz') as z:
            prediction=z['prediction']
            np.testing.assert_allclose(prediction,sar,atol=2e-5,rtol=1e-6)
            self.assertEqual(prediction.shape,(27,1,2,14))
            self.assertTrue(np.array_equal(z['origins'],origins))
        report=json.loads((dest/'metrics.json').read_text())
        self.assertEqual(report['target_mode'],'sar_residual')
        self.assertIn('mlp',report['models'])
        self.assertIn('seasonal_ar1',report['models'])

        collect=collect_results.build_parser().parse_args([
            '--target-mode','sar_residual','--points','P3',
            '--results-dir',str(self.results),'--split','test'
        ])
        collect_results.collect(collect)
        ranking=(self.results/'h14'/'sar_residual'/'summary'/'forecast_1d_test_P3_h14_ranking.csv')
        self.assertTrue(ranking.is_file())
        self.assertIn('mlp',ranking.read_text())

    def test_zero_lr_retains_validation_baseline_checkpoint(self):
        # No gradient update can change a zero-head model: the epoch-0 SAR
        # candidate must remain the selected checkpoint, never epoch 1.
        train.run(self.training_args('mlp', [
            '--learning-rate','0', '--epochs','1'
        ]))
        path=self.results/'h14'/'sar_residual'/'mlp'/'P3'/'train_metadata.json'
        metadata=json.loads(path.read_text())
        self.assertEqual(metadata['best_epoch'],0)
        self.assertAlmostEqual(metadata['best_validation_sar_skill'],0.0,places=7)
        self.assertAlmostEqual(metadata['best_selection_loss'],1.0,places=7)

    def test_checkpoint_radius_and_dataset_mismatch_blocked(self):
        train.run(self.training_args('gru'))
        args = test.build_parser().parse_args([
            '--model','gru','--target-mode','sar_residual',
            '--dataset-dir',str(self.dataset),'--results-dir',str(self.results),
            '--points','P3','--in-len','20','--out-len','14',
            '--val-days','40','--test-days','40','--split','val','--device','cpu',
            '--climatology-radius','13',
        ])
        with self.assertRaisesRegex(ValueError,'checkpoint configuration mismatch'):
            test.run(args)
        args.climatology_radius=15
        # Explicit metadata fingerprint tampering is detected independently.
        meta_path=self.results/'h14'/'sar_residual'/'gru'/'P3'/'train_metadata.json'
        meta=json.loads(meta_path.read_text())
        meta['dataset_fingerprint_sha256']='wrong'
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'Checkpoint mismatch'):
            test.run(args)

    def test_default_all_fifteen_points_end_to_end(self):
        args = train.build_parser().parse_args([
            '--model','mlp','--target-mode','sar_residual','--monitor','sar',
            '--dataset-dir',str(self.dataset),'--results-dir',str(self.results),
            '--in-len','20','--out-len','14',
            '--val-days','40','--test-days','40',
            '--epochs','1','--patience','1','--scheduler','none',
            '--batch-size','64','--hidden-size','8','--device','cpu',
        ])
        train.run(args)
        checkpoint_root=self.results/'h14'/'sar_residual'/'mlp'
        for i in range(1,16):
            self.assertTrue((checkpoint_root/f'P{i}'/'best.pt').is_file())
        evaluate = test.build_parser().parse_args([
            '--model','mlp','--target-mode','sar_residual',
            '--dataset-dir',str(self.dataset),'--results-dir',str(self.results),
            '--in-len','20','--out-len','14',
            '--val-days','40','--test-days','40',
            '--split','val','--device','cpu',
        ])
        dest = test.run(evaluate)
        with np.load(dest/'mlp_predictions.npz') as z:
            self.assertEqual(z['prediction'].shape,(27,15,2,14))
        report=json.loads((dest/'metrics.json').read_text())
        self.assertEqual(len(report['point_ids']),15)
        collect=collect_results.build_parser().parse_args([
            '--target-mode','sar_residual','--results-dir',str(self.results),
            '--split','val'
        ])
        collect_results.collect(collect)
        summary=self.results/'h14'/'sar_residual'/'summary'
        self.assertTrue(any(summary.glob('*_ranking.csv')))

    def test_isolated_paths_and_folds_validation(self):
        base = Path('/tmp/atlantic')
        self.assertEqual(target_root(base,14,'raw'), base/'h14')
        self.assertEqual(target_root(base,14,'anomaly'), base/'h14'/'anomaly')
        self.assertEqual(target_root(base,14,'sar_residual'),base/'h14'/'sar_residual')
        with self.assertRaises(ValueError):
            fold_edges(140,20,14,100)


if __name__=='__main__':
    unittest.main()
