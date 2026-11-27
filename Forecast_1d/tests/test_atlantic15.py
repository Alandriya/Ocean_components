"""Regression coverage of 15-point defaults, isolation and SAR-selected checkpoints."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from Forecast_1d import train, test
from Forecast_1d.data import dataset_dir, results_dir, PointDataset
from Forecast_1d.experiment import DEFAULT_POINTS, DEFAULT_SUBDIR
from Forecast_1d.registry import NEURAL_MODELS, ALL_MODELS
from Forecast_1d.plotting import plot_results
from Forecast_1d.collect_results import build_parser


class Atlantic15Tests(unittest.TestCase):
    """Regression tests covering Atlantic15."""
    def test_default_layout(self):
        """Verify that default layout."""
        root = Path('/example/root')
        self.assertEqual(dataset_dir(root_override=root), root/'DATA'/'Forecast_1d'/DEFAULT_SUBDIR)
        self.assertEqual(results_dir(root_override=root), root/'Forecast'/'Results'/'Forecast_1d'/'atlantic15')
        self.assertEqual(tuple(build_parser().parse_args([]).points), DEFAULT_POINTS)
        self.assertIsNone(train.build_parser().parse_args([]).points)
        self.assertIsNone(test.build_parser().parse_args([]).points)
        self.assertIsNone(plot_results.build_parser().parse_args(['--model','gru']).points)

    def test_complete_pipeline(self):
        """Verify that complete pipeline."""
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); source=base/'DATA'/'Forecast_1d'/DEFAULT_SUBDIR; source.mkdir(parents=True)
            root=results_dir(root_override=base)
            legacy=root.parent/'h14'/'gru'/'P3'/'best.pt'; legacy.parent.mkdir(parents=True); legacy.write_bytes(b'legacy')
            rng=np.random.default_rng(121)
            values=rng.normal(size=(440,15,2)).astype(np.float32)
            values += np.sin(np.arange(len(values))[:,None,None]*2*np.pi/365).astype(np.float32)
            np.save(source/'centers.npy',values)
            dates=np.datetime64('2000-01-01')+np.arange(len(values)).astype('timedelta64[D]')
            np.save(source/'dates.npy',dates)
            points_file=Path(__file__).parents[1]/'points.json'
            (source/'points.json').write_bytes(points_file.read_bytes())
            self.assertEqual(PointDataset(source).point_ids, DEFAULT_POINTS)
            # Default root paths are unit-checked above; CLI receives explicit synthetic roots.
            common=['--dataset-dir',str(source),'--results-dir',str(root),'--val-days','35','--test-days','35','--in-len','60','--out-len','14']
            train.main(common+['--model','sarima'])
            for name in NEURAL_MODELS:
                train.main(common+['--model',name,'--epochs','1','--batch-size','64','--hidden-size','8','--device','cpu'])
            for name in NEURAL_MODELS:
                for point in DEFAULT_POINTS:
                    self.assertTrue((root/'h14'/name/point/'best.pt').is_file())
            # Exercise a SAR-selected checkpoint and actual LR reduction in a separate run.
            tune=base/'tune'; tune_common=[*common]; tune_common[tune_common.index('--results-dir')+1]=str(tune)
            train.main(tune_common+['--model','gru','--points','P15','--epochs','4','--patience','3','--scheduler','plateau','--lr-patience','0','--min-delta','10','--monitor','sar','--batch-size','64','--hidden-size','8','--device','cpu'])
            meta=json.loads((tune/'h14'/'gru'/'P15'/'train_metadata.json').read_text())
            hist=json.loads((tune/'h14'/'gru'/'P15'/'history.json').read_text())
            self.assertEqual(meta['best_epoch'],min(hist,key=lambda r:r['selection_loss'])['epoch'])
            self.assertTrue(any(r['next_learning_rate']<r['learning_rate'] for r in hist))
            dest=test.main(tune_common+['--model','gru','--points','P15','--split','val','--device','cpu'])
            test.main(tune_common+['--model','level0_new','--points','P15','--split','val','--device','cpu'])
            summary=json.loads((dest/'metrics.json').read_text())['models']
            ratios=[summary['gru'][str(h)]['points']['P15'][ch]['mse']/summary['seasonal_ar1'][str(h)]['points']['P15'][ch]['mse'] for h in range(1,15) for ch in ('sensible','latent')]
            self.assertAlmostEqual(meta['best_validation_sar_skill'],1-np.mean(ratios),places=6)
            for split in ('val','test'):
                dest=test.main(common+['--model','all','--split',split,'--device','cpu'])
                summary=json.loads((dest/'metrics.json').read_text())
                self.assertEqual(summary['point_ids'],list(DEFAULT_POINTS))
                self.assertEqual(set(summary['models']),set(ALL_MODELS))
                for name in ALL_MODELS:
                    with np.load(dest/f'{name}_predictions.npz',allow_pickle=False) as archive:
                        self.assertEqual(archive['prediction'].shape,(22,15,2,14))
                        np.testing.assert_array_equal(archive['target'][...,13],values[archive['origins']+14])
                subprocess.run([sys.executable,'-E','-S','-m','Forecast_1d.collect_results','--results-dir',str(root),'--split',split],check=True,capture_output=True)
                stem=f"forecast_1d_{split}_{'_'.join(DEFAULT_POINTS)}_h14"
                report=json.loads((root/'h14'/'summary'/f'{stem}.json').read_text())
                self.assertEqual(report['primary_reference'],'seasonal_ar1')
                self.assertFalse(report['missing_models'])
                self.assertEqual(len(report['rows']),14*14*33)
                self.assertTrue((root/'h14'/'summary'/f'{stem}_sar_ranking.csv').is_file())
            files=plot_results.main(['--dataset-dir',str(source),'--results-dir',str(root),'--model','gru','--point','P15','--horizon','14','--compare','seasonal_ar1','--dpi','50'])
            self.assertEqual(len(files),1)
            self.assertEqual(legacy.read_bytes(),b'legacy')


if __name__=='__main__':
    unittest.main()
