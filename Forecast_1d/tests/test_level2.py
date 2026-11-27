"""Level-2 integration and causal baseline regression tests."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from Forecast_1d import train, test
from Forecast_1d.registry import LEVEL2, NEW_BASELINES
from Forecast_1d.models.neural import build_model, default_config
from Forecast_1d.models.baselines import calendar_index, fit_seasonal, forecast_seasonal
from Forecast_1d.models.ridge_var import fit_ridge, forecast_ridge


class Level2Tests(unittest.TestCase):
    """Regression tests covering Level2."""
    def setUp(self):
        """Prepare the test fixtures before running the assertions."""
        torch.set_num_threads(1)

    def test_audit_dispatch_preserves_arguments(self):
        """Dispatch each audit to its own parser without altering its CLI options."""
        from unittest.mock import patch
        from Forecast_1d import audit
        for mode, entry in (
            ("future-weather", "main_future_weather"),
            ("oracle-drivers", "main_oracle_drivers"),
            ("predictability", "main_predictability"),
            ("sst-vs-atmosphere", "main_sst_vs_atmosphere"),
            ("training", "main_training"),
        ):
            with self.subTest(mode=mode), patch.object(audit, entry) as handler:
                audit.main([mode, "--output", "report.json"])
                handler.assert_called_once_with(["--output", "report.json"])

    def test_shapes_backward(self):
        """Check output shape and finite gradients for every level-2 model."""
        for name in LEVEL2:
            with self.subTest(model=name):
                model=build_model(name,60,14)
                x=torch.randn(3,60,2)
                y=model(x)
                self.assertEqual(tuple(y.shape),(3,14,2))
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_channel_independence(self):
        """Ensure changes to one target channel cannot influence the other for channel-independent models."""
        x=torch.randn(2,60,2)
        altered=x.clone(); altered[:,:,1]+=100
        for name in ('patchtst','dlinear'):
            model=build_model(name,60,14).eval()
            torch.testing.assert_close(model(x)[:,:,0], model(altered)[:,:,0])

    def test_calendar(self):
        """Check calendar alignment around leap day and year-end."""
        dates=np.array(['2000-02-28','2000-02-29','2000-03-01','2001-02-28','2001-03-01','2000-12-31'],dtype='datetime64[D]')
        np.testing.assert_array_equal(calendar_index(dates),[58,59,60,58,60,365])

    def test_new_baseline_future_invariance(self):
        """Ensure seasonal and ridge baselines do not use observations after the forecast origin."""
        rng=np.random.default_rng(1)
        values=rng.normal(size=(1000,4,2))
        dates=np.datetime64('2000-01-01')+np.arange(len(values)).astype('timedelta64[D]')
        train_end=800; origins=np.array([850]); modified=values.copy(); modified[851:]=1e9
        clim,phi=fit_seasonal(values,dates,train_end)
        c2,p2=fit_seasonal(modified,dates,train_end)
        np.testing.assert_array_equal(clim,c2); np.testing.assert_array_equal(phi,p2)
        for coef in (None,phi):
            np.testing.assert_array_equal(forecast_seasonal(values,dates,origins,14,clim,coef),forecast_seasonal(modified,dates,origins,14,c2,coef))
        fits=fit_ridge(values,train_end,60,14); altered_fits=fit_ridge(modified,train_end,60,14)
        np.testing.assert_array_equal(forecast_ridge(values,origins,60,14,fits),forecast_ridge(modified,origins,60,14,altered_fits))

    def test_all_models_incremental_evaluation_and_stdlib_collection(self):
        """Exercise training, incremental evaluation, checkpoint preservation and standard-library result collection."""
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); source=base/'data'; source.mkdir(); results=base/'results'
            rng=np.random.default_rng(33)
            vals=rng.normal(size=(450,10,2)).astype(np.float32)
            np.save(source/'centers.npy',vals)
            np.save(source/'dates.npy', np.datetime64('2000-01-01')+np.arange(len(vals)).astype('timedelta64[D]'))
            (source/'points.json').write_text(json.dumps({'points':[{'id':f'P{i}','region':f'Test {i}'} for i in range(1,11)]}))
            common=['--dataset-dir',str(source),'--results-dir',str(results),'--in-len','60','--out-len','14','--val-days','35','--test-days','35']
            train.main(common+['--model','sarima'])
            common += ['--points','P3','P5','P7','P9']
            for model in ('mlp','tcn','gru',*LEVEL2):
                train.main(common+['--model',model,'--epochs','1','--batch-size','32','--hidden-size','8','--device','cpu'])
            for split in ('val','test'):
                dest=test.main(common+['--split',split,'--model','level01','--device','cpu'])
                prior=json.loads((dest/'metrics.json').read_text())['models']
                old_checkpoint=(results/'h14'/'gru'/'P3'/'best.pt').read_bytes()
                test.main(common+['--split',split,'--model','level0_new','--device','cpu'])
                test.main(common+['--split',split,'--model','level2','--device','cpu'])
                metrics=json.loads((dest/'metrics.json').read_text())
                self.assertEqual(len(metrics['models']),14)
                for name in ('sarima','mlp','tcn','gru'):
                    self.assertEqual(metrics['models'][name],prior[name])
                self.assertEqual((results/'h14'/'gru'/'P3'/'best.pt').read_bytes(),old_checkpoint)
                for name in (*NEW_BASELINES,*LEVEL2):
                    with np.load(dest/f'{name}_predictions.npz') as file:
                        self.assertEqual(file['prediction'].shape,(22,4,2,14))
                        self.assertTrue(np.isfinite(file['prediction']).all())
                subprocess.run([sys.executable,'-S','-m','Forecast_1d.collect_results','--results-dir',str(results),'--split',split,'--points','P3','P5','P7','P9'],check=True,capture_output=True)
                report=json.loads((results/'h14'/'summary'/f'forecast_1d_{split}_P3_P5_P7_P9_h14.json').read_text())
                self.assertEqual(len(report['ranking_by_rmse']),14)
                self.assertFalse(report['missing_models'])
            from Forecast_1d.data import PointDataset
            dataset=PointDataset(source)
            splits=dataset.get_splits(35,35,60,14)
            origins=splits.origins('test')[:1]
            args=test.build_parser().parse_args(common+['--device','cpu'])
            fp=dataset.fingerprint()
            def predict(name):
                """Predict the requested test level2 data using the supplied inputs."""
                return test._forecast_neural(name,'P3',2,dataset,origins,results/'h14',splits,args,torch.device('cpu'))
            original={name:predict(name) for name in LEVEL2}
            dataset.values=vals.copy()
            dataset.values[origins[0]+1:]=1e8
            dataset.fingerprint=lambda:fp
            for name in LEVEL2:
                np.testing.assert_array_equal(original[name],predict(name))
            from Forecast_1d.plotting.plot_results import main as plot
            outputs=plot(['--dataset-dir',str(source),'--results-dir',str(results),'--model','nhits','--points','P3','P5','P7','P9','--point','P3','--horizon','14','--compare','seasonal_ar1','ridge_var','--dpi','50'])
            self.assertEqual(len(outputs),1)
            self.assertTrue(outputs[0].is_file())


if __name__=='__main__':
    unittest.main()
