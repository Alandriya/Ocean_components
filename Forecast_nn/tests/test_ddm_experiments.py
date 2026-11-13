"""Synthetic end-to-end tests; no real ocean arrays and no GPU needed."""
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import torch

from Forecast_nn.config import cfg
from Forecast_nn.models.simvp import SimVPForecaster
from Forecast_nn.models.ddm import DDMSimVP, EOFProjector, LatentSDERNN
from Forecast_nn.joint_sde import covariance_from_b, JointSDEReader
from Forecast_nn.train_ddm import arguments as train_args, run as train
from Forecast_nn.test_ddm import arguments as eval_args, run as evaluate


class TestDDM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory()
        d=Path(cls.temp.name)
        cls.root=d
        cls.orig={k:getattr(cfg,k) for k in ('root_path','mask_file','data_files','results_root',
                   'stats_root','in_len','out_len','stride','downsample','data_last_date',
                   'val_steps','test_steps','base_date','fast_cuda','psnr_max_sample_frames')}
        count=96;ds=8;h,w=21,23
        rng=np.random.default_rng(11)
        a=rng.standard_normal((count,h,w)).astype(np.float32)*0.1
        b=rng.standard_normal((count,h,w)).astype(np.float32)*0.2
        a=np.cumsum(a,axis=0)+np.arange(count)[:,None,None]*0.002
        b=np.cumsum(b,axis=0)+0.25*a
        np.save(d/'sensible.npy',a);np.save(d/'latent.npy',b)
        mask=np.ones((161,181),dtype=np.bool_)
        mask[::7,::5]=False
        mask.tofile(d/'mask')
        drift=np.zeros((count-1,2,h,w),dtype=np.float32)
        drift[:,0]=0.002
        drift[:,1]=0.0
        diff=np.zeros((count-1,4,h,w),dtype=np.float32)
        diff[:,0]=0.2;diff[:,1]=0.05;diff[:,2]=0.02;diff[:,3]=0.3
        np.save(d/'a.npy',drift);np.save(d/'b.npy',diff)
        cfg.root_path=d
        cfg.mask_file=d/'mask'
        cfg.data_files={'sensible':d/'sensible.npy','latent':d/'latent.npy'}
        cfg.results_root=d/'Forecast'/'Results'
        cfg.stats_root=d/'Stats'
        cfg.in_len=4;cfg.out_len=2;cfg.stride=1;cfg.downsample=8
        cfg.base_date='1979-01-01'
        cfg.data_last_date=(date(1979,1,1)+timedelta(days=count-1)).isoformat()
        cfg.val_steps=12;cfg.test_steps=12
        cfg.fast_cuda=False;cfg.psnr_max_sample_frames=16
        cls.sde=[str(d/'a.npy'),str(d/'b.npy')]

    @classmethod
    def tearDownClass(cls):
        for k,v in cls.orig.items():setattr(cfg,k,v)
        cls.temp.cleanup()

    def test_simvp_compat_and_auxariance(self):
        params=dict(base_channels=4,depth=2,dropout=0.05,residual=True,
                    use_mask_channel=True,use_coord_channels=True,mask_features=True,
                    temporal_blocks=1,temporal_kernel=3,spatial_kernel=3,horizon_embedding=True)
        model=DDMSimVP(4,2,params)
        x=torch.randn(2,4,2,21,23)
        m=torch.ones(21,23,dtype=torch.bool)
        pred,ahat,d=model(x,m)
        self.assertEqual(tuple(pred.shape),(2,2,2,21,23))
        self.assertEqual(tuple(ahat.shape),(2,2,21,23))
        self.assertEqual(tuple(d.shape),(2,2,2,21,23))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(d.permute(0,3,4,1,2))>0))
        original=SimVPForecaster(4,2,in_channels=2,**params)
        self.assertIsInstance(original(x,m),torch.Tensor)
        self.assertEqual(tuple(original(x,m).shape),tuple(pred.shape))

    def test_coefficient_alignment(self):
        reader=JointSDEReader(*self.sde,downsample=8,hw=(21,23),state_std=np.array([1.,1.]))
        a,b=reader.get_normalized([1,2])
        self.assertEqual(tuple(a.shape),(2,2,21,23))
        self.assertEqual(tuple(b.shape),(2,2,2,21,23))
        self.assertEqual(float(a[0,0,0,0]),0.0020000000949949026)
        # Important: native repository packing is [B11, B22, B12, B21].
        np.testing.assert_allclose(b[:, 0, 0].numpy(), 0.2, rtol=0, atol=1e-6)
        np.testing.assert_allclose(b[:, 1, 1].numpy(), 0.05, rtol=0, atol=1e-6)
        np.testing.assert_allclose(b[:, 0, 1].numpy(), 0.02, rtol=0, atol=1e-6)
        np.testing.assert_allclose(b[:, 1, 0].numpy(), 0.3, rtol=0, atol=1e-6)
        d=covariance_from_b(b)
        self.assertTrue((d[:,0,1]!=0).any())
        # B @ B.T (not entrywise B squared): D12 = B11*B21 + B12*B22.
        self.assertAlmostEqual(float(d[0, 0, 1, 0, 0]), 0.2*0.3 + 0.02*0.05, places=5)
        with self.assertRaises(IndexError):reader.get_normalized([0])

    def test_matrix_layouts(self):
        # Natural matrix dimensions and flattened matrix layouts must agree.
        d=self.root
        original=np.load(d/'b.npy')
        t,c,h,w=original.shape
        full=np.empty((t,2,2,h,w),dtype=original.dtype)
        full[:,0,0]=original[:,0]  # B11
        full[:,1,1]=original[:,1]  # B22
        full[:,0,1]=original[:,2]  # B12
        full[:,1,0]=original[:,3]  # B21
        shapes={'tmmhw':full, 'thwmm':full.transpose(0,3,4,1,2),
                'stmm':full.transpose(3,4,0,1,2).reshape(h*w,t,2,2),
                'tsmm':full.transpose(0,3,4,1,2).reshape(t,h*w,2,2),
                'tcs':original.reshape(t,4,h*w)}
        for label,mat in shapes.items():
            file=d/f'b_{label}.npy';np.save(file,mat)
            rd=JointSDEReader(self.sde[0],str(file),8,(h,w),[1.,1.])
            _,b=rd.get_normalized([1,2])
            np.testing.assert_allclose(b.numpy(),full[:2])

    def test_legacy_row_major_layout(self):
        """Other datasets' row-major packed channels remain explicitly supported."""
        original=np.load(self.root/'b.npy')
        t,_,h,w=original.shape
        natural=np.stack((original[:,0],original[:,2],original[:,3],original[:,1]),axis=1)
        file=self.root/'b_row_major.npy';np.save(file,natural)
        rd=JointSDEReader(self.sde[0],str(file),8,(h,w),[1.,1.],b_layout='row_major')
        _,b=rd.get_normalized([1])
        self.assertAlmostEqual(float(b[0,0,1,0,0]),0.02,places=5)
        self.assertAlmostEqual(float(b[0,1,0,0,0]),0.3,places=5)

    def test_train_and_test_both(self):
        for model in ('simvp_ddm','latent_sde_rnn'):
            args=train_args(['--model',model,'--variable','joint',
                '--epochs','1','--batch-size','2','--num-workers','0','--no-amp',
                '--patience','2','--max-train-batches','2','--max-val-batches','2',
                '--joint-a-file',self.sde[0],'--joint-b-file',self.sde[1],
                '--base-channels','4','--depth','2','--temporal-blocks','1',
                '--rank','3','--hidden','12','--pca-snapshots','20',
                '--lambda-a','0.05','--lambda-d','0.01', '--run-id','smoke'])
            path=train(args)
            self.assertTrue(path.is_file())
            record=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(record['metadata']['data_end_idx'],96)
            self.assertEqual(record['metadata']['test_start_idx'],84)
            self.assertEqual(record['metadata']['val_start_idx'],72)
            self.assertEqual(record['metadata']['sde_time_offset'],1)
            self.assertEqual(record['metadata']['joint_b_layout'],'diag_first')
            if model=='latent_sde_rnn':
                proj=EOFProjector.from_state_dict(record['projector_state']).to_torch('cpu')
                zz=proj.encode(torch.randn(2,4,2,21,23))
                self.assertEqual(tuple(zz.shape),(2,4,3))
                pred=proj.decode(zz[:,-1])
                self.assertEqual(tuple(pred.shape),(2,2,21,23))
            argv=['--model',model,'--variable','joint','--split','val','--checkpoint',str(path),
                  '--batch-size','2','--num-workers','0','--members','3','--max-batches','2']
            result=evaluate(eval_args(argv))
            self.assertGreater(result['samples_evaluated'],0)
            self.assertIn('overall_rmse',result['summary'])
            self.assertIn('mse_skill_vs_persistence',result['summary'])
            self.assertTrue((path.parent/'val'/f'metrics_{model}_joint.json').exists())
            # A fresh TEST request must not be used to choose best; evaluated only on demand.
            with self.assertRaises(ValueError):
                evaluate(eval_args(argv+['--end-date','1979-03-31']))


if __name__=='__main__':unittest.main()
