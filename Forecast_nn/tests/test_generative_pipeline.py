"""Small, deterministic smoke tests requiring only installed PyTorch/NumPy."""
import tempfile
import sys
from unittest.mock import patch
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from Forecast_nn.config import cfg
from Forecast_nn.models.generative import ConditionalDDPM, ConditionalGAN
from Forecast_nn.models.eof_var import EOFVAR
from Forecast_nn.joint_sde import JointSDEReader, JointSDEDataset, covariance_from_b, joint_drift_loss
from Forecast_nn.data import OceanSequenceDataset
from Forecast_nn import train_generative, test_generative, train


class GenerativeSmoke(unittest.TestCase):
    def test_shapes_gradients_and_covariance(self):
        x=torch.randn(2,6,2,13,15)
        y=torch.randn(2,3,2,13,15)
        mask=torch.ones(13,15)
        ddpm=ConditionalDDPM(2,6,3,8,4)
        ddpm.training_loss(x,y,mask).backward()
        self.assertEqual(ddpm.sample(x,mask,members=2).shape,(2,2,3,2,13,15))
        gan=ConditionalGAN(2,6,3,8)
        out=gan.generate(x,mask)
        (gan.critic(x,out,mask).mean()+out.square().mean()).backward()
        b=torch.tensor([[[[[2.]],[[1.]]],[[[3.]],[[4.]]]]])
        d=covariance_from_b(b)
        self.assertTrue(torch.allclose(d.flatten(),torch.tensor([5.,10.,10.,25.])))
        self.assertTrue(torch.isfinite(joint_drift_loss(y,x[:,-1],torch.zeros_like(y[:,0]),
                                                      torch.ones(2,2,2,13,15),mask)))

    def test_full_small_pipeline(self):
        old={k:getattr(cfg,k) for k in ('base_date','data_last_date','in_len','out_len','stats_root','results_root','data_files','mask_file')}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            try:
                cfg.base_date='1979-01-01'
                cfg.data_last_date=str(date(1979,1,1)+timedelta(days=92))
                cfg.in_len=5;cfg.out_len=3
                cfg.stats_root=root/'stats';cfg.results_root=root/'Forecast'/'Results'
                cfg.mask_file=root/'mask'; np.ones((161,181),np.bool_).tofile(cfg.mask_file)
                rng=np.random.default_rng(1)
                files={}
                for name in ('sensible','latent'):
                    seq=rng.standard_normal((115,161,181)).astype('float32')
                    seq=np.cumsum(seq,axis=0)*.2
                    path=root/f'{name}.npy';np.save(path,seq);files[name]=path
                cfg.data_files=files
                common=dict(variable='joint',epochs=1,batch_size=2,num_workers=0,lr=.0001,
                    patience=1,width=8,diffusion_steps=4,downsample=8,end_date=cfg.data_last_date,
                    val_steps=12,test_steps=10,run_id='smoke',amp=False,resume=False,grad_clip=1,
                    max_train_batches=2,max_val_batches=1,var_rank=3,var_lags=2,var_ridge=1.,
                    var_snapshot_stride=6,sde_joint_a_file=None,sde_joint_b_file=None,
                    sde_b_layout='row_major',sde_moment_weight=0.,sde_cov_weight=.01,seed=17,
                    gan_recon_weight=10.,gan_adv_weight=.1)
                for name in ('ddpm','gan','var'):
                    args=SimpleNamespace(model=name,**common)
                    path=train_generative.run(args)
                    self.assertTrue(Path(path).exists())
                    ev=SimpleNamespace(**dict(vars(args),split='val',checkpoint=str(path),members=2,
                                             max_batches=1))
                    result=test_generative.run(ev)
                    self.assertTrue(np.isfinite(result['summary']['overall_rmse']))
                    self.assertTrue(np.isfinite(result['summary']['crps']))
                aa=np.zeros((110,2,21,23),np.float32)
                bb=np.zeros((110,4,21,23),np.float32);bb[:,0]=1;bb[:,3]=2
                np.save(root/'a.npy',aa);np.save(root/'b.npy',bb)
                jr=JointSDEReader(root/'a.npy',root/'b.npy',8,(21,23),[5,10])
                a,b=jr.get_normalized([1,50])
                self.assertEqual(tuple(a.shape),(2,2,21,23))
                self.assertEqual(tuple(b.shape),(2,2,2,21,23))
                # Exercise the optional ensemble-moment SDE loss inside GAN training.
                with_sde=dict(common, sde_moment_weight=1e-6, sde_joint_a_file=str(root/'a.npy'),
                              sde_joint_b_file=str(root/'b.npy'), run_id='sdegan',max_train_batches=1)
                sdegan=train_generative.run(SimpleNamespace(model='gan', **with_sde))
                self.assertTrue(Path(sdegan).exists())
                # Exercise the original SimVP training loop with joint loss-only SDE.
                orig_params=cfg.model_params['simvp']
                cfg.model_params['simvp']={**orig_params,'base_channels':4,'depth':2,'temporal_blocks':1}
                argv=['train','--model','simvp','--variable','joint','--epochs','1',
                    '--downsample','8','--batch-size','128','--num-workers','0',
                    '--end-date',cfg.data_last_date,'--val-steps','12','--test-steps','10',
                    '--sde-weight','0.0001','--joint-a-file',str(root/'a.npy'),
                    '--joint-b-file',str(root/'b.npy'),'--run-id','jointsmoke','--no-amp']
                try:
                    with patch.object(sys,'argv',argv):
                        train._run(train.parse_args())
                finally:
                    cfg.model_params['simvp']=orig_params
            finally:
                for k,v in old.items():setattr(cfg,k,v)

if __name__=='__main__':unittest.main()
