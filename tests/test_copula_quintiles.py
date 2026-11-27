"""Regression tests for copula quintiles."""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from Functional.copula import histogram_classes, bernstein_classes, elliptical_classes, estimate_joint_quantile_classes, _trapz

class JointClassesTest(unittest.TestCase):
    """Regression tests covering Joint Classes."""
    def test_independence(self):
        """Check uniform joint quintile probabilities under independence."""
        for p in (histogram_classes(np.ones((7,7))/49,5),
                  bernstein_classes(np.ones((7,7))/49,5),
                  elliptical_classes(0,5)):
            np.testing.assert_allclose(p, .04, atol=1e-9)

    def test_asymmetry_and_orientation(self):
        """Check matrix orientation, marginal masses and positive association."""
        p = np.roll(np.eye(5)/5, 1, axis=1)
        np.testing.assert_allclose(histogram_classes(p,5), p, atol=1e-12)
        for x in (bernstein_classes(p,5), elliptical_classes(.666,5,df=7.662)):
            np.testing.assert_allclose(x.sum(0), .2, atol=2e-8)
            np.testing.assert_allclose(x.sum(1), .2, atol=2e-8)
        self.assertGreater(elliptical_classes(.666,5)[0,0], .04)
        self.assertGreater(elliptical_classes(.666,5,df=7.662)[0,0], .04)

    def test_cached_workflow(self):
        """Generate joint quintile summaries from a synthetic cached copula fit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = root/'Functional'/'Copulas'; src.mkdir(parents=True)
            p = np.roll(np.eye(5)/5,1,axis=1)
            centers = (np.arange(5)+.5)/5
            x = np.linspace(-3,3,101); g = np.exp(-x*x/2); g = g/_trapz(g,x)
            joint = np.outer(g,g)
            np.savez_compressed(src/'sensible-latent_1979-2026.npz',
                empirical=p.T*25, centers=centers, checkerboard=p.T*25,
                checkerboard_centers=centers, x1=x,x2=x,
                joint_gaussian=joint, joint_checkerboard=joint,joint_bernstein=joint)
            (src/'sensible-latent_1979-2026.json').write_text(json.dumps(dict(
                pair=['sensible','latent'], period='1979-2026', gaussian_rho=.666,
                student_t_rho=.666, student_t_df=7.662, moments={})))
            target = estimate_joint_quantile_classes(root,1979,2026)
            res = json.loads((target/'summary.json').read_text())
            np.testing.assert_allclose(res['matrices']['checkerboard'],p,atol=1e-12)
            self.assertAlmostEqual(res['marginal_audit']['joint']['bernstein']['sensible']['mean'],0)
            self.assertTrue((target/'joint_quintiles_results.tex').exists())
            self.assertTrue((target/'matrices.npz').exists())

if __name__ == '__main__': unittest.main()
