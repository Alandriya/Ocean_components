"""Synthetic regression tests for consolidated period output and stable maps."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_DATE = date(1979, 1, 1)


def start_label(year):
    return (date(year, 1, 1) - BASE_DATE).days + 1


def setup_modules():
    # Import the implementation without relying on the full, original repo.
    io = types.ModuleType("Data_processing.io")
    io.BASE_DATE = BASE_DATE
    io.start_label = start_label
    io.load_field = lambda *_: None
    io.read_ocean_mask = lambda root, shape=(2,2): np.ones(shape, dtype=bool)
    sys.modules["Data_processing.io"] = io
    paths = types.ModuleType("Plotting.paths")
    def plot_path(root, *parts):
        path = Path(root) / "Plots" / Path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    paths.plot_path = plot_path
    paths.save_figure = lambda fig, path, **kwargs: (fig.savefig(path, **kwargs), path)[1]
    sys.modules["Plotting.paths"] = paths
    return io


class BridgeAndMapsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.io = setup_modules()
        from Coefficients import bridges, collection
        from Plotting import map_workflows
        cls.bridges, cls.collection, cls.maps = bridges, collection, map_workflows

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.periods = [(2023, 2024), (2024, 2025), (2025, 2026)]
        self.lengths = [365, 366, 367]
        self.fields = {}
        for period, count in zip(self.periods, self.lengths):
            first = start_label(period[0])
            for name in ("sensible", "latent"):
                self.fields[name, tuple(period)] = np.arange(count*4, dtype=float).reshape(count,2,2)
            interval = range(first+1, first+count)
            for method in ("np", "sp"):
                folder = self.collection.period_dir(self.root, *period) / "2d_bins_12" / "sensible_latent" / self.bridges.METHOD_FOLDERS[method]
                folder.mkdir(parents=True, exist_ok=True)
                for field in (("A", "B") if method=="np" else ("A", "B", "B_with_reg")):
                    nch = 2 if field=="A" else 4
                    data = np.stack([np.full((nch,2,2), label+(0.1 if method == "sp" else 0), dtype="float32") for label in interval])
                    np.save(folder / f"{field}_{interval.start}-{interval.stop-1}.npy", data)
        self.load = lambda root, name, first, last: self.fields[name, (first,last)]

    def _make_bridges(self):
        for period in self.periods[1:]:
            index = start_label(period[0])
            for method in ("np", "sp"):
                directory = self.bridges.bridge_daily_dir(self.root, ("sensible", "latent"), 12, method)
                directory.mkdir(parents=True, exist_ok=True)
                for field in (("A", "B") if method=="np" else ("A", "B", "B_with_reg")):
                    nch = 2 if field=="A" else 4
                    np.save(directory / f"{field}_{index}.npy", np.full((nch,2,2), index+.1 if method=="sp" else index, dtype="float32"))

    def test_assembly_from_collected_periods_no_daily(self):
        self._make_bridges()
        with patch.object(self.bridges,"load_field",side_effect=self.load):
            out = self.bridges.assemble_continuous_coefficients(self.root,self.periods)
        self.assertEqual(set(out["methods"]),{"np","sp"})
        self.assertEqual(out["count"],sum(self.lengths)-1)
        self.assertEqual(out["last_transition_date"],"2026-01-02")
        self.assertTrue((self.root / "Coefficients" / "2023-2026" / "sensible_latent" / "A_2023-2026_np.npy").is_file())
        self.assertTrue((self.root / "Coefficients" / "2023-2026" / "sensible_latent" / "B_with_reg_2023-2026_sp.npy").is_file())
        array = np.load(out["methods"]["np"]["A"],mmap_mode="r")
        self.assertEqual(array.shape,(sum(self.lengths)-1,2,2,2))
        first_bridge = self.lengths[0]-1
        self.assertEqual(array[0,0,0,0],start_label(2023)+1)
        self.assertEqual(array[first_bridge,0,0,0],start_label(2024))
        self.assertEqual(array[first_bridge+1,0,0,0],start_label(2024)+1)
        self.assertEqual(array[-1,0,0,0],out["last_label"])
        with patch.object(self.bridges,"load_field",side_effect=self.load):
            reused = self.bridges.assemble_continuous_coefficients(self.root,self.periods)
        self.assertEqual(reused["count"],out["count"])

    def test_exact_1979_to_2026_naming_and_actual_last_day(self):
        periods = [(1979,1989),(1989,1999),(1999,2009),
                   (2009,2019),(2019,2026)]
        final = date(2026,10,29)
        sizes = {}
        for period in periods:
            first,last = period
            last_date = (final if period==periods[-1] else date(last,1,1)-timedelta(days=1))
            size=(last_date-date(first,1,1)).days+1
            sizes[period] = size
            labels=np.arange(start_label(first)+1,start_label(first)+size,dtype="float32")
            folder=self.collection.period_dir(self.root,*period) / "2d_bins_12" / "sensible_latent" / "Nonparametric_2d"
            folder.mkdir(parents=True,exist_ok=True)
            for field,nch in (("A",2),("B",4)):
                array=np.broadcast_to(labels[:,None,None,None],(size-1,nch,1,1)).copy()
                np.save(folder/f"{field}_{int(labels[0])}-{int(labels[-1])}.npy",array)
        for period in periods[1:]:
            label=start_label(period[0])
            folder=self.bridges.bridge_daily_dir(self.root,("sensible","latent"),12,"np")
            folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/f"A_{label}.npy",np.full((2,1,1),label,dtype="float32"))
            np.save(folder/f"B_{label}.npy",np.full((4,1,1),label,dtype="float32"))
        fake=lambda root,name,first,last: np.broadcast_to(
            0.,(sizes[(first,last)],1,1))
        with patch.object(self.bridges,"load_field",side_effect=fake):
            out=self.bridges.assemble_continuous_coefficients(
                self.root,periods,methods=("np",))
        expected=self.root/"Coefficients"/"1979-2026"/"sensible_latent"/"A_1979-2026_np.npy"
        self.assertEqual(out["last_transition_date"],final.isoformat())
        self.assertEqual(Path(out["methods"]["np"]["A"]),expected)
        data=np.load(expected,mmap_mode="r")
        self.assertEqual(len(data),(final-BASE_DATE).days)
        self.assertEqual(float(data[0,0,0,0]),2.)
        self.assertEqual(float(data[-1,0,0,0]),start_label(2026)+(final-date(2026,1,1)).days)

    def test_missing_bridge_is_detected(self):
        self._make_bridges()
        missing = self.bridges.bridge_daily_dir(self.root, ("sensible","latent"),12,"np") / f"A_{start_label(2024)}.npy"
        missing.unlink()
        with patch.object(self.bridges,"load_field",side_effect=self.load):
            with self.assertRaisesRegex(FileNotFoundError,"Missing cross-period bridge"):
                self.bridges.assemble_continuous_coefficients(self.root,self.periods)

    def test_incomplete_middle_period_rejected(self):
        self._make_bridges()
        self.fields["sensible",(2024,2025)] = self.fields["sensible",(2024,2025)][:-1]
        self.fields["latent",(2024,2025)] = self.fields["latent",(2024,2025)][:-1]
        with patch.object(self.bridges,"load_field",side_effect=self.load):
            with self.assertRaisesRegex(ValueError,"Gap before bridge"):
                self.bridges.assemble_continuous_coefficients(self.root,self.periods)

    def test_bridge_only_new_transition_calculated(self):
        self._make_bridges()
        directory = self.bridges.bridge_daily_dir(self.root, ("sensible", "latent"),12,"np")
        (directory / f"A_{start_label(2024)}.npy").unlink()
        common = types.ModuleType("Coefficients.common")
        common.prepare_pair_transition = lambda *_: object()
        npmod = types.ModuleType("Coefficients.nonparametric")
        npmod.estimate_np_2d = lambda grouped,**kwargs: (np.ones((2,2,2)),np.ones((4,2,2)))
        spmod = types.ModuleType("Coefficients.semiparametric")
        spmod.estimate_sp_2d = lambda grouped,**kwargs: (np.ones((2,2,2)),np.ones((4,2,2)),np.ones((4,2,2)),{})
        with patch.dict(sys.modules,{"Coefficients.common":common,"Coefficients.nonparametric":npmod,"Coefficients.semiparametric":spmod}), patch.object(self.bridges,"load_field",side_effect=self.load):
            self.bridges.calculate_boundary_transition(self.root,(2023,2024),(2024,2025),mask=np.ones((2,2),bool),methods=("np",))
        self.assertTrue((directory / f"A_{start_label(2024)}.npy").exists())
        self.assertEqual(np.load(directory/f"A_{start_label(2024)}.npy").shape,(2,2,2))

    def test_signed_zero_white_and_mask_lightgreen(self):
        cmap, norm = self.maps._historical_palette("A",-30,20)
        self.assertAlmostEqual(norm(0),.5)
        self.assertTrue(np.allclose(cmap(norm(0))[:3],(1,1,1)))
        from matplotlib.colors import to_rgba
        self.assertTrue(np.allclose(cmap.get_bad(),to_rgba("lightgreen")))
        for name in ("sensible","latent","Flux","B","C","F","FS","SST","Pressure"):
            cm, _ = self.maps._historical_palette(name,-5,100)
            self.assertTrue(np.allclose(cm.get_bad(),to_rgba("lightgreen")))

    def test_stable_palette_over_frame_interval(self):
        data = np.array([[[1.,2.],[3.,4.]], [[50.,20.],[10.,-7.]], [[9.,4.],[2.,1.]]])
        water = np.array([[True,False],[True,True]])
        mn,mx=self.maps._extrema(data,0,3,water)
        self.assertEqual((mn,mx),(-7,50))
        cmap,norm=self.maps._historical_palette("Flux",mn,mx)
        self.assertEqual(norm(0),.5)
        # A single frame's smaller extrema do not alter this normalization.
        self.assertEqual(norm(2),self.maps._historical_palette("Flux",mn,mx)[1](2))

    def test_full_boundary_workflow_produces_both_methods(self):
        common = types.ModuleType("Coefficients.common")
        common.prepare_pair_transition = lambda *_: object()
        npmod = types.ModuleType("Coefficients.nonparametric")
        npmod.estimate_np_2d = lambda grouped,**kwargs: (
            np.ones((2,2,2),dtype="float32"),
            np.ones((4,2,2),dtype="float32"))
        spmod = types.ModuleType("Coefficients.semiparametric")
        spmod.estimate_sp_2d = lambda grouped,**kwargs: (
            np.ones((2,2,2),dtype="float32"),
            np.ones((4,2,2),dtype="float32"),
            np.ones((4,2,2),dtype="float32"),{})
        with patch.dict(sys.modules, {
            "Coefficients.common":common,
            "Coefficients.nonparametric":npmod,
            "Coefficients.semiparametric":spmod,
        }),patch.object(self.bridges,"load_field",side_effect=self.load):
            output = self.bridges.run_boundary_workflow(
                self.root,self.periods,("sensible","latent"),
                mask=np.ones((2,2),dtype=bool), methods=("np","sp"))
        self.assertEqual(set(output["methods"]), {"np","sp"})
        for year in (2024,2025):
            label=start_label(year)
            for method in ("np","sp"):
                self.assertTrue((self.bridges.bridge_daily_dir(self.root,
                     ("sensible","latent"),12,method) / f"A_{label}.npy").is_file())

    def test_variables_draw_multiple_frames_at_fixed_scale(self):
        series = {
            "sensible":np.array([[[0.,0.],[1.,-2.]],[[0.,0.],[10.,-20.]],[[0.,0.],[3.,-4.]]]),
            "latent":np.array([[[0.,0.],[3.,-2.]],[[0.,0.],[100.,-20.]],[[0.,0.],[3.,-4.]]]),
        }
        scale_ids=[]
        original_draw=self.maps._draw_map
        def draw(ax,field,water,label,palette):
            scale_ids.append((label,id(palette[1]), palette[1](0)))
            return original_draw(ax,field,water,label,palette)
        with patch.object(self.maps,"load_field",side_effect=lambda root,name,a,b:series[name]),patch.object(self.maps,"_draw_map",side_effect=draw):
            paths=self.maps.plot_variables(
                self.root,2023,2024,start=0,stop=3,
                groups=("Sensible-Latent",))
        self.assertEqual(len(paths),3)
        self.assertTrue(all(path.is_file() for path in paths))
        sensible=[item for item in scale_ids if item[0]=="Явный тепловой поток"]
        latent=[item for item in scale_ids if item[0]=="Скрытый тепловой поток"]
        self.assertEqual(len({item[1] for item in sensible}),1)
        self.assertEqual(len({item[1] for item in latent}),1)
        self.assertTrue(all(item[2]==.5 for item in scale_ids))

    def test_coefficient_frames_from_flat_merged_file(self):
        folder=self.root / "Coefficients" / "2023-2026" / "sensible_latent"
        folder.mkdir(parents=True,exist_ok=True)
        stack=np.array([
            [[[2.,np.nan],[0.,-3.]],[[1.,2.],[0.,-1.]]],
            [[[20.,np.nan],[0.,-30.]],[[10.,20.],[0.,-10.]]],
        ])
        np.save(folder/"A_2023-2026_np.npy",stack)
        frames=self.maps.coefficient_frames(self.root,("sensible","latent"),12,"np","A",start=0,stop=2)
        self.assertEqual(len(frames),2)
        self.assertTrue(all(path.is_file() for path in frames))

    def test_old_f_color_boundaries(self):
        cmap,norm=self.maps._historical_palette("F",0,3)
        self.assertEqual(len(norm.boundaries),6)
        self.assertEqual(list(norm.boundaries[:5]),[0.,.25,.5,1.,1.5])
        self.assertEqual(cmap.N,5)


if __name__ == "__main__":
    unittest.main()
