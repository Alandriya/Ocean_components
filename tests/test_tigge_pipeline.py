"""Offline tests for ECDS request composition and accumulated-flux decoding."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import numpy as np

from Forecast_1d.download_tigge_ecds import (
    EXPECTED_FIELDS, FLUX_HOURS, filename_for, make_request, validate_grib2, run,
)
from Data_processing.tigge_fluxes import cumulative_to_daily_mean, convert_flux_grib


class TestDownloadAndMath(unittest.TestCase):
    """Regression tests covering Test Download And Math."""
    def test_requests_and_file_naming(self):
        """Verify that requests and file naming."""
        d = date(2026, 10, 1)
        met = make_request(d, (80, -90, 0, 0), "0.25/0.25", "met")
        flux = make_request(d, (80, -90, 0, 0), "0.25/0.25", "fluxes")
        self.assertEqual(len(met["variable"]), 5)
        self.assertEqual(len(met["leadtime_hour"]), 61)
        self.assertEqual(EXPECTED_FIELDS["met"], 305)
        self.assertEqual(len(flux["variable"]), 2)
        self.assertEqual(flux["leadtime_hour"], [str(x) for x in range(24, 361, 24)])
        self.assertEqual(EXPECTED_FIELDS["fluxes"], 30)
        self.assertEqual(filename_for(d, Path("root"), "met").name, "tigge_ecmwf_control_20261001_0000.grib")
        self.assertEqual(filename_for(d, Path("root"), "fluxes").name, "tigge_ecmwf_control_20261001_0000_fluxes.grib")

    def test_cumulative_mean_shifts_and_sign(self):
        """Verify that cumulative mean shifts and sign."""
        hours = np.arange(24, 361, 24, dtype=np.float64)
        acc = np.column_stack(((-10 * 3600 * hours), (-100 * 3600 * hours)))
        mean0 = cumulative_to_daily_mean(acc, 0)
        mean1 = cumulative_to_daily_mean(acc, 1)
        self.assertEqual(mean0.shape, (14, 2))
        np.testing.assert_allclose(mean0, [[-10, -100]] * 14, rtol=1e-6)
        np.testing.assert_allclose(mean1, [[-10, -100]] * 14, rtol=1e-6)
        # Nonstationary cumulative flux distinguishes the two UTC-day conventions.
        curved = np.column_stack((hours ** 2, -hours ** 2))
        self.assertAlmostEqual(float(cumulative_to_daily_mean(curved, 0)[0, 0]), 24**2 / 86400)
        self.assertAlmostEqual(float(cumulative_to_daily_mean(curved, 1)[0, 0]), (48**2-24**2)/86400)
        with self.assertRaises(ValueError):
            cumulative_to_daily_mean(acc[:-1])

    def test_resume_keeps_existing_met_and_fetches_only_fluxes(self):
        """Verify that resume keeps existing met and fetches only fluxes."""
        class MockClient:
            """Container for mock client state and associated operations."""
            def __init__(self):
                """Initialize Mock Client with default state."""
                self.requests = []
            def retrieve(self, dataset, request, target):
                """Compute retrieve for Mock Client from ``dataset``, ``request``, ``target``."""
                self.requests.append(request)
                Path(target).write_bytes(record() * 30)
        def record():
            """Compute record for test resume keeps existing met and fetches only fluxes."""
            size = 64
            return b"GRIB" + b"\0\0\0\2" + size.to_bytes(8, "big") + b"\0"*(size-20) + b"7777"
        with TemporaryDirectory() as t:
            root = Path(t)
            issued = date(2026, 10, 1)
            metpath = filename_for(issued, root, "met")
            metpath.parent.mkdir(parents=True)
            metpath.write_bytes(record()*305)
            client = MockClient()
            with patch('Forecast_1d.download_tigge_ecds.make_client', return_value=client):
                code = run(['--start', '2026-10-01', '--end', '2026-10-01',
                            '--out', str(root), '--execute'])
            self.assertEqual(code, 0)
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(client.requests[0]['variable'], list(make_request(issued, (80,-90,0,0), '0.25/0.25', 'fluxes')['variable']))
            self.assertEqual(validate_grib2(filename_for(issued, root, 'met'), 305)[0], 305)
            self.assertEqual(validate_grib2(filename_for(issued, root, 'fluxes'), 30)[0], 30)

    def test_grib2_framing(self):
        """Verify that grib2 framing."""
        with TemporaryDirectory() as t:
            p = Path(t) / "dummy.grib"
            def rec():
                """Compute rec for test TIGGE pipeline."""
                size = 64
                return b"GRIB" + b"\0\0\0\2" + size.to_bytes(8, "big") + b"\0" * (size-20) + b"7777"
            p.write_bytes(rec()*30)
            self.assertEqual(validate_grib2(p, expected_fields=30)[0], 30)
            with self.assertRaises(ValueError):
                validate_grib2(p, expected_fields=305)
            p.write_bytes(rec()*29 + rec()[:-1])
            with self.assertRaises(ValueError):
                validate_grib2(p, expected_fields=30)


class FakeEcCodes(types.ModuleType):
    """PyTorch module implementing the Fake Ec Codes architecture or layer."""
    def __init__(self):
        """Initialize Fake Ec Codes with default state."""
        super().__init__("eccodes")
        self._lat = np.repeat(np.linspace(80,0,321), 361)
        self._lon = np.tile(np.linspace(270,360,361),321)
        self.bad_param = False
        self.bad_step = False

    def codes_grib_new_from_file(self, stream):
        """Compute codes GRIB new from file for Fake Ec Codes from ``stream``."""
        pos = stream.read(1)
        if not pos:
            return None
        return pos[0]

    def codes_release(self, msg):
        """Compute codes release for Fake Ec Codes from ``msg``."""
        return None

    def codes_get(self, msg, key):
        """Compute codes get for Fake Ec Codes from ``msg``, ``key``."""
        field = "sensible" if msg % 2 == 0 else "latent"
        hour = 24 * (msg//2 + 1)
        ref = {"edition": 2, "typeOfProcessedData": 3, "centre": "ecmf",
               "paramId": 146 if field=="sensible" else 147,
               "shortName": "sshf" if field=="sensible" else "slhf",
               "typeOfLevel": "surface", "stepType": "accum", "startStep": 0,
               "endStep": hour, "units": "W m**-2 s", "dataDate": 20261001,
               "dataTime": 0, "Ni": 361, "Nj": 321,
               "gridType": "regular_ll",
               "latitudeOfFirstGridPointInDegrees": 80.,
               "longitudeOfFirstGridPointInDegrees": 270.,
               "latitudeOfLastGridPointInDegrees": 0.,
               "longitudeOfLastGridPointInDegrees": 360.,
               }
        from datetime import datetime, timedelta
        valid = datetime(2026,10,1) + timedelta(hours=hour)
        ref["validityDate"] = int(valid.strftime("%Y%m%d"))
        ref["validityTime"] = int(valid.strftime("%H%M"))
        if self.bad_param:
            ref["paramId"] = 999
        if self.bad_step:
            ref["startStep"] = 24
        return ref[key]

    def codes_get_array(self, msg, key):
        """Compute codes get array for Fake Ec Codes from ``msg``, ``key``."""
        if key == "latitudes":
            return self._lat
        if key == "longitudes":
            return self._lon
        raise KeyError(key)

    def codes_get_values(self, msg):
        """Compute codes get values for Fake Ec Codes from ``msg``."""
        hour = 24 * (msg//2 + 1)
        q = -10. if msg % 2 == 0 else -100.
        return np.full((321*361,), q * hour * 3600., dtype=np.float64)


class TestFakeEcCodes(unittest.TestCase):
    """Regression tests covering Test Fake Ec Codes."""
    def setUp(self):
        """Prepare the test fixtures before running the assertions."""
        self.fake = FakeEcCodes()

    def test_end_to_end(self):
        """Verify that end to end."""
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fn = root / "tigge_ecmwf_control_20261001_0000_fluxes.grib"
            fn.write_bytes(bytes(range(30)))
            points = [{"id": f"P{i+1}", "lat_deg": 34., "lon_deg": -70.,
                       "grid_y": 92, "grid_x": 40} for i in range(15)]
            pp = root / "points.json"
            pp.write_text(json.dumps({"points": points}), encoding="utf-8")
            with patch.dict(sys.modules, {"eccodes": self.fake}):
                for offset in (0, 1):
                    p, meta, grid = convert_flux_grib(fn, pp, root / "out", day_offset=offset, full_grid=True)
                    self.assertTrue(p.exists() and meta.exists() and grid.exists())
                    x = np.load(p)
                    self.assertEqual(x["flux_daily_mean"].shape, (14, 15, 2))
                    np.testing.assert_allclose(x["flux_daily_mean"], np.broadcast_to([-10., -100.], (14,15,2)), rtol=1e-6)
                    self.assertEqual(str(x["target_dates"][0]), "2026-10-01" if offset==0 else "2026-10-02")
                    self.assertEqual(x["interval_hours"][0].tolist(), [0,24] if offset==0 else [24,48])
                    y = np.load(grid)
                    self.assertEqual(y["flux_daily_mean"].shape,(14,161,181,2))
                    np.testing.assert_allclose(y["flux_daily_mean"][:,92,40,:],
                                               x["flux_daily_mean"][:,0,:])
            self.fake.bad_step = True
            with patch.dict(sys.modules, {"eccodes": self.fake}):
                with self.assertRaisesRegex(ValueError, "start at"):
                    convert_flux_grib(fn, pp, root / "out3")


if __name__ == "__main__":
    unittest.main()
