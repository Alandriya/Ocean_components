"""Offline, dependency-free tests: python -m unittest discover -s tests -p test_download_gefs_reforecast.py"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

SCRIPT = Path(__file__).resolve().parents[1] / "Forecast_1d" / "download_gefs_reforecast.py"
spec = importlib.util.spec_from_file_location("gefs_pilot", SCRIPT)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def fake_grib(length=70, filler=b"\x00"):
    # Fully valid message envelope; no GRIB2 field decoding required for this test.
    return b"GRIB" + b"\x00\x00\x00\x02" + length.to_bytes(8, "big") + filler * (length - 20) + b"7777"


class Response:
    def __init__(self, status, payload, headers=None):
        self.status = status
        self.payload = payload
        self.pointer = 0
        self.headers = headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return None
    def read(self, size=-1):
        if size < 0:
            size = len(self.payload)
        segment = self.payload[self.pointer:self.pointer + size]
        self.pointer += len(segment)
        return segment


class Tests(unittest.TestCase):
    def test_days(self):
        self.assertEqual(g.parse_days("1,3-4,14"), [1, 3, 4, 14])
        with self.assertRaises(ValueError): g.parse_days("0")
        with self.assertRaises(ValueError): g.parse_days("15")
        with self.assertRaises(ValueError): g.parse_days("10-8")

    def test_hours(self):
        self.assertEqual(g.forecast_hours(1), (3, 6, 9, 12, 15, 18, 21, 24))
        self.assertEqual(g.forecast_hours(10), tuple(range(219, 241, 3)))
        self.assertEqual(g.forecast_hours(11), (246, 252, 258, 264))
        self.assertEqual(g.forecast_hours(14), (318, 324, 330, 336))

    def test_urls(self):
        x, idx = g.grib_urls(g.date(2017, 3, 14), "shtfl_sfc", "c00", "Days:1-10")
        self.assertIn("2017/2017031400/c00/Days%3A1-10/shtfl_sfc_2017031400_c00.grib2", x)
        self.assertEqual(idx, x + ".idx")

    def test_index_parse_and_select(self):
        a, b, c = fake_grib(70), fake_grib(65), fake_grib(80)
        idx = "\n".join([
            "1:0:d=2017031400:SHTFL:surface:0-3 hour ave fcst:ENS=low-res ctl",
            "2:70:d=2017031400:SHTFL:surface:3-6 hour ave fcst:ENS=low-res ctl",
            "3:135:d=2017031400:SHTFL:surface:6-9 hour ave fcst:ENS=low-res ctl",
        ])
        rows = g.parse_idx(idx, len(a+b+c))
        self.assertEqual([(r['start'],r['end'],r['size'],r['lead_h']) for r in rows],
                         [(0,69,70,3),(70,134,65,6),(135,214,80,9)])
        # Set expected hours to what is present: selecting day 1 must fail because +12..24 missing.
        with self.assertRaises(ValueError):
            g.selection_for_days(rows, [1], "shtfl_sfc", "Days:1-10")
        self.assertEqual(g.selection_for_days(rows, [11], "shtfl_sfc", "Days:1-10"), [])

    def test_download_exact_range(self):
        msg = fake_grib(70)
        def access(req, timeout):
            self.assertEqual(req.headers['Range'], 'bytes=0-69')
            return Response(206, msg, {'Content-Range':'bytes 0-69/10000'})
        from io import BytesIO
        with patch.object(g, 'urlopen', side_effect=access):
            dst = BytesIO()
            self.assertEqual(g.append_range(dst, 'https://noaa.test/test', 0, 69), 70)
            self.assertEqual(dst.getvalue(), msg)

    def test_download_wrong_http_status(self):
        from io import BytesIO
        with patch.object(g, 'urlopen', return_value=Response(200, b'ABC')):
            with self.assertRaisesRegex(RuntimeError, 'ignored Range'):
                g.append_range(BytesIO(), 'https://noaa.test/test', 0, 10)

    def test_atomic_download_and_skip(self):
        block = fake_grib(70)
        spec = {'url':'https://noaa.test/shtfl.grib2', 'idx_url':'https://noaa.test/shtfl.grib2.idx',
                'variable':'shtfl_sfc','segment':'Days:1-10',
                'rows':[{'lead_h':3,'start':0,'end':69,'raw':'1:0:d=2017031400:SHTFL:surface:0-3 hour ave fcst'}],
                'download_bytes':70}
        def fake_append(stream, url, start, end):
            stream.write(block)
            return len(block)
        with tempfile.TemporaryDirectory() as d:
            with patch.object(g, 'append_range', side_effect=fake_append) as m:
                outputs = g.run_download([spec], Path(d), g.date(2017,3,14), 'c00')
                self.assertEqual(len(outputs), 1)
                self.assertTrue(outputs[0].exists())
                self.assertEqual(g.validate_concatenated_grib2(outputs[0], 1), (1,70))
                g.run_download([spec], Path(d), g.date(2017,3,14), 'c00')
                self.assertEqual(m.call_count, 1)
                self.assertEqual(json.loads(outputs[0].with_suffix('.json').read_text())['hours'], [3])

    def test_missing_end(self):
        with self.assertRaises(ValueError):
            g.parse_idx('1:0:d=2017031400:SHTFL:surface:3 hour ave fcst\n',0)

    def test_decode_idx_text(self):
        with patch.object(g,'urlopen',return_value=Response(200,b'1:0:d=2017031400:TMP:2 m above ground:3 hour fcst')):
            self.assertIn('3 hour fcst', g.request_text('https://test/foo'))


if __name__ == '__main__':
    unittest.main()
