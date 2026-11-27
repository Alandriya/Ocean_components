"""Offline regression tests for the combined one-request TIGGE downloader."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import MagicMock

from Forecast_1d import download_tigge_ecds as dl
from Data_processing.tigge_fluxes import cumulative_to_daily_mean
import numpy as np


class FakeJobs:
    def __init__(self, ids=()):
        self.request_ids = list(ids)
        self.next = None


class FakeRemote:
    def __init__(self, request_id, status="accepted"):
        self.request_id = request_id
        self.status = status
        self.results_ready = False


class FakeClient:
    def __init__(self, active=0):
        self.active = active
        self.submits = []
        self.cancelled = []
        self.requests = {}

    def get_jobs(self, **kwargs):
        return FakeJobs([f"existing-{i}" for i in range(self.active)])

    def submit(self, dataset, request):
        self.submits.append((dataset, request))
        rid = f"new-{len(self.submits)}"
        self.requests[rid] = FakeRemote(rid)
        return self.requests[rid]

    def get_remote(self, rid):
        return self.requests[rid]

    def delete(self, rid):
        self.cancelled.append(rid)


class CombinedRequestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.today = date(2026, 10, 1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_request_one_date_seven_variables(self):
        r = dl.make_request(self.today, (80, -90, 0, 0), "0.25/0.25")
        self.assertEqual(len(r["variable"]), 7)
        self.assertEqual(len(set(r["variable"])), 7)
        self.assertIn("surface_sensible_heat_flux", r["variable"])
        self.assertIn("surface_latent_heat_flux", r["variable"])
        self.assertEqual(len(r["leadtime_hour"]), 61)
        self.assertEqual(r["leadtime_hour"][0], "0")
        self.assertEqual(r["leadtime_hour"][-1], "360")
        self.assertEqual(dl.filename_for(self.today, self.root).name,
                         "tigge_ecmwf_control_20261001_0000_all.grib")

    def test_one_server_job_per_day(self):
        c = FakeClient()
        rc = dl._submit_async(c, self.root, [self.today], (80,-90,0,0), "0.25/0.25", 2, 1)
        self.assertEqual(rc, 0)
        self.assertEqual(len(c.submits), 1)
        data = json.loads((self.root / "download_jobs.json").read_text())
        self.assertEqual(set(data), {"2026-10-01:combined"})
        self.assertEqual(len(data["2026-10-01:combined"]["request"]["variable"]), 7)
        rc = dl._submit_async(c, self.root, [self.today], (80,-90,0,0), "0.25/0.25", 2, 1)
        self.assertEqual(rc, 0)
        self.assertEqual(len(c.submits), 1)

    def test_limit_respects_external_active_job(self):
        c = FakeClient(active=1)
        rc = dl._submit_async(c, self.root, [self.today], (80,-90,0,0), "0.25/0.25", 1, 1)
        self.assertEqual(rc, 0)
        self.assertEqual(c.submits, [])

    def test_cancellation_with_known_id(self):
        c = FakeClient()
        c.requests["old-id"] = FakeRemote("old-id", "running")
        self.assertEqual(dl._cancel_async(c, self.root, "old-id"), 0)
        self.assertEqual(c.cancelled, ["old-id"])

    def test_flux_daily_mean_preserves_sign(self):
        e = np.arange(1, 16, dtype=float)[:,None,None]*86400.0 * -12.0
        pred = cumulative_to_daily_mean(e, 0)
        self.assertEqual(pred.shape, (14, 1, 1))
        self.assertTrue(np.all(pred == -12.0))


class HttpMissing(Exception):
    status_code = 404


class StatusAwareClient(FakeClient):
    def __init__(self, active=0, *, missing=(), errors=None):
        super().__init__(active=active)
        self.missing = set(missing)
        self.errors = dict(errors or {})

    def get_remote(self, rid):
        if rid in self.errors:
            raise self.errors[rid]
        if rid in self.missing:
            raise HttpMissing("server request removed")
        return super().get_remote(rid)


class CancelledLegacyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.today = date(2025, 10, 31)
        self.settings = ((80,-90,0,0), "0.25/0.25", 3, 3)

    def tearDown(self):
        self.tmp.cleanup()

    def _old_jobs(self):
        return {
            f"{self.today}:met": {"request_id": "old-met", "status": "accepted"},
            f"{self.today}:fluxes": {"request_id": "old-flux", "status": "running"},
        }

    def test_cancelled_legacy_404_releases_date_and_keeps_audit(self):
        dl._write_registry(self.root, self._old_jobs())
        client = StatusAwareClient(missing={"old-met", "old-flux"})
        rc = dl._submit_async(client, self.root, [self.today], *self.settings)
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.submits), 1)
        registry = dl._read_registry(self.root)
        self.assertEqual(registry[f"{self.today}:met"]["status"], "deleted")
        self.assertEqual(registry[f"{self.today}:fluxes"]["status"], "deleted")
        self.assertEqual(len(registry[f"{self.today}:combined"]["request"]["variable"]), 7)

    def test_cancelled_legacy_explicit_status_releases_date(self):
        dl._write_registry(self.root, self._old_jobs())
        client = StatusAwareClient()
        client.requests["old-met"] = FakeRemote("old-met", "cancelled")
        client.requests["old-flux"] = FakeRemote("old-flux", "dismissed")
        rc = dl._submit_async(client, self.root, [self.today], *self.settings)
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.submits), 1)

    def test_legacy_successful_blocks_combined_job(self):
        dl._write_registry(self.root, self._old_jobs())
        client = StatusAwareClient(missing={"old-flux"})
        client.requests["old-met"] = FakeRemote("old-met", "successful")
        rc = dl._submit_async(client, self.root, [self.today], *self.settings)
        self.assertEqual(rc, 0)
        self.assertEqual(client.submits, [])

    def test_network_failure_does_not_assume_cancellation(self):
        dl._write_registry(self.root, self._old_jobs())
        client = StatusAwareClient(errors={"old-met": ConnectionError("offline")})
        rc = dl._submit_async(client, self.root, [self.today], *self.settings)
        self.assertEqual(rc, 1)
        self.assertEqual(client.submits, [])

    def test_combined_cancelled_can_be_resubmitted_and_old_id_retained(self):
        dl._write_registry(self.root, {
            f"{self.today}:combined": {"request_id": "old-combined", "status": "running"},
        })
        client = StatusAwareClient(missing={"old-combined"})
        rc = dl._submit_async(client, self.root, [self.today], *self.settings)
        self.assertEqual(rc, 0)
        self.assertEqual(len(client.submits), 1)
        registry = dl._read_registry(self.root)
        record = registry[f"{self.today}:combined"]
        self.assertNotEqual(record["request_id"], "old-combined")
        self.assertEqual(record["history"][0]["request_id"], "old-combined")
        self.assertEqual(record["history"][0]["status"], "deleted")

    def test_combined_pending_is_not_duplicated(self):
        dl._write_registry(self.root, {
            f"{self.today}:combined": {"request_id": "old-combined", "status": "accepted"},
        })
        client = StatusAwareClient()
        client.requests["old-combined"] = FakeRemote("old-combined", "running")
        rc = dl._submit_async(client, self.root, [self.today], *self.settings)
        self.assertEqual(rc, 0)
        self.assertEqual(client.submits, [])

    def test_404_detection_never_classifies_other_errors_as_cancellation(self):
        self.assertTrue(dl._is_http_404(HttpMissing()))
        self.assertFalse(dl._is_http_404(TimeoutError("404 in description")))
        self.assertFalse(dl._is_http_404(ConnectionError("offline")))

    def test_limit_blocks_new_requests_when_slots_busy(self):
        dl._write_registry(self.root, self._old_jobs())
        client = StatusAwareClient(active=3, missing={"old-met", "old-flux"})
        rc = dl._submit_async(client, self.root, [self.today], *self.settings)
        self.assertEqual(rc, 0)
        self.assertEqual(client.submits, [])


if __name__ == "__main__":
    unittest.main()
