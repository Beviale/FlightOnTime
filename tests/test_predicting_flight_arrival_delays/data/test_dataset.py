"""Tests for predicting_flight_arrival_delays.data.dataset.

The BTS archive is never contacted: requests.get is replaced by a stub that
serves ZIP bytes built in memory, so the download, extraction and cleanup logic
can be exercised month by month.
"""

import io
import zipfile

import pytest
import requests
from typer.testing import CliRunner

from predicting_flight_arrival_delays.data import dataset as dataset_module
from predicting_flight_arrival_delays.data.dataset import app

runner = CliRunner()


def _zip_bytes(names=("flights.csv", "readme.html")) -> bytes:
    """A real ZIP archive, in memory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, "a,b\n1,2\n")
    return buffer.getvalue()


class _Response:
    def __init__(self, content=b"", status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    directory = tmp_path / "raw"
    monkeypatch.setattr(dataset_module, "RAW_DATA_DIR", directory)
    return directory


@pytest.fixture
def served(monkeypatch):
    """Serve a valid archive for every month, recording the URLs requested."""
    requested = []

    def fake_get(url, headers=None, timeout=None):
        requested.append(url)
        return _Response(_zip_bytes())

    monkeypatch.setattr(dataset_module.requests, "get", fake_get)
    return requested


class TestDownloadBtsData:
    def test_one_directory_per_month(self, raw_dir, served):
        result = runner.invoke(app, ["--year", "2025"])

        assert result.exit_code == 0, result.output
        assert sorted(p.name for p in raw_dir.iterdir()) == [
            f"2025_{month:02d}" for month in range(1, 13)
        ]

    def test_the_archive_contents_are_extracted(self, raw_dir, served):
        runner.invoke(app, ["--year", "2025"])

        assert (raw_dir / "2025_01" / "flights.csv").exists()

    def test_html_files_are_stripped(self, raw_dir, served):
        """BTS ships a readme alongside the data; only the CSV is wanted."""
        runner.invoke(app, ["--year", "2025"])

        assert not list((raw_dir / "2025_01").glob("*.html"))

    def test_the_zip_is_deleted_afterwards(self, raw_dir, served):
        runner.invoke(app, ["--year", "2025"])

        assert not list(raw_dir.glob("*.zip"))

    def test_months_already_present_are_skipped(self, raw_dir, served):
        """An interrupted download can be restarted with the same arguments."""
        runner.invoke(app, ["--year", "2025"])
        after_first = len(served)

        runner.invoke(app, ["--year", "2025"])

        assert len(served) == after_first

    def test_force_re_downloads_everything(self, raw_dir, served):
        runner.invoke(app, ["--year", "2025"])
        after_first = len(served)

        result = runner.invoke(app, ["--year", "2025", "--force"])

        assert result.exit_code == 0
        assert len(served) == after_first * 2

    def test_several_years_in_one_run(self, raw_dir, served):
        runner.invoke(app, ["--year", "2025", "--year", "2026"])

        assert len(list(raw_dir.iterdir())) == 24

    def test_a_month_not_yet_published_is_skipped(self, raw_dir, monkeypatch):
        """BTS answers 404 for months that do not exist yet: not an error."""

        def fake_get(url, headers=None, timeout=None):
            if url.endswith("2025_12.zip"):
                return _Response(status_code=404)
            return _Response(_zip_bytes())

        monkeypatch.setattr(dataset_module.requests, "get", fake_get)

        result = runner.invoke(app, ["--year", "2025"])

        assert result.exit_code == 0
        assert not (raw_dir / "2025_12").exists()
        assert (raw_dir / "2025_11").exists()

    def test_an_http_error_fails_the_run(self, raw_dir, monkeypatch):
        def fake_get(url, headers=None, timeout=None):
            raise requests.ConnectionError("network down")

        monkeypatch.setattr(dataset_module.requests, "get", fake_get)

        result = runner.invoke(app, ["--year", "2025"])

        assert result.exit_code == 1

    def test_a_corrupted_archive_fails_the_run(self, raw_dir, monkeypatch):
        def fake_get(url, headers=None, timeout=None):
            return _Response(b"this is not a zip")

        monkeypatch.setattr(dataset_module.requests, "get", fake_get)

        result = runner.invoke(app, ["--year", "2025"])

        assert result.exit_code == 1

    def test_a_corrupted_archive_leaves_nothing_behind(self, raw_dir, monkeypatch):
        """A half-extracted month would look 'already present' on the next run."""

        def fake_get(url, headers=None, timeout=None):
            return _Response(b"this is not a zip")

        monkeypatch.setattr(dataset_module.requests, "get", fake_get)
        runner.invoke(app, ["--year", "2025"])

        assert not list(raw_dir.glob("2025_*"))
        assert not list(raw_dir.glob("*.zip"))

    def test_a_failed_retry_clears_the_month_it_was_replacing(self, raw_dir, served, monkeypatch):
        """A failed --force must not leave last run's files looking current."""
        runner.invoke(app, ["--year", "2025"])
        assert (raw_dir / "2025_01" / "flights.csv").exists()

        monkeypatch.setattr(
            dataset_module.requests,
            "get",
            lambda url, headers=None, timeout=None: _Response(b"this is not a zip"),
        )
        result = runner.invoke(app, ["--year", "2025", "--force"])

        assert result.exit_code == 1
        assert not (raw_dir / "2025_01").exists()

    def test_an_unexpected_error_is_recorded_as_a_failed_month(self, raw_dir, monkeypatch):
        """Anything the two specific handlers miss must still be caught per month."""

        def fake_get(url, headers=None, timeout=None):
            return _Response(_zip_bytes())

        def boom(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(dataset_module.requests, "get", fake_get)
        monkeypatch.setattr(dataset_module.zipfile, "ZipFile", boom)

        result = runner.invoke(app, ["--year", "2025"])

        assert result.exit_code == 1
        assert not list(raw_dir.glob("2025_*"))

    def test_an_existing_month_is_replaced_when_forced(self, raw_dir, served):
        """--force must not merge new files into a stale directory."""
        runner.invoke(app, ["--year", "2025"])
        stale = raw_dir / "2025_01" / "leftover.csv"
        stale.write_text("old")

        runner.invoke(app, ["--year", "2025", "--force"])

        assert not stale.exists()
        assert (raw_dir / "2025_01" / "flights.csv").exists()

    def test_one_bad_month_does_not_stop_the_others(self, raw_dir, monkeypatch):
        def fake_get(url, headers=None, timeout=None):
            if url.endswith("2025_3.zip"):
                raise requests.ConnectionError("flaky")
            return _Response(_zip_bytes())

        monkeypatch.setattr(dataset_module.requests, "get", fake_get)

        result = runner.invoke(app, ["--year", "2025"])

        assert result.exit_code == 1  # the run is still reported as failed
        assert len(list(raw_dir.glob("2025_*"))) == 11
