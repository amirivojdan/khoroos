"""Web API tests.

The job manager's worker thread is never allowed to load real models here: jobs are
driven with stub models, or inspected before the worker picks them up.
"""

from __future__ import annotations

import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from khoroos.config import Settings, WelfareThresholds  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    from khoroos.web.app import create_app

    settings = Settings(cache_dir=tmp_path / "cache", device="cpu")
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    app.state.jobs.close()


@pytest.fixture
def stub_client(client, stub_runner):
    """A client whose job worker runs the stub models instead of real weights."""
    client.app.state.jobs.runner = stub_runner
    return client


def wait_for(client, job_id, states, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] in states:
            return status
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} never reached {states}")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_config_endpoint_describes_the_ui(client):
    payload = client.get("/api/config").json()
    assert set(payload["presets"]) == {"fast", "balanced", "thorough"}
    assert len(payload["classes"]) == 15
    assert "comfort" in payload["behaviour_groups"]
    assert "min_comfort_share" in payload["thresholds"]


def test_static_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Khoroos" in response.text
    assert 'class="player-stage"' in response.text
    assert 'id="player-overlay"' in response.text
    assert "styles.css?v=responsive-32" in response.text
    assert "app.js?v=responsive-10" in response.text


def test_select_screen_uses_a_single_video_dropzone(client):
    page = client.get("/").text
    script = client.get("/app.js").text

    assert "Drop a video or select a video file" in page
    assert 'id="file-input" type="file" accept="video/*"' in page
    assert 'id="select-stage-file"' in page
    assert 'id="select-stage-options"' in page
    assert 'id="select-stage-options" class="select-stage" tabindex="-1" hidden' in page
    assert 'id="change-video"' in page
    assert 'id="browse-list"' not in page
    assert 'class="tabs"' not in page
    assert "showSelectStage('options')" in script
    assert "showSelectStage('file')" in script


def test_select_screen_opens_with_project_intro_before_setup(client):
    page = client.get("/").text
    script = client.get("/app.js").text

    assert "<h1>Khoroos</h1>" in page
    assert 'class="hero-subtitle">An open toolkit for poultry welfare analysis' in page
    assert 'class="hero-name-note">(Persian for “rooster”, pronounced kho-ROOS)' in page
    assert "quantitative welfare indicators. Developed at the" in " ".join(page.split())
    assert 'class="site-nav"' in page
    assert 'class="hero-features"' in page
    assert "Precision Livestock Farming" in page
    assert "Livestock Video Analytics" in page
    assert "Automated Welfare Assessment" in page
    assert "https://www.ut-smartagriculture.com/" in page
    assert "Khoroos is an open research toolkit for video-based poultry behavior analysis" in page
    assert "individual bird trajectories, behavior timelines" in page
    assert "UT Smart Agriculture Lab" in page
    assert "View on GitHub" in page
    assert 'id="getting-started"' in page
    assert "Getting Started" in page
    assert 'id="select-setup" hidden' in page
    assert "showSelectSetup" in script
    assert "showSelectLanding" in script
    assert "document.body.dataset.selectView = 'landing'" in script
    assert "document.body.dataset.selectView = 'setup'" in script


def test_select_screen_uses_aligned_pixel_art(client):
    from io import BytesIO

    from PIL import Image

    page = client.get("/").text
    styles = client.get("/styles.css").text
    image = client.get("/Khoroos-transparent.png")

    landing = page[page.index('id="select-landing"') : page.index('id="select-setup"')]
    assert 'class="hero-art"' in landing
    assert 'src="/Khoroos-transparent.png"' in page
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert ".hero-art" in styles
    assert "grid-template-columns" in styles
    assert "scaleX(-1)" in styles
    assert "image-rendering: pixelated" in styles

    artwork = Image.open(BytesIO(image.content))
    alpha = artwork.getchannel("A")
    assert artwork.mode == "RGBA"
    assert alpha.getextrema() == (0, 255)
    assert all(
        alpha.getpixel(point) == 0
        for point in ((0, 0), (artwork.width - 1, 0), (0, artwork.height - 1))
    )


def test_processing_and_results_share_the_pixel_theme(client):
    styles = client.get("/styles.css").text

    assert 'body[data-screen="select"][data-select-view="setup"] .brand-mark' in styles
    assert 'body[data-screen="running"] .brand-mark' in styles
    assert 'body[data-screen="results"] .brand-mark' in styles
    assert 'body[data-screen="running"] .progress-track' in styles
    assert 'body[data-screen="running"] .stage .dot' in styles
    assert 'body[data-screen="results"] .kpi' in styles
    assert 'body[data-screen="results"] .player-stage' in styles
    assert 'body[data-screen="results"] .timeline-canvas' in styles


def test_layout_has_narrow_and_short_viewport_guards(client):
    styles = client.get("/styles.css").text

    assert "@media (max-width: 620px)" in styles
    assert "@media (max-width: 380px)" in styles
    assert "@media (max-height: 680px) and (min-width: 621px)" in styles
    assert "overflow-x: clip" in styles
    assert ".chart-grid > *" in styles
    assert "min-inline-size: 0" in styles
    assert "min-height: 44px" in styles
    assert "body[data-screen=\"select\"] .hero-copy { display: contents; }" in styles
    assert "touch-action: manipulation" in styles


def test_logo_is_served_and_is_a_transparent_png(client):
    """The mark ships inside the package, so a packaging slip must fail loudly here."""
    from io import BytesIO

    from PIL import Image

    response = client.get("/khoroos.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    logo = Image.open(BytesIO(response.content))
    # Transparency matters: the topbar sits on a near-black surface in the dark theme,
    # and the source artwork is drawn on opaque white.
    assert logo.mode == "RGBA"
    assert logo.getchannel("A").getextrema()[0] == 0, "the logo has no transparent pixels"
    assert len(response.content) < 60_000, "the logo should stay small enough to inline-load"


def test_logo_is_used_as_the_brand_mark(client):
    page = client.get("/").text
    styles = client.get("/styles.css").text

    assert 'class="brand-mark" src="/khoroos.png"' in page
    assert 'rel="icon" href="/khoroos.png"' in page
    assert "image-rendering: pixelated" in styles


def test_track_overlay_assets_are_layered_and_animate(client):
    styles = client.get("/styles.css").text
    script = client.get("/app.js").text

    assert ".player-stage" in styles and "position: relative" in styles
    assert ".player-overlay" in styles and "position: absolute" in styles
    assert "TRACK_TRAIL_SECONDS" in script
    assert "drawTrackTrail" in script
    assert "drawActionBox" in script
    assert "player.clientWidth" in script


def test_browse_lists_directories(client, tmp_path, synthetic_video):
    response = client.get("/api/browse", params={"path": str(synthetic_video.parent)})
    assert response.status_code == 200
    payload = response.json()
    assert any(v["name"] == synthetic_video.name for v in payload["videos"])


def test_browse_rejects_a_missing_directory(client, tmp_path):
    response = client.get("/api/browse", params={"path": str(tmp_path / "nope")})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def test_job_requires_a_video(client):
    response = client.post("/api/jobs", data={"preset": "balanced"})
    assert response.status_code == 400


def test_job_rejects_an_unknown_preset(client, synthetic_video):
    response = client.post(
        "/api/jobs", data={"server_path": str(synthetic_video), "preset": "nonsense"}
    )
    assert response.status_code == 400


def test_job_accepts_threshold_overrides(client, synthetic_video):
    response = client.post(
        "/api/jobs",
        data={
            "server_path": str(synthetic_video),
            "thresholds": json.dumps({"min_locomotion_share": 0.25}),
        },
    )
    assert response.status_code == 200

    job = client.app.state.jobs.get(response.json()["job_id"])
    assert job.thresholds.min_locomotion_share == 0.25
    assert job.thresholds.min_comfort_share == WelfareThresholds().min_comfort_share


def test_job_rejects_an_unknown_threshold(client, synthetic_video):
    response = client.post(
        "/api/jobs",
        data={"server_path": str(synthetic_video), "thresholds": json.dumps({"min_comfort": 0.2})},
    )
    assert response.status_code == 400
    assert "min_comfort_share" in response.json()["detail"]  # lists the valid names


def test_job_rejects_malformed_threshold_json(client, synthetic_video):
    response = client.post(
        "/api/jobs", data={"server_path": str(synthetic_video), "thresholds": "not json"}
    )
    assert response.status_code == 400


def test_bad_parameters_are_rejected_before_the_upload_is_stored(client, synthetic_video, tmp_path):
    """A gigabyte upload must not be written to disk only to fail on a typo'd preset."""
    uploads = tmp_path / "cache" / "jobs" / "uploads"
    with synthetic_video.open("rb") as handle:
        response = client.post(
            "/api/jobs",
            data={"preset": "nonsense"},
            files={"file": ("farm.mp4", handle, "video/mp4")},
        )

    assert response.status_code == 400
    assert not uploads.exists() or not list(uploads.iterdir())


def test_job_rejects_a_non_video_file(client, tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not a video")
    response = client.post("/api/jobs", data={"server_path": str(text_file)})
    assert response.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeef/result").status_code == 404


def test_full_job_run_produces_results(stub_client, synthetic_video):
    response = stub_client.post(
        "/api/jobs",
        data={"server_path": str(synthetic_video), "preset": "balanced", "min_confidence": "0.5"},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = wait_for(stub_client, job_id, {"completed", "failed"})
    assert status["state"] == "completed", status.get("error")
    assert status["progress"] == 1.0

    result = stub_client.get(f"/api/jobs/{job_id}/result").json()
    assert result["schema_version"] == "1.0"
    assert result["predictions"]
    assert result["metrics"]["time_budget"]["total_bird_seconds"] > 0


def test_batch_sizes_are_selectable_per_job(stub_client, synthetic_video):
    response = stub_client.post(
        "/api/jobs",
        data={
            "server_path": str(synthetic_video),
            "preset": "balanced",
            "detection_batch_size": "7",
            "action_batch_size": "2",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = wait_for(stub_client, job_id, {"completed", "failed"})
    assert status["state"] == "completed", status.get("error")

    params = stub_client.get(f"/api/jobs/{job_id}/result").json()["params"]
    assert params["detection_batch_size"] == 7
    assert params["action_batch_size"] == 2


def test_omitted_batch_sizes_fall_back_to_the_preset(stub_client, synthetic_video):
    """An untouched field must leave the detail level's own value in place."""
    from khoroos.config import PRESETS

    job_id = stub_client.post(
        "/api/jobs", data={"server_path": str(synthetic_video), "preset": "fast"}
    ).json()["job_id"]
    wait_for(stub_client, job_id, {"completed", "failed"})

    params = stub_client.get(f"/api/jobs/{job_id}/result").json()["params"]
    assert params["detection_batch_size"] == PRESETS["fast"]["detection_batch_size"]
    assert params["action_batch_size"] == PRESETS["fast"]["action_batch_size"]


def test_a_zero_batch_size_is_rejected(client, synthetic_video):
    response = client.post(
        "/api/jobs",
        data={"server_path": str(synthetic_video), "detection_batch_size": "0"},
    )
    assert response.status_code == 400
    assert "detection_batch_size" in response.json()["detail"]


def test_result_is_409_before_completion(stub_client, synthetic_video):
    job_id = stub_client.post("/api/jobs", data={"server_path": str(synthetic_video)}).json()["job_id"]
    response = stub_client.get(f"/api/jobs/{job_id}/result")
    assert response.status_code in (200, 409)  # may already have finished


def test_exports_download_after_completion(stub_client, synthetic_video):
    job_id = stub_client.post(
        "/api/jobs", data={"server_path": str(synthetic_video)}
    ).json()["job_id"]
    wait_for(stub_client, job_id, {"completed", "failed"})

    for artifact in ("predictions", "time_budget", "per_bird", "metrics", "result"):
        response = stub_client.get(f"/api/jobs/{job_id}/export/{artifact}")
        assert response.status_code == 200, artifact
        assert response.content

    assert stub_client.get(f"/api/jobs/{job_id}/export/bogus").status_code == 404


def test_source_video_is_served_for_the_player(stub_client, synthetic_video):
    job_id = stub_client.post(
        "/api/jobs", data={"server_path": str(synthetic_video)}
    ).json()["job_id"]
    response = stub_client.get(f"/api/jobs/{job_id}/video")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/")


def test_job_can_be_deleted(stub_client, synthetic_video):
    job_id = stub_client.post(
        "/api/jobs", data={"server_path": str(synthetic_video)}
    ).json()["job_id"]
    wait_for(stub_client, job_id, {"completed", "failed"})

    assert stub_client.delete(f"/api/jobs/{job_id}").json() == {"deleted": True}
    assert stub_client.get(f"/api/jobs/{job_id}").status_code == 404


def test_active_job_cannot_be_deleted(client, synthetic_video):
    from khoroos.config import params_for_preset
    from khoroos.pipeline.jobs import Job, JobState

    manager = client.app.state.jobs
    job = Job(
        job_id="still-running",
        video_path=synthetic_video,
        original_filename=synthetic_video.name,
        params=params_for_preset(),
        thresholds=WelfareThresholds(),
        state=JobState.RUNNING,
    )
    with manager._lock:
        manager._jobs[job.job_id] = job
        manager._subscribers[job.job_id] = []

    response = client.delete(f"/api/jobs/{job.job_id}")
    assert response.status_code == 409
    assert synthetic_video.exists()


def test_cancelling_a_finished_job_is_409(stub_client, synthetic_video):
    job_id = stub_client.post(
        "/api/jobs", data={"server_path": str(synthetic_video)}
    ).json()["job_id"]
    wait_for(stub_client, job_id, {"completed", "failed"})
    assert stub_client.post(f"/api/jobs/{job_id}/cancel").status_code == 409
