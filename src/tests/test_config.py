import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src import config as conf


@pytest.fixture
def cfg_dir(tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AUTOGYM_")}
    env["AUTOGYM_CONFIG_DIR"] = str(tmp_path)
    with patch.dict(os.environ, env, clear=True):
        yield tmp_path


def test_defaults_when_no_file(cfg_dir: Path) -> None:
    cfg = conf.Config()
    assert cfg.get("AUTOGYM_CAMERA") == "webcam"
    assert cfg.get_bool("AUTOGYM_UI_FULLSCREEN") is True
    assert cfg.get_float("AUTOGYM_SET_PAUSE_SECONDS") == 3.0


def test_write_then_read_round_trip(cfg_dir: Path) -> None:
    conf.write_envars_file({"AUTOGYM_CAMERA": "file", "AUTOGYM_CAMERA_PATH": "/tmp/my video.mp4"})
    assert conf.envars_path().is_file()
    values = conf.read_envars_file()
    assert values["AUTOGYM_CAMERA_PATH"] == "/tmp/my video.mp4"
    assert conf.Config().get("AUTOGYM_CAMERA") == "file"


def test_file_is_sourceable_by_shell(cfg_dir: Path) -> None:
    import subprocess
    conf.write_envars_file({"AUTOGYM_WEB_PASSWORD": "it's \"tricky\" $HOME"})
    out = subprocess.run(["bash", "-c", f'source "{conf.envars_path()}"; printf %s "$AUTOGYM_WEB_PASSWORD"'],
                         capture_output=True, text=True, check=True)
    assert out.stdout == "it's \"tricky\" $HOME"


def test_write_merges_and_keeps_unknown_keys(cfg_dir: Path) -> None:
    conf.envars_path().write_text('export MY_OWN="x"\nexport AUTOGYM_CAMERA="picamera"\n')
    conf.write_envars_file({"AUTOGYM_UI_FONT_SCALE": "0.8"})
    values = conf.read_envars_file()
    assert values == {"MY_OWN": "x", "AUTOGYM_CAMERA": "picamera", "AUTOGYM_UI_FONT_SCALE": "0.8"}
    assert conf.envars_path().with_suffix(".sh.bak").is_file()


def test_process_env_overrides_file(cfg_dir: Path) -> None:
    conf.write_envars_file({"AUTOGYM_UI_FULLSCREEN": "1"})
    with patch.dict(os.environ, {"AUTOGYM_UI_FULLSCREEN": "0"}):
        assert conf.Config().get_bool("AUTOGYM_UI_FULLSCREEN") is False


def test_override_ignores_none(cfg_dir: Path) -> None:
    cfg = conf.Config().override(AUTOGYM_CAMERA=None, AUTOGYM_POSE_BACKEND="coral")
    assert cfg.get("AUTOGYM_CAMERA") == "webcam"
    assert cfg.get("AUTOGYM_POSE_BACKEND") == "coral"


def test_validate() -> None:
    clean, errors = conf.validate({
        "AUTOGYM_CAMERA_INDEX": "2",
        "AUTOGYM_UI_FULLSCREEN": "true",
        "AUTOGYM_POSE_BACKEND": "gpu",
        "AUTOGYM_UI_COLOR_BG": "red",
        "AUTOGYM_SET_PAUSE_SECONDS": "abc",
        "NOT_A_SETTING": "1",
    })
    assert clean == {"AUTOGYM_CAMERA_INDEX": "2", "AUTOGYM_UI_FULLSCREEN": "1"}
    assert set(errors) == {"AUTOGYM_POSE_BACKEND", "AUTOGYM_UI_COLOR_BG",
                           "AUTOGYM_SET_PAUSE_SECONDS", "NOT_A_SETTING"}


def test_bad_number_falls_back_to_default() -> None:
    assert conf.Config({"AUTOGYM_CAMERA_INDEX": "x"}).get_int("AUTOGYM_CAMERA_INDEX") == 0


def test_validate_threshold_order() -> None:
    _, errors = conf.validate({"AUTOGYM_SQUAT_DOWN_ANGLE": "170", "AUTOGYM_SQUAT_UP_ANGLE": "160"})
    assert "AUTOGYM_SQUAT_UP_ANGLE" in errors
