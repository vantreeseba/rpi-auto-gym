import base64
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from src import config as conf
from src.webconf.server import PASSWORD_MASK, Handler


@pytest.fixture
def web(tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AUTOGYM_")}
    env["AUTOGYM_CONFIG_DIR"] = str(tmp_path)
    # Point the control port at something closed so "app not running" is tested.
    with patch.dict(os.environ, env, clear=True):
        conf.write_envars_file({"AUTOGYM_CONTROL_PORT": "1"})
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{server.server_address[1]}"
        server.shutdown()


def call(url, data=None, headers=None, ctype="application/json"):
    hdrs = dict(headers or {})
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        hdrs["Content-Type"] = ctype
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_serves_index_and_schema(web) -> None:
    status, body = call(web + "/")
    assert status == 200 and b"Auto Gym" in body
    status, body = call(web + "/api/schema")
    keys = [s["key"] for s in json.loads(body)]
    assert "AUTOGYM_CAMERA" in keys


def test_static_path_traversal_blocked(web) -> None:
    status, _ = call(web + "/static/../server.py")
    assert status == 404


def test_save_config_and_validation(web) -> None:
    status, body = call(web + "/api/config", {"values": {"AUTOGYM_UI_FONT_SCALE": "0.75"}})
    assert status == 200 and json.loads(body)["ok"]
    assert conf.read_envars_file()["AUTOGYM_UI_FONT_SCALE"] == "0.75"

    status, body = call(web + "/api/config", {"values": {"AUTOGYM_POSE_BACKEND": "nope"}})
    assert status == 400 and "AUTOGYM_POSE_BACKEND" in json.loads(body)["errors"]


def test_post_requires_json_content_type(web) -> None:
    status, _ = call(web + "/api/config", {"values": {}}, ctype="text/plain")
    assert status == 415


def test_state_when_app_not_running(web) -> None:
    status, body = call(web + "/api/state")
    assert status == 503 and json.loads(body)["running"] is False


def test_password(web) -> None:
    call(web + "/api/config", {"values": {"AUTOGYM_WEB_PASSWORD": "s3cret"}})
    assert call(web + "/api/info")[0] == 401
    bad = {"Authorization": "Basic " + base64.b64encode(b"x:wrong").decode()}
    good = {"Authorization": "Basic " + base64.b64encode(b"admin:s3cret").decode()}
    assert call(web + "/api/info", headers=bad)[0] == 401
    status, body = call(web + "/api/config", headers=good)
    assert status == 200
    assert json.loads(body)["values"]["AUTOGYM_WEB_PASSWORD"] == PASSWORD_MASK
    # Saving the mask back must not overwrite the real password.
    call(web + "/api/config", {"values": {"AUTOGYM_WEB_PASSWORD": PASSWORD_MASK}}, headers=good)
    assert conf.read_envars_file()["AUTOGYM_WEB_PASSWORD"] == "s3cret"


def test_info_lists_web_urls(web) -> None:
    status, body = call(web + "/api/info")
    assert status == 200 and json.loads(body)["web_urls"][0].endswith(":8080")
