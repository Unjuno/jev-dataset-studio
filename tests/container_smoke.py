"""Black-box container smoke flow using only the Python standard library."""

import io
import json
import time
from urllib.request import Request, urlopen
import zipfile


def request(path, data=None):
    payload = json.dumps(data).encode() if data is not None else None
    req = Request("http://127.0.0.1:8000" + path, data=payload,
                  headers={"Content-Type": "application/json", "X-Studio-Request": "1"})
    with urlopen(req, timeout=20) as response:
        return response.read()


def main():
    assert b"Jev" in request("/")
    dataset = json.loads(request("/api/datasets/sample", {}))
    template = json.loads(request("/api/templates"))[0]
    job = json.loads(request("/api/jobs", {"dataset_id": dataset["id"], "recipe": template["recipe"]}))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        job = json.loads(request("/api/jobs/" + job["id"]))
        if job["status"] in {"completed", "completed_with_errors"}:
            break
        time.sleep(0.5)
    assert job["status"] == "completed" and job["failed"] == 0, job
    result = json.loads(request("/api/jobs/" + job["id"] + "/exports", template["export_spec"]))
    archive = zipfile.ZipFile(io.BytesIO(request(result["download_url"])))
    manifest = json.loads(archive.read("manifest.json"))
    assert result["rows"] > 0 and manifest["demo_annotations"] > 0
    print("PASS: web → durable worker → annotated export ZIP")


if __name__ == "__main__":
    main()
