"""Exercise the browser workflow against a running server, including safe text."""

import io
import json
import os
from pathlib import Path
import zipfile

from playwright.sync_api import expect, sync_playwright


def main():
    base = os.environ.get("STUDIO_TEST_URL", "http://127.0.0.1:8000")
    artifacts = Path("artifacts/browser")
    artifacts.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text)
                if message.type == "error" and "Content Security Policy" in message.text else None)
        try:
            page.goto(base)
            expect(page.locator("#load-sample")).to_be_visible()
            with page.expect_response(lambda r: r.url.endswith("/api/datasets/sample")):
                page.locator("#load-sample").click()
            expect(page.locator("#dataset-detail")).to_be_visible()
            with page.expect_response(lambda r: r.url.endswith("/api/preview")) as response:
                page.locator("#run-preview").click()
            preview = response.value.json()
            assert response.value.status == 200 and preview["items"], preview
            expect(page.locator("#preview-content mark").first).to_be_visible()
            assert all(item["origin"] == "demo" for item in preview["items"])
            page.screenshot(path=str(artifacts / "desktop-preview.png"), full_page=True)

            # Saving/loading a recipe is a form workflow, not hand-written JSON.
            page.locator("#recipe-name").fill("ブラウザ検証用レシピ")
            with page.expect_response(lambda r: r.url.endswith("/api/templates") and r.request.method == "POST"):
                page.locator("#save-template").click()
            expect(page.locator("#notice")).to_contain_text("保存")

            with page.expect_response(lambda r: r.url.endswith("/api/jobs") and r.request.method == "POST") as response:
                page.locator("#create-job").click()
            job = response.value.json()
            assert response.value.status == 200, job
            job_id = job["id"]
            # Poll the real worker through the UI; no direct DB state alteration.
            expect(page.locator("#jobs-page")).to_be_visible()
            expect(page.locator("#jobs-list .job-card").first).to_contain_text("6 / 6 行完了", timeout=60000)
            page.locator("#jobs-list .job-card").first.get_by_role("link", name="レビュー・出力へ").click()
            expect(page.locator("#annotation-editor")).to_contain_text("判定の確認")
            editor = page.locator("#annotation-editor")
            editor.get_by_label("現象", exact=True).select_option("other")
            editor.get_by_label("異常の確認", exact=True).select_option("false")
            editor.get_by_label("スコア（未確定なら空欄）", exact=True).fill("1")
            editor.get_by_label("レビューのメモ（任意）", exact=True).fill("ブラウザから確認")
            with page.expect_response(lambda r: "/api/annotations/" in r.url and r.request.method == "PATCH") as response:
                editor.get_by_role("button", name="承認する", exact=True).click()
            reviewed = response.value.json()
            assert response.value.status == 200 and reviewed["reviewed"] and reviewed["origin"] == "demo", reviewed
            output = page.locator("#job-output")
            output.get_by_label("出力する注釈", exact=True).select_option("reviewed")
            with page.expect_response(lambda r: r.url.endswith("/exports") and r.request.method == "POST") as response:
                page.locator("#create-export").click()
            export = response.value.json()
            assert response.value.status == 200 and export["rows"] == 1, export
            with page.expect_download() as download:
                page.get_by_role("link", name="ZIP をダウンロード", exact=True).click()
            contents = Path(download.value.path()).read_bytes()
            archive = zipfile.ZipFile(io.BytesIO(contents))
            row = json.loads(archive.read("dataset.jsonl"))
            assert row["reviewed"] is True and row["origin"] == "demo"
            assert "ブラウザから確認" in archive.read("reviews.jsonl").decode()
            page.screenshot(path=str(artifacts / "desktop-review.png"), full_page=True)

            # An input string must remain text, including HTML-like source content.
            page.locator('#navigation a[data-page="recipe"]').click()
            text = '<img src=x onerror="window.__source_xss=1">🔧e\u0301 振動。振動。'
            payload = json.dumps({"id": "001", "equipment": "X", "text": text}, ensure_ascii=False).encode()
            with page.expect_response(lambda r: r.url.endswith("/api/datasets") and r.request.method == "POST"):
                page.locator("#upload-file").set_input_files({"name": "unicode.jsonl", "mimeType": "application/json", "buffer": payload})
            page.locator("#target-mode").select_option("literal")
            page.locator("#target-terms").fill("振動")
            with page.expect_response(lambda r: r.url.endswith("/api/preview")) as response:
                page.locator("#run-preview").click()
            unicode_preview = response.value.json()
            assert response.value.status == 200 and len(unicode_preview["items"]) == 2, unicode_preview
            for item in unicode_preview["items"]:
                target = item["target"]
                assert text[target["start"]:target["end"]] == "振動"
            assert page.evaluate("window.__source_xss ?? null") is None
            assert page.locator("#sample-table img").count() == 0
            assert page.locator("#preview-content img").count() == 0

            mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
            mobile.on("pageerror", lambda error: errors.append(str(error)))
            mobile.goto(base)
            mobile.locator("#load-sample").click()
            expect(mobile.locator("#dataset-detail")).to_be_visible()
            assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), "Mobile page overflows horizontally."
            mobile.screenshot(path=str(artifacts / "mobile-recipe.png"), full_page=True)
            mobile.close()
            assert not errors, errors
            print("PASS: desktop preview → saved recipe → job → review → ZIP; Unicode/XSS; mobile layout")
            print("Job:", job_id)
        finally:
            page.screenshot(path=str(artifacts / "last-page.png"), full_page=True)
            (artifacts / "browser-errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2))
            browser.close()


if __name__ == "__main__":
    main()
