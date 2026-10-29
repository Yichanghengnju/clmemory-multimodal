import base64
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi.testclient import TestClient
import mm_memory as mm


def picture(color="red", fmt="PNG"):
    out = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(out, format=fmt)
    mime = "jpeg" if fmt == "JPEG" else fmt.lower()
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64," + base64.b64encode(out.getvalue()).decode()}}


def txt(value):
    return {"type": "text", "text": value}


class Fake:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def caption(self, user, url):
        self.calls += 1
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        return "red square" if image.getpixel((0, 0))[0] > 200 else "blue square"

    def embed(self, user, text, kind):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic provider failure")
        return [[1.0, 0.0] if "red" in text.lower() else [0.0, 1.0]]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = mm.Store(Path(self.temp.name) / "test.db")
        self.fake = Fake()
        self.memory = mm.Memory(self.store, self.fake)
        self.client = TestClient(mm.create_app(self.memory, "test-only-token"))
        self.headers = {"Authorization": "Token test-only-token"}
        self.payload = {"request_id": "r1", "user_id": "u1", "session_id": "s1", "messages": [
            {"role": "user", "timestamp": 1704067200000, "content": [txt("red folder 37"), picture(), txt("private storage")]}]}

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def post(self, path, value):
        return self.client.post(path, json=value, headers=self.headers)

    def query(self, parts=None, user="u1", top=100):
        return {"user_id": user, "query": parts or [txt("red")], "top_k": top}

    def test_auth_rejects_missing_and_wrong(self):
        for headers in ({}, {"Authorization": "Token wrong"}):
            self.assertEqual(self.client.post("/add", json=self.payload, headers=headers).status_code, 401)

    def test_add_echo_source_order_image_and_metadata(self):
        response = self.post("/add", self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {k: self.payload[k] for k in ("request_id", "user_id", "session_id")} | {"success": True})
        result = self.post("/search", self.query()).json()["data"][0]
        self.assertEqual(result["content"][1:], self.payload["messages"][0]["content"])
        self.assertIn('"role":"user"', result["content"][0]["text"])
        self.assertNotIn("Image description", json.dumps(result))

    def test_pure_image_query_and_stable_reload(self):
        self.post("/add", self.payload)
        first = self.post("/search", self.query([picture()])).json()
        other = mm.Memory(mm.Store(self.store.path), self.fake)
        self.assertEqual(first, other.search(self.query([picture()])))

    def test_text_query_and_top_k(self):
        self.post("/add", self.payload)
        second = dict(self.payload, request_id="r2", messages=[{"role": "assistant", "content": [txt("blue folder"), picture("blue")]}])
        self.post("/add", second)
        result = self.post("/search", self.query(top=1)).json()["data"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"][1]["text"], "red folder 37")

    def test_empty_other_user_avoids_models(self):
        self.post("/add", self.payload)
        calls = self.fake.calls
        self.assertEqual(self.post("/search", self.query(user="u2")).json(), {"data": []})
        self.assertEqual(self.fake.calls, calls)

    def test_idempotence_no_provider_calls(self):
        self.post("/add", self.payload)
        calls = self.fake.calls
        self.assertEqual(self.post("/add", self.payload).status_code, 200)
        self.assertEqual(self.fake.calls, calls)

    def test_conflicting_payload_keeps_original(self):
        self.post("/add", self.payload)
        other = dict(self.payload, session_id="different")
        self.assertEqual(self.post("/add", other).status_code, 409)
        self.assertEqual(self.post("/search", self.query()).json()["data"][0]["content"][1:], self.payload["messages"][0]["content"])

    def test_concurrent_duplicate_add_is_single_write(self):
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda _: self.memory.add(self.payload), range(16)))
        self.assertTrue(all(r["success"] for r in results))
        self.assertEqual(self.fake.calls, 2)
        with self.store.db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 1)

    def test_failed_add_does_not_commit_partial_sources(self):
        self.fake.fail = True
        self.assertEqual(self.post("/add", self.payload).status_code, 503)
        with self.store.db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM requests").fetchone()[0], 0)

    def test_external_invalid_and_wrong_media_rejected(self):
        for url in ("https://example.com/image.png", "data:image/png;base64,AAAA", picture()["image_url"]["url"].replace("image/png", "image/jpeg")):
            bad = dict(self.payload, messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}])
            self.assertEqual(self.post("/add", bad).status_code, 422)
        self.assertEqual(self.fake.calls, 0)

    def test_jpeg_png_webp_are_accepted(self):
        for fmt in ("JPEG", "PNG", "WEBP"):
            self.assertGreater(mm.parts_info([picture(fmt=fmt)])[2], 0)

    def test_request_and_response_media_caps(self):
        n = mm.parts_info([picture()])[2]
        with patch.object(mm, "MEDIA_LIMIT", n - 1):
            self.assertEqual(self.post("/add", self.payload).status_code, 422)
        self.post("/add", self.payload)
        self.post("/add", dict(self.payload, request_id="r2"))
        with patch.object(mm, "MEDIA_LIMIT", n):
            self.assertEqual(len(self.post("/search", self.query()).json()["data"]), 1)

    def test_topk_options_and_schema_validation(self):
        for top in (0, 101, True, "100"):
            self.assertEqual(self.post("/search", self.query(top=top)).status_code, 422)
        self.assertEqual(self.post("/search", dict(self.query(), options=["A red", "B blue"])).status_code, 200)
        self.assertEqual(self.post("/search", dict(self.query(), options="invalid")).status_code, 422)
        self.assertEqual(self.post("/search", self.query(parts="legacy string")).status_code, 422)

    def test_retention_deletes_all_derived_records(self):
        self.post("/add", self.payload)
        with self.store.db() as db:
            db.execute("UPDATE memories SET at=0")
            db.execute("UPDATE requests SET at=0")
        self.store.purge()
        self.assertEqual(self.post("/search", self.query()).json(), {"data": []})

    def test_provider_unknown_is_not_resent(self):
        keyfile = Path(self.temp.name) / "dummy.key"
        keyfile.write_text("test-only")
        config = {"data_dir": self.temp.name, "vision": {"endpoint": "https://example.invalid", "api_key_file": str(keyfile)}}
        provider = mm.Providers(self.store, config)
        with patch.object(mm, "build_opener", side_effect=TimeoutError):
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    provider.call("u", "vision", {"model": "gpt-4o-mini"}, lambda x: x)
        with self.store.db() as db:
            rows = db.execute("SELECT state FROM calls").fetchall()
        self.assertEqual([r[0] for r in rows], ["unknown"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
