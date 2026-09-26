import json
import os
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

import studio_discovery as discovery


class DiscoveryTests(unittest.TestCase):
    def test_code_descriptions_explain_role_benefit_and_actual_release_changes(self):
        release = {"tag_name": "v2", "body": "## Changes\n- Fix memory usage [#42](https://example.test/42)\n"
                   "* Add `image editing` support\n", "published_at": "2026-09-26"}
        for repo, reason in discovery.CODE.items():
            item = discovery.code_feed(repo, reason, lambda *a: release)[0]
            self.assertEqual(item["repo"], repo)
            self.assertEqual(item["version"], "v2")
            self.assertEqual(item["highlights"], ["Fix memory usage #42", "Add image editing support"])
            self.assertTrue(item["purpose"])
            self.assertTrue(item["improves"])
            self.assertTrue(item["integration"])

    def fake(self, url, token):
        if "api.github.com" in url:
            self.assertEqual(token, "")
            return {"tag_name": "v1", "body": "Release notes", "published_at": "2026-09-26"}
        self.assertEqual(token, "provider-secret")
        if "huggingface.co" in url:
            return [{"id": "org/image-model", "downloads": 40, "tags": ["license:apache-2.0"]}]
        return {"items": [{"id": 123, "name": "Style", "type": "LORA",
                           "modelVersions": [{"name": "v1", "baseModel": "Flux.1 D"}]},
                          {"id": 124, "name": "Other", "type": "LORA",
                           "modelVersions": [{"baseModel": "Unrelated"}]}]}

    def test_both_sources_include_models_and_code_cache_without_keys(self):
        with tempfile.TemporaryDirectory() as root:
            for source in ("huggingface", "civitai"):
                result = discovery.discover(root, source, "provider-secret", fetch=self.fake)
                self.assertFalse(result["errors"])
                self.assertEqual(len(result["items"]), 4)
                self.assertEqual(sum(x["kind"] == "Code / library" for x in result["items"]), 3)
                with open(discovery.cache_path(root, source), encoding="utf-8") as stream:
                    self.assertNotIn("provider-secret", stream.read())
                with patch.object(discovery, "get_json", side_effect=AssertionError("cache missed")):
                    self.assertEqual(discovery.discover(root, source), result)

    def test_failures_retain_cached_results_and_allow_retry(self):
        with tempfile.TemporaryDirectory() as root:
            discovery.discover(root, "huggingface", "provider-secret", fetch=self.fake)
            def broken(url, token):
                raise discovery.DiscoveryError("Unavailable")
            result = discovery.discover(root, "huggingface", force=True, fetch=broken)
            self.assertTrue(result["errors"])
            self.assertEqual(len(result["items"]), 4)
            self.assertTrue(all(item["stale"] for item in result["items"]))
            result = discovery.discover(root, "huggingface", "provider-secret", fetch=self.fake)
            self.assertFalse(result["errors"])
            self.assertTrue(all(not item.get("stale") for item in result["items"]))

    def test_force_refresh_and_expired_cache_fetch_again(self):
        with tempfile.TemporaryDirectory() as root:
            first = discovery.discover(root, "civitai", "provider-secret", fetch=self.fake)
            with patch.object(discovery, "model_feed", wraps=discovery.model_feed) as feed:
                discovery.discover(root, "civitai", "provider-secret", force=True, fetch=self.fake)
                self.assertEqual(feed.call_count, 2)
            with patch.object(discovery.time, "time", return_value=first["checked"] + discovery.TTL + 10):
                with patch.object(discovery, "model_feed", wraps=discovery.model_feed) as feed:
                    discovery.discover(root, "civitai", "provider-secret", fetch=self.fake)
                    self.assertEqual(feed.call_count, 2)

    def test_redirect_cannot_send_a_provider_key_to_another_origin(self):
        handler = discovery.SameOriginRedirect()
        req = urllib.request.Request("https://huggingface.co/api/models",
                                     headers={"Authorization": "Bearer secret"})
        for url in ("https://other.test/api/models", "http://huggingface.co/api/models"):
            with self.assertRaises(discovery.DiscoveryError):
                handler.redirect_request(req, None, 302, "Found", {}, url)

    def test_bad_repository_ids_never_become_links(self):
        rows = discovery.model_feed("huggingface", "", "FLUX", lambda *a: [
            {"id": "org/model"}, {"id": "https://evil.test"}, {"id": "org/model?token=secret"}])
        self.assertEqual([x["url"] for x in rows], ["https://huggingface.co/org/model"])
