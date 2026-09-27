"""Bounded, read-only discovery of image models and relevant upstream releases.

Only metadata is fetched. Remote descriptions never become instructions, and
provider keys are confined to their own HTTPS origin (including redirects).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import apps.image_studio.model_sources as sources

TTL = 24 * 60 * 60
MAX_BYTES = 2 * 1024 * 1024
CODE = {
    "Comfy-Org/ComfyUI": "Rendering backend: review model support, memory use and API changes.",
    "huggingface/diffusers": "Image pipeline reference: review new sampling and editing techniques for backend workflows.",
    "facefusion/facefusion": "Face restoration and swapping: review changes relevant to the existing FaceFusion adapter.",
}
CODE_DESCRIPTIONS = {
    "Comfy-Org/ComfyUI": {
        "purpose": "Runs the image-generation workflows that Image Studio sends to your ComfyUI backends.",
        "improves": "Can expand the models and workflow nodes available to Image Studio and bring rendering or memory-management fixes.",
        "integration": "Updates belong on the selected ComfyUI backend. Existing workflows must still pass a test render.",
    },
    "huggingface/diffusers": {
        "purpose": "Provides Python building blocks for generating and editing images with diffusion models.",
        "improves": "Can provide new sampling, editing and memory-saving techniques for a new backend integration.",
        "integration": "Studio Assist does not currently use Diffusers directly. It needs an adapter in an isolated backend environment.",
    },
    "facefusion/facefusion": {
        "purpose": "Replaces or restores faces in finished pictures using your reference photos.",
        "improves": "Can improve the final face-swap stage through updated face processors, supported models and compatibility fixes.",
        "integration": "Uses the existing FaceFusion adapter. Verify identity matching and that pixels outside the face mask stay unchanged.",
    },
}


def release_highlights(body):
    """Readable release bullets, attributed to the publisher rather than promised gains."""
    highlights = []
    for line in str(body or "").splitlines():
        line = line.strip()
        if not line.startswith(("- ", "* ")):
            continue
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line[2:])
        line = re.sub(r"https?://\S+", "", line)
        line = text(line.replace("`", "").replace("**", ""), 260)
        if line and not line.lower().startswith(("new contributor", "full changelog")):
            highlights.append(line)
        if len(highlights) == 4:
            break
    return highlights


class DiscoveryError(Exception):
    pass


class SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if (new.scheme, new.netloc) != (old.scheme, old.netloc):
            raise DiscoveryError("The service redirected to a different host; request stopped.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def get_json(url, token=""):
    headers = {"Accept": "application/json", "User-Agent": "StudioAssist/1 discovery"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.build_opener(SameOriginRedirect()).open(request, timeout=15) as response:
            data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise DiscoveryError("Service response was too large.")
        return json.loads(data)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise DiscoveryError("Access refused; check the API key or service rate limit.") from None
        if error.code == 429:
            raise DiscoveryError("Service rate limit reached; try again later.") from None
        raise DiscoveryError("Service returned HTTP %d." % error.code) from None
    except (OSError, ValueError):
        raise DiscoveryError("Could not read the service response; check the connection.") from None


def text(value, limit=600):
    return " ".join(str(value or "").split())[:limit]


def cache_path(root, source):
    sources.SOURCES[source]
    return os.path.join(root, source + "-discoveries.json")


def cached(root, source):
    try:
        with open(cache_path(root, source), encoding="utf-8") as stream:
            data = json.load(stream)
        if (isinstance(data, dict) and isinstance(data.get("items"), list)
                and isinstance(data.get("checked"), (int, float))):
            return data
    except (OSError, ValueError):
        pass
    return {"items": [], "checked": 0, "errors": []}


def model_feed(source, token, query, fetch):
    if source == "huggingface":
        url = "https://huggingface.co/api/models?" + urllib.parse.urlencode({
            "search": query, "sort": "downloads", "direction": -1, "limit": 6})
        rows = fetch(url, token)
        if not isinstance(rows, list):
            raise DiscoveryError("Hugging Face returned an unexpected model list.")
        result = []
        for row in rows:
            rid = row.get("id", "")
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", rid):
                continue
            tags = row.get("tags") or []
            license_ = next((t[8:] for t in tags if isinstance(t, str) and t.startswith("license:")), "not listed")
            result.append({"title": text(rid), "url": "https://huggingface.co/" + rid,
                           "kind": "Model", "why": "Matches %s, an image model family used by Studio Assist." % query,
                           "details": "Downloads: %s · License: %s · Backend compatibility and VRAM need review." % (
                               text(row.get("downloads", "unknown")), text(license_)),
                           "importable": False})
        return result
    url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode({
        "types": query, "limit": 12, "sort": "Most Downloaded", "period": "Month", "nsfw": "false"})
    data = fetch(url, token)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise DiscoveryError("Civitai returned an unexpected model list.")
    result = []
    for row in data["items"]:
        if not isinstance(row.get("id"), int):
            continue
        versions = row.get("modelVersions") or []
        version = versions[0] if versions else {}
        base = text(version.get("baseModel", "unknown"))
        # Families for which this application has workflows.
        if not any(word in base.lower() for word in ("flux", "sdxl", "z image", "z-image", "qwen")):
            continue
        result.append({"title": text(row.get("name")), "url": "https://civitai.com/models/%d" % row["id"],
                       "kind": text(row.get("type", query)),
                       "why": "Popular %s for %s; a candidate for image styles or detail." % (query, base),
                       "details": "Base: %s · Version: %s · Review license and exact backend compatibility before importing." % (
                           base, text(version.get("name"))),
                       "importable": row.get("type") == "LORA"})
    return result


def code_feed(repo, reason, fetch):
    data = fetch("https://api.github.com/repos/%s/releases/latest" % repo, "")
    if not isinstance(data, dict) or not data.get("tag_name"):
        raise DiscoveryError("No published release was returned.")
    return [{"title": "%s — %s" % (repo, text(data["tag_name"])),
             "repo": repo, "version": str(data["tag_name"]),
             **CODE_DESCRIPTIONS[repo],
             "highlights": release_highlights(data.get("body")),
             "url": "https://github.com/%s/releases/tag/%s" % (
                 repo, urllib.parse.quote(str(data["tag_name"]), safe="")),
             "kind": "Code / library", "why": reason,
             "details": "Published %s. %s" % (text(data.get("published_at")), text(data.get("body"), 1000)),
             "importable": False}]


def discover(root, source, token="", force=False, fetch=None):
    """Return a daily cache or refresh independent feeds, keeping stale items on errors."""
    previous = cached(root, source)
    if (not force and previous.get("schema") == 2
            and 0 <= time.time() - previous["checked"] < TTL and not previous.get("errors")):
        return previous
    fetch = fetch or get_json
    queries = ("FLUX", "Z-Image", "Qwen-Image") if source == "huggingface" else ("LORA", "Checkpoint")
    items, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(model_feed, source, token, q, fetch): q for q in queries}
        futures.update({pool.submit(code_feed, repo, reason, fetch): repo for repo, reason in CODE.items()})
        for future in as_completed(futures):
            try:
                items.extend(future.result())
            except Exception as error:
                # Do not expose exception URLs or request credentials.
                why = str(error) if isinstance(error, DiscoveryError) else "Could not read this feed."
                errors.append(futures[future] + ": " + why)
    unique = {item["url"]: item for item in items}
    if errors:
        for item in previous["items"]:
            if isinstance(item, dict) and item.get("url") not in unique:
                unique[item["url"]] = dict(item, stale=True)
    result = {"schema": 2, "items": sorted(unique.values(), key=lambda item: (item["kind"], item["title"])),
              "checked": time.time(), "errors": sorted(errors)}
    tmp = None
    try:
        os.makedirs(root, exist_ok=True)
        target = cache_path(root, source)
        # Distinct temp file for simultaneous source windows.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as stream:
            tmp = stream.name
            json.dump(result, stream, indent=2)
        os.replace(tmp, target)
    except OSError:
        result["errors"].append("Could not save the discovery cache; results are available this session.")
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return result
