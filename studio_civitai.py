#!/usr/bin/env python3
"""
studio_civitai - a LoRA profile from a CivitAI link or a .safetensors file.

The Image Studio's LoRA library is filled by hand or by a backend scan that
can only guess from a filename. CivitAI already knows the rest: the trigger
words, the base model it was trained for, a preview, a description. This
module turns either thing the user has in hand into a library record:

- **A link** (a model page, a `modelVersionId`, a download link, an AIR
  `urn:air:...:civitai:<model>@<version>`, or a bare version number) is read
  from CivitAI's public API (`/api/v1/model-versions/<id>`, `/models/<id>`).
- **A file** is read for the metadata trainers write into the safetensors
  header (kohya's `ss_*`, the `modelspec.*` keys), then hashed and looked up
  on CivitAI by SHA-256, which is how CivitAI itself names files. Offline, or
  for a file CivitAI has never seen, the header is the profile.

Getting the file onto a backend is the other half: `download()` fetches a
version's file into a backend's LoRA folder (streamed to `.part`, checked
against CivitAI's SHA-256, then renamed), and `copy_into()` places a picked
file there. Only a folder on this PC can be written; the other machine's
disk is never assumed (AGENTS.md: no path is assumed on the other machine).

No tkinter, stdlib only, and every URL goes through `Client.opener` so the
tests swap it for a table of answers.
"""

import hashlib
import html
import json
import os
import re
import shutil
import struct
import urllib.error
import urllib.parse
import urllib.request

API = "https://civitai.com/api/v1"
SITE = "https://civitai.com"
TIMEOUT = 20
CHUNK = 1 << 20
NOTES_MAX = 1200              # chars of the model's description kept in notes
HEADER_MAX = 100 << 20        # a safetensors header bigger than this is not one

# CivitAI's `baseModel` strings, to the studio's families. Matched on the
# lowercased name, first hit wins, so the specific ones come first.
BASE_FAMILIES = [
    ("kontext", "flux1-kontext"),
    ("flux.2", "flux2"), ("flux 2", "flux2"), ("flux2", "flux2"),
    ("flux", "flux1"),
    ("z-image", "z-image"), ("zimage", "z-image"), ("z image", "z-image"),
    ("krea 2", "krea2"), ("krea2", "krea2"),
    ("qwen", "qwen-image"),
    ("sdxl", "sdxl"), ("pony", "sdxl"), ("illustrious", "sdxl"), ("noobai", "sdxl"),
    ("sd 1.5", "sd15"), ("sd1.5", "sd15"), ("sd 1.4", "sd15"),
]

# What trainers write into the header about the base model.
HEADER_FAMILIES = [
    ("kontext", "flux1-kontext"), ("flux-2", "flux2"), ("flux2", "flux2"),
    ("flux", "flux1"), ("qwen", "qwen-image"), ("z-image", "z-image"),
    ("z_image", "z-image"), ("sdxl", "sdxl"), ("stable-diffusion-xl", "sdxl"),
    ("sd_v1", "sd15"), ("stable-diffusion-v1", "sd15"),
]

# CivitAI tags to the library's categories, first hit wins.
TAG_CATEGORIES = [
    (("celebrity", "person", "real person", "influencer", "actor", "actress"), "Identity"),
    (("photography", "film", "analog", "polaroid", "camera", "cinematic", "lens",
      "film grain"), "Camera / Film"),
    (("clothing", "outfit", "costume", "dress", "fashion"), "Clothing"),
    (("character", "anime character", "game character", "video game"), "Character"),
    (("tool", "detail", "enhancer", "slider", "quality", "hands", "lightning",
      "turbo"), "Detail / Enhancement"),
    (("style", "art style", "artstyle", "painting", "illustration", "concept art",
      "artist"), "Style"),
]


class CivitAIError(Exception):
    """Something the user should read: what failed, in words."""


# =================================================================== links

def parse_link(text):
    """What a pasted thing names on CivitAI -> {"model": int|None,
    "version": int|None}, or None when it names nothing there.

        https://civitai.com/models/12345/some-name?modelVersionId=67890
        https://civitai.com/models/12345                 (the latest version)
        https://civitai.com/api/download/models/67890
        urn:air:flux1:lora:civitai:12345@67890
        67890                                            (a version id)
    """
    t = (text or "").strip().strip("<>\"'")
    if not t:
        return None
    if t.isdigit():
        return {"model": None, "version": int(t)}
    m = re.match(r"^urn:air:[^:]+:[^:]+:civitai:(\d+)(?:@(\d+))?", t, re.I)
    if m:
        return {"model": int(m.group(1)),
                "version": int(m.group(2)) if m.group(2) else None}
    if "://" not in t:
        t = "https://" + t
    u = urllib.parse.urlparse(t)
    host = (u.hostname or "").lower()
    if not (host == "civitai.com" or host.endswith(".civitai.com")
            or re.match(r"^(www\.)?civitai\.[a-z]+$", host)):
        return None
    query = urllib.parse.parse_qs(u.query)
    version = None
    for v in query.get("modelVersionId", []) + query.get("version", []):
        if v.isdigit():
            version = int(v)
    m = re.search(r"/api/download/models/(\d+)", u.path)
    if m:
        return {"model": None, "version": int(m.group(1))}
    m = re.search(r"/model-versions/(\d+)", u.path)
    if m:
        return {"model": None, "version": int(m.group(1))}
    m = re.search(r"/models/(\d+)", u.path)
    if m:
        return {"model": int(m.group(1)), "version": version}
    if version is not None:
        return {"model": None, "version": version}
    return None


def page_url(model_id, version_id=None):
    url = "%s/models/%s" % (SITE, model_id)
    return url + ("?modelVersionId=%s" % version_id if version_id else "")


# ================================================================ the API

class Client:
    """CivitAI's public API. `token` is an API key from the user's CivitAI
    account settings; reading needs none, but many downloads do."""

    def __init__(self, token="", opener=None, timeout=TIMEOUT):
        self.token = (token or "").strip()
        self.opener = opener or urllib.request.urlopen     # read late: tests swap it
        self.timeout = timeout

    def _request(self, url):
        req = urllib.request.Request(url, headers={
            "User-Agent": "StudioAssist/1 (+LoRA library import)",
            "Accept": "application/json"})
        if self.token:
            req.add_header("Authorization", "Bearer " + self.token)
        return req

    def get(self, path):
        url = path if path.startswith("http") else API + path
        try:
            with self.opener(self._request(url), timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise CivitAIError("CivitAI has nothing at %s." % url) from e
            if e.code in (401, 403):
                raise CivitAIError("CivitAI refused %s (HTTP %d): it needs an API key, "
                                   "or a different one." % (url, e.code)) from e
            raise CivitAIError("CivitAI answered HTTP %d for %s." % (e.code, url)) from e
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise CivitAIError("Could not reach CivitAI (%s)." % reason) from e
        except ValueError as e:
            raise CivitAIError("CivitAI's answer for %s was not JSON." % url) from e

    def version(self, version_id):
        return self.get("/model-versions/%d" % int(version_id))

    def model(self, model_id):
        return self.get("/models/%d" % int(model_id))

    def by_hash(self, sha256):
        """The version whose file has this hash, or None if CivitAI has
        never seen it."""
        try:
            return self.get("/model-versions/by-hash/%s" % sha256.upper())
        except CivitAIError as e:
            if isinstance(e.__cause__, urllib.error.HTTPError) and e.__cause__.code == 404:
                return None
            raise

    def lookup(self, link):
        """(version, model) for a parsed link. A model link with no version
        is its newest version, which is what the page shows first."""
        ref = parse_link(link) if isinstance(link, str) else link
        if ref is None:
            raise CivitAIError("That does not look like a CivitAI link: %s" % link)
        model = None
        if ref["version"] is None:
            model = self.model(ref["model"])
            versions = model.get("modelVersions") or []
            if not versions:
                raise CivitAIError("CivitAI model %s has no versions." % ref["model"])
            vid = versions[0].get("id")
        else:
            vid = ref["version"]
        version = self.version(vid)
        mid = version.get("modelId") or ref["model"]
        if model is None and mid:
            try:
                model = self.model(mid)          # the tags and the description
            except CivitAIError:
                model = None
        return version, model


# =========================================================== the profile

def family_of(base_model):
    b = (base_model or "").lower()
    for needle, fam in BASE_FAMILIES:
        if needle in b:
            return fam
    return ""


def category_of(tags):
    tags = [str(t).lower() for t in tags or ()]
    for words, cat in TAG_CATEGORIES:
        if any(t in words for t in tags):
            return cat
    return ""


def plain_text(markup):
    """CivitAI's HTML description as text a notes box can hold."""
    t = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", markup or "")
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t).strip()
    return t if len(t) <= NOTES_MAX else t[:NOTES_MAX].rsplit(" ", 1)[0] + " …"


def primary_file(version):
    """The version's model file: the one marked primary, else the first
    safetensors, else the first file."""
    files = [f for f in version.get("files") or [] if isinstance(f, dict)]
    for pick in (lambda f: f.get("primary"),
                 lambda f: str(f.get("name", "")).endswith(".safetensors"),
                 lambda f: True):
        for f in files:
            if pick(f) and f.get("name"):
                return f
    return None


def preview_image(version):
    """The URL of the version's tamest still picture, or ""."""
    best = None
    for img in version.get("images") or []:
        if not isinstance(img, dict) or not img.get("url"):
            continue
        if img.get("type", "image") != "image":
            continue
        level = img.get("nsfwLevel", 0)
        if not isinstance(level, int):
            level = 99 if img.get("nsfw") not in (None, False, "None") else 0
        if best is None or level < best[0]:
            best = (level, img["url"])
    return best[1] if best else ""


def profile(version, model=None):
    """A LoRA record (what `studio_imagegen.clean_lora` takes) from a
    version and, when there is one, its model. Pure."""
    model = model or version.get("model") or {}
    f = primary_file(version) or {}
    mid = version.get("modelId") or model.get("id")
    vid = version.get("id")
    model_name = (model.get("name") or "").strip()
    vname = (version.get("name") or "").strip()
    name = model_name or vname or f.get("name", "")
    if model_name and vname and vname.lower() not in ("v1", "v1.0", "1.0", model_name.lower()):
        name = "%s (%s)" % (model_name, vname)
    words = [w.strip().strip(",") for w in version.get("trainedWords") or []
             if isinstance(w, str) and w.strip()]
    source = page_url(mid, vid) if mid else ("%s/model-versions/%s" % (SITE, vid))
    lines = ["From CivitAI: " + source]
    if version.get("baseModel"):
        lines.append("Base model: %s" % version["baseModel"])
    creator = model.get("creator")
    if isinstance(creator, dict) and creator.get("username"):
        lines.append("By %s" % creator["username"])
    about = plain_text(version.get("description") or model.get("description") or "")
    if about:
        lines += ["", about]
    sha = ((f.get("hashes") or {}).get("SHA256") or "").lower()
    return {
        "file": f.get("name", ""),
        "name": name,
        "category": category_of(model.get("tags")),
        "trigger": ", ".join(dict.fromkeys(words)),
        "family": family_of(version.get("baseModel")),
        "notes": "\n".join(lines),
        "source": source,
        "sha256": sha,
        "_download": version.get("downloadUrl") or f.get("downloadUrl") or (
            "%s/api/download/models/%s" % (SITE, vid) if vid else ""),
        "_preview_url": preview_image(version),
        "_size": int(float(f.get("sizeKB") or 0) * 1024),
    }


# ============================================================== the file

def read_header(path):
    """The `__metadata__` a safetensors file carries (strings to strings),
    or {} for a file with none. Raises CivitAIError for a file that is not
    safetensors at all."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise CivitAIError("%s is too short to be a safetensors file."
                                   % os.path.basename(path))
            (n,) = struct.unpack("<Q", raw)
            if n <= 1 or n > HEADER_MAX:
                raise CivitAIError("%s is not a safetensors file." % os.path.basename(path))
            head = json.loads(f.read(n).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise CivitAIError("Could not read %s as safetensors (%s)."
                           % (os.path.basename(path), e)) from e
    meta = head.get("__metadata__") if isinstance(head, dict) else None
    return {str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else {}


def sha256_of(path, progress=None):
    h, done = hashlib.sha256(), 0
    total = os.path.getsize(path)
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            done += len(block)
            if progress:
                progress(done, total)
    return h.hexdigest()


def profile_from_header(meta, filename):
    """What a file's own metadata says about it. Only what the trainer wrote
    as a trigger is a trigger: the tag counts are captions, not triggers."""
    arch = " ".join(meta.get(k, "") for k in (
        "modelspec.architecture", "ss_base_model_version", "ss_sd_model_name",
        "ss_base_model")).lower()
    family = next((fam for needle, fam in HEADER_FAMILIES if needle in arch), "")
    trigger = meta.get("modelspec.trigger_phrase") or meta.get("ss_trigger") or ""
    name = meta.get("modelspec.title") or meta.get("ss_output_name") or ""
    notes = []
    if meta.get("modelspec.description"):
        notes.append(plain_text(meta["modelspec.description"]))
    if meta.get("modelspec.author"):
        notes.append("By " + meta["modelspec.author"])
    return {"file": filename, "name": name.strip(), "trigger": trigger.strip(),
            "family": family, "category": "", "notes": "\n".join(notes)}


# ============================================================ downloading

def _open_download(client, url):
    req = client._request(url)
    req.add_header("Accept", "*/*")
    try:
        return client.opener(req, timeout=client.timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise CivitAIError("CivitAI wants an API key for this download (HTTP %d). "
                               "Make one under your CivitAI account settings and paste "
                               "it into the import window." % e.code) from e
        raise CivitAIError("The download failed: HTTP %d from %s." % (e.code, url)) from e
    except (urllib.error.URLError, OSError) as e:
        raise CivitAIError("Could not reach CivitAI to download (%s)."
                           % getattr(e, "reason", e)) from e


def fetch_preview(client, url, folder, stem):
    """A version's preview picture saved under `folder` -> its path, or ""
    when there is none or it will not come. Best effort: a profile is worth
    having without its picture."""
    if not url:
        return ""
    try:
        with _open_download(client, url) as r:
            data = r.read()
            kind = (r.headers.get("Content-Type") or "") if hasattr(r, "headers") else ""
    except (CivitAIError, OSError):
        return ""
    ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else (
        ".gif" if data[:4] == b"GIF8" else ".webp" if data[8:12] == b"WEBP" else
        ".jpg" if data[:2] == b"\xff\xd8" else
        os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or
        (".png" if "png" in kind else ".jpg"))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, stem + ext)
    with open(path, "wb") as f:
        f.write(data)
    return path


def download(client, url, folder, filename, sha256="", progress=None, stop=None):
    """A version's file into `folder` as `filename` -> its path. Streamed to
    `<name>.part` and renamed only once whole (and matching CivitAI's
    SHA-256 when there is one), so a ComfyUI listing the folder never sees
    half a LoRA. A file already there with the right hash is kept."""
    dest = os.path.join(folder, filename)
    if os.path.isfile(dest) and (not sha256 or sha256_of(dest) == sha256.lower()):
        return dest
    if os.path.exists(dest):
        raise CivitAIError("%s is already in %s and is a different file; move it aside "
                           "first." % (filename, folder))
    os.makedirs(folder, exist_ok=True)
    part = dest + ".part"
    h, done = hashlib.sha256(), 0
    try:
        with _open_download(client, url) as r, open(part, "wb") as out:
            ctype = (r.headers.get("Content-Type") or "") if hasattr(r, "headers") else ""
            if "text/html" in ctype or "application/json" in ctype:
                raise CivitAIError("CivitAI sent a page instead of the file; it usually "
                                   "means the download needs an API key.")
            total = int((r.headers.get("Content-Length") if hasattr(r, "headers") else 0)
                        or 0)
            while True:
                if stop is not None and stop.is_set():
                    raise CivitAIError("Download cancelled.")
                block = r.read(CHUNK)
                if not block:
                    break
                out.write(block)
                h.update(block)
                done += len(block)
                if progress:
                    progress(done, total)
        if sha256 and h.hexdigest() != sha256.lower():
            raise CivitAIError("%s arrived damaged (its SHA-256 is not CivitAI's); "
                               "nothing was kept." % filename)
        os.replace(part, dest)
    except BaseException:
        try:
            os.remove(part)
        except OSError:
            pass
        raise
    return dest


def copy_into(path, folder):
    """A picked file placed in a LoRA folder -> the name ComfyUI will list it
    by (relative to the folder). A file already inside it is not copied."""
    folder_abs = os.path.abspath(folder)
    src = os.path.abspath(path)
    if os.path.commonpath([folder_abs, src]) == folder_abs:
        return os.path.relpath(src, folder_abs)
    name = os.path.basename(src)
    dest = os.path.join(folder_abs, name)
    if os.path.exists(dest):
        if os.path.getsize(dest) == os.path.getsize(src) and \
                sha256_of(dest) == sha256_of(src):
            return name
        raise CivitAIError("%s is already in %s and is a different file; rename one of "
                           "them first." % (name, folder))
    os.makedirs(folder_abs, exist_ok=True)
    shutil.copyfile(src, dest + ".part")
    os.replace(dest + ".part", dest)
    return name


# ============================================================== importing
# The two things the import window does, each ending in a saved library
# record. `lib` is a `studio_imagegen.Library`; `say(text)` reports progress
# from the worker thread (the window posts it back through its queue).

def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "lora"


def _with_preview(lib, client, rec, url):
    if rec.get("preview") and os.path.isfile(rec["preview"]):
        return
    path = fetch_preview(client, url, lib.preview_dir(), _slug(rec["id"]))
    if path:
        rec["preview"] = path


def import_link(lib, client, link, folder="", say=lambda text: None, stop=None):
    """A CivitAI link -> (record, added). With `folder` (a backend's LoRA
    folder on this PC) the file is downloaded there too."""
    say("Reading %s from CivitAI" % link.strip())
    version, model = client.lookup(link)
    p = profile(version, model)
    if not p["file"]:
        raise CivitAIError("CivitAI lists no file for %s." % link.strip())
    if model and str(model.get("type", "LORA")).upper() not in ("LORA", "LOCON", "DORA",
                                                              "LYCORIS"):
        say("Note: CivitAI calls %s a %s, not a LoRA." % (p["name"], model.get("type")))
    if folder:
        mb = p["_size"] / float(1 << 20)

        def progress(done, total):
            total = total or p["_size"]
            say("Downloading %s: %d of %s MB" % (
                p["file"], done >> 20, "%d" % (total >> 20) if total else "%.0f" % mb))
        download(client, p["_download"], folder, p["file"], p["sha256"], progress, stop)
    rec, added = lib.import_lora(p)
    _with_preview(lib, client, rec, p["_preview_url"])
    lib.save("loras")
    return rec, added


def import_file(lib, client, path, folder="", lora_dirs=(), lookup=True,
                say=lambda text: None):
    """A .safetensors file -> (record, added). With `folder` the file is
    copied there first; a file already inside one of `lora_dirs` is named
    as ComfyUI lists it. The header is read, the file hashed and, with
    `lookup`, found on CivitAI by that hash; what CivitAI knows wins over
    what the header says, and the header fills what CivitAI left empty."""
    base = os.path.basename(path)
    meta = read_header(path)
    name = base
    if folder:
        say("Copying %s into %s" % (base, folder))
        name = copy_into(path, folder)
    else:
        src = os.path.abspath(path)
        for d in lora_dirs:
            d = os.path.abspath(d) if d else ""
            if d and os.path.isdir(d) and os.path.commonpath([d, src]) == d:
                name = os.path.relpath(src, d)
                break
    p = profile_from_header(meta, name)
    say("Hashing %s" % base)
    p["sha256"] = sha256_of(path)
    preview = ""
    if lookup:
        say("Looking %s up on CivitAI" % base)
        try:
            version = client.by_hash(p["sha256"])
            if version is None:
                say("CivitAI does not know %s; using what the file says." % base)
            else:
                model = None
                if version.get("modelId"):
                    try:
                        model = client.model(version["modelId"])
                    except CivitAIError:
                        model = None
                online = profile(version, model)
                preview = online["_preview_url"]
                for k in ("name", "category", "trigger", "family", "notes", "source"):
                    if online.get(k):
                        p[k] = online[k]
        except CivitAIError as e:
            say("Not looked up on CivitAI (%s); using what the file says." % e)
    rec, added = lib.import_lora(p)
    _with_preview(lib, client, rec, preview)
    lib.save("loras")
    return rec, added


# ================================================================ the key

TOKEN_ENV = "CIVITAI_API_KEY"


def token_path(root):
    return os.path.join(root, "civitai.json")


def load_token(root):
    """The API key: the environment's, else the one the import window kept."""
    if os.environ.get(TOKEN_ENV):
        return os.environ[TOKEN_ENV].strip()
    try:
        with open(token_path(root), encoding="utf-8") as f:
            d = json.load(f)
        return str(d.get("token", "")).strip() if isinstance(d, dict) else ""
    except (OSError, ValueError):
        return ""


def save_token(root, token):
    os.makedirs(root, exist_ok=True)
    path = token_path(root)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"token": (token or "").strip()}, f)
    os.replace(tmp, path)
