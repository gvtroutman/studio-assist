#!/usr/bin/env python3
"""
studio_hub - the Image Studio's Add-ons beyond CivitAI: Hugging Face LoRAs
for a model, and ComfyUI custom-node plugins from GitHub. No tkinter (the
window is `studio_images_ui.AddonsWindow`, tabs "Hugging Face" and
"GitHub"); stdlib only.

- **Hugging Face** is `/api/models` filtered to adapters of the model's own
  base repos (`FAMILY_REPOS`), by downloads, each card with its license
  group (`hf_license`) so the window can list them by license. Install
  picks the repo's top-level `.safetensors` (the largest when there are
  several), downloads it
  into a backend's LoRA folder checked against the Hub's LFS SHA-256, and
  files it through the importer's `import_file`, so the record is the same
  kind CivitAI's install makes.
- **GitHub** is repository search on the `comfyui-nodes` topic, by stars.
  Install unpacks the default branch's zip into `<ComfyUI>/custom_nodes/
  <repo>`: every path is confined to that folder, an existing folder is never
  overwritten, and nothing is run - its Python requirements are the user's
  to install, and ComfyUI loads it on restart. It is third-party code; the
  window says so before installing.

The token of a service goes only to that service's own origin: it is sent
as an unredirected header, so the Hub's CDN redirect never carries it.
"""

import hashlib
import io
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import apps.image_studio.addons.catalog as catalog
import apps.image_studio.addons.civitai as civitai
import apps.image_studio.imagegen as ig

HF = "https://huggingface.co"
GH_API = "https://api.github.com"
TIMEOUT = 20
CHUNK = 1 << 20
MAX_JSON = 4 << 20
MAX_ZIP = 200 << 20
PAGE = 20
ABOUT_MAX = 280
GH_TOPIC = "comfyui-nodes"

# The base repos a family's LoRAs name as `base_model` on the Hub.
FAMILY_REPOS = {
    "flux1": ("black-forest-labs/FLUX.1-dev",),
    "flux1-kontext": ("black-forest-labs/FLUX.1-Kontext-dev",),
    "flux2": ("black-forest-labs/FLUX.2-dev",),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0",),
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),
    "z-image": ("Tongyi-MAI/Z-Image-Turbo",),
    "qwen-image": ("Qwen/Qwen-Image",),
}
REPO_ID = re.compile(r"[\w.-]+/[\w.-]+")


class HubError(Exception):
    """Said to the user as it is."""


def _open(url, token="", accept="application/json", opener=None):
    req = urllib.request.Request(url, headers={
        "User-Agent": "StudioAssist/1 (+Add-ons)", "Accept": accept})
    if token:
        req.add_unredirected_header("Authorization", "Bearer " + token)
    name = "GitHub" if "github" in url else "Hugging Face"
    try:
        return (opener or urllib.request.urlopen)(req, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise HubError("%s refused the request (HTTP %d): a rate limit, or it needs "
                           "a token." % (name, e.code)) from None
        if e.code == 404:
            raise HubError("%s has nothing there (HTTP 404)." % name) from None
        raise HubError("%s answered HTTP %d." % (name, e.code)) from None
    except (urllib.error.URLError, OSError) as e:
        raise HubError("Could not reach %s (%s)." % (name, getattr(e, "reason", e))) from None


def get_json(url, token="", opener=None):
    with _open(url, token, opener=opener) as r:
        data = r.read(MAX_JSON + 1)
    if len(data) > MAX_JSON:
        raise HubError("The answer was too large.")
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError:
        raise HubError("The answer was not JSON.") from None


def _text(value, limit=ABOUT_MAX):
    return " ".join(str(value or "").split())[:limit]


# ========================================================== Hugging Face

# Hub license ids by what they let the studio do with a LoRA's pictures
# (catalog.LICENSE_GROUPS). Prefixes; a "-nc" anywhere wins over them.
FREE_LICENSES = ("apache", "mit", "bsd", "cc0", "cc-by", "openrail", "creativeml-openrail",
                 "bigscience-openrail", "unlicense", "gpl", "lgpl", "agpl", "mpl", "isc",
                 "artistic", "wtfpl", "afl", "ecl", "epl", "zlib", "ofl", "postgresql")
# Named non-commercial for the weights, but its terms let the pictures be used
# commercially: own terms, not "non-commercial only".
OWN_TERMS = ("flux-1-dev-non-commercial-license",)


def hf_license(license_id, license_name=""):
    """(group, words) for a Hub license id ("other" + the card's
    `license_name` for a custom one)."""
    lid = str(license_id or "").strip().lower()
    name = str(license_name or "").strip().lower()
    shown = name if lid == "other" and name else lid
    if not shown:
        return "unstated", ""
    if shown in OWN_TERMS:
        return "custom", shown
    if re.search(r"(^|-)nc(-|$)|non-?commercial|research-only", shown):
        return "noncommercial", shown
    if lid != "other" and lid.startswith(FREE_LICENSES):
        return "commercial", shown
    return "custom", shown


def repos_for(model):
    return [r for fam in sorted(ig.model_families(model)) for r in FAMILY_REPOS.get(fam, ())]


def hf_search(model, query="", token="", opener=None):
    """Hugging Face LoRAs for `model` -> cards, most downloaded first (the
    window groups them by license, `catalog.by_license`)."""
    seen, cards = set(), []
    for base in repos_for(model):
        url = HF + "/api/models?" + urllib.parse.urlencode({
            "filter": "base_model:adapter:" + base, "search": query.strip(),
            "sort": "downloads", "direction": -1, "limit": PAGE, "cardData": "true"})
        rows = get_json(url, token, opener)
        if not isinstance(rows, list):
            raise HubError("Hugging Face returned an unexpected model list.")
        for row in rows:
            rid = row.get("id") or row.get("modelId") or ""
            if not REPO_ID.fullmatch(rid) or rid in seen:
                continue
            seen.add(rid)
            tags = [t for t in row.get("tags") or [] if isinstance(t, str)]
            meta = row.get("cardData") if isinstance(row.get("cardData"), dict) else {}
            lid = meta.get("license") if isinstance(meta.get("license"), str) else next(
                (t[8:] for t in tags if t.startswith("license:")), "")
            group, terms = hf_license(lid, meta.get("license_name") or "")
            cards.append({
                "kind": "hf", "id": rid, "name": rid.split("/", 1)[1],
                "creator": rid.split("/", 1)[0], "base": base,
                "downloads": int(row.get("downloads") or 0),
                "likes": int(row.get("likes") or 0),
                "license": terms, "license_group": group,
                "link": HF + "/" + rid})
    cards.sort(key=lambda c: -c["downloads"])
    return cards


def hf_pick_file(info):
    """The LoRA file of a repo's `/api/models/<id>?blobs=true` answer ->
    (filename, size, sha256), or None when it has no top-level .safetensors."""
    files = []
    for s in info.get("siblings") or []:
        name = s.get("rfilename") or ""
        if not name.lower().endswith(".safetensors"):
            continue
        try:
            civitai.own_name(name)     # top level only, and nothing that leaves the folder
        except civitai.CivitAIError:
            continue
        lfs = s.get("lfs") or {}
        files.append((name, int(s.get("size") or lfs.get("size") or 0),
                      str(lfs.get("sha256") or "").lower()))
    return max(files, key=lambda f: f[1]) if files else None


def hf_install(lib, card, folder, family="", token="", say=lambda text: None,
               stop=None, opener=None):
    """A Hugging Face LoRA downloaded into `folder` and filed in the library
    -> (record, added)."""
    rid = card["id"]
    if not REPO_ID.fullmatch(rid):
        raise HubError("Not a Hugging Face repo: %s" % rid)
    say("Reading %s from Hugging Face" % rid)
    info = get_json(HF + "/api/models/%s?blobs=true" % rid, token, opener)
    picked = hf_pick_file(info if isinstance(info, dict) else {})
    if picked is None:
        raise HubError("%s has no .safetensors file at its top level." % rid)
    name, size, sha = picked
    stem, ext = os.path.splitext(name)
    filename = name
    here = os.path.join(folder, name)
    if os.path.isfile(here) and not (sha and civitai.sha256_of(here) == sha):
        filename = "%s-%s%s" % (rid.split("/")[1], stem, ext)   # a different file's name
    dest = os.path.join(folder, filename)
    url = HF + "/%s/resolve/%s/%s" % (rid, info.get("sha") or "main",
                                      urllib.parse.quote(name))
    _download(url, dest, sha, size, token, say, stop, opener)
    rec, added = civitai.import_file(lib, None, dest, lora_dirs=[folder], lookup=False,
                                     say=say)
    if not rec.get("source"):
        rec["source"] = card["link"]
    if not rec.get("family") and family:
        rec["family"] = family
    if rec.get("name") in ("", filename, stem):
        rec["name"] = card["name"]
    lib.save("loras")
    return rec, added


def _download(url, dest, sha256, size, token, say, stop, opener):
    if os.path.isfile(dest):
        if sha256 and civitai.sha256_of(dest) == sha256:
            return dest
        raise HubError("%s is already there and is a different file; move it aside "
                       "first." % os.path.basename(dest))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part, h, done = dest + ".part", hashlib.sha256(), 0
    try:
        with _open(url, token, "*/*", opener) as r, open(part, "wb") as out:
            total = civitai.promised(r)
            while True:
                if stop is not None and stop.is_set():
                    raise HubError("Download cancelled.")
                block = r.read(CHUNK)
                if not block:
                    break
                out.write(block)
                h.update(block)
                done += len(block)
                say("Downloading %s: %d of %d MB" % (os.path.basename(dest), done >> 20,
                                                      max(size, done) >> 20))
        # The Hub's size is exact; a stream cut off ends like a whole one.
        short = civitai.cut_short(done, total, size, slack=0)
        if short:
            raise HubError("The download of %s was cut off (%s); nothing was kept. "
                           "Try again." % (os.path.basename(dest), short))
        if sha256 and h.hexdigest() != sha256:
            raise HubError("%s arrived damaged (its SHA-256 is not the Hub's); nothing "
                           "was kept." % os.path.basename(dest))
        os.replace(part, dest)
    except BaseException:
        try:
            os.remove(part)
        except OSError:
            pass
        raise
    return dest


# ================================================================ GitHub

def gh_search(query="", page=1, token="", opener=None):
    """ComfyUI custom-node repos -> (cards, next page or 0), most starred."""
    q = ("%s topic:%s" % (query.strip(), GH_TOPIC)).strip()
    url = GH_API + "/search/repositories?" + urllib.parse.urlencode({
        "q": q, "sort": "stars", "order": "desc", "per_page": PAGE, "page": page})
    data = get_json(url, token, opener)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise HubError("GitHub returned an unexpected repository list.")
    cards = []
    for row in data["items"]:
        full = row.get("full_name") or ""
        if not REPO_ID.fullmatch(full) or row.get("archived"):
            continue
        cards.append({
            "kind": "github", "id": full, "name": full.split("/", 1)[1],
            "creator": full.split("/", 1)[0], "about": _text(row.get("description")),
            "stars": int(row.get("stargazers_count") or 0),
            "updated": str(row.get("pushed_at") or "")[:10],
            "branch": row.get("default_branch") or "main",
            "license": ((row.get("license") or {}).get("spdx_id") or ""),
            "link": "https://github.com/" + full})
    more = page + 1 if len(data["items"]) == PAGE and page * PAGE < min(
        int(data.get("total_count") or 0), 1000) else 0
    return cards, more


def custom_nodes(comfy_folder):
    """`<ComfyUI>/custom_nodes`, once `comfy_folder` is shown to be ComfyUI."""
    root = os.path.abspath(comfy_folder or "")
    if not (os.path.isfile(os.path.join(root, "main.py")) and
            os.path.isfile(os.path.join(root, "folder_paths.py"))):
        raise HubError("Choose the ComfyUI folder containing main.py and folder_paths.py.")
    return os.path.join(root, "custom_nodes")


def gh_installed(comfy_folder, card):
    try:
        return os.path.isdir(os.path.join(custom_nodes(comfy_folder), card["name"]))
    except HubError:
        return False


def gh_install(card, comfy_folder, token="", say=lambda text: None, opener=None):
    """A repo's default branch unpacked into custom_nodes/<repo> -> its path.
    Never overwrites; every member is confined to the new folder."""
    full, branch = card["id"], card.get("branch") or "main"
    if not REPO_ID.fullmatch(full) or not re.fullmatch(r"[\w./-]+", branch) or ".." in branch:
        raise HubError("Not a GitHub repository: %s" % full)
    nodes = custom_nodes(comfy_folder)
    dest = os.path.join(nodes, card["name"])
    if os.path.exists(dest):
        raise HubError("%s is already in custom_nodes; remove it first to reinstall."
                       % card["name"])
    say("Downloading %s" % full)
    url = GH_API + "/repos/%s/zipball/%s" % (full, urllib.parse.quote(branch, safe=""))
    with _open(url, token, "application/vnd.github+json", opener) as r:
        data = r.read(MAX_ZIP + 1)
    if len(data) > MAX_ZIP:
        raise HubError("%s is larger than %d MB; install it by hand." % (full, MAX_ZIP >> 20))
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise HubError("GitHub did not send a zip for %s." % full) from None
    tmp = dest + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        _unpack(zf, tmp)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def _unpack(zf, target):
    """A GitHub zipball (one top folder) into `target`, confined to it."""
    target = os.path.abspath(target)
    os.makedirs(target)
    total = 0
    for info in zf.infolist():
        parts = info.filename.replace("\\", "/").split("/")[1:]
        if not parts or not any(parts) or info.filename.endswith("/"):
            continue
        if any(p in ("", ".", "..") for p in parts) or ":" in info.filename:
            raise HubError("The zip has an unsafe path: %s" % info.filename)
        # Symlinks come as small files holding a path; skip them.
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            continue
        out = os.path.abspath(os.path.join(target, *parts))
        if os.path.commonpath([target, out]) != target:
            raise HubError("The zip has an unsafe path: %s" % info.filename)
        total += info.file_size
        if total > MAX_ZIP * 3:
            raise HubError("The unpacked plugin is too large.")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with zf.open(info) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)


# ======================================================= remembered folder

def _conf(root):
    return os.path.join(root, "comfyui-folder.json")


def load_comfy_folder(root):
    try:
        with open(_conf(root), encoding="utf-8") as f:
            value = json.load(f).get("folder", "")
        return value if isinstance(value, str) else ""
    except (OSError, ValueError, AttributeError):
        return ""


def save_comfy_folder(root, folder):
    os.makedirs(root, exist_ok=True)
    with open(_conf(root) + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"folder": folder}, f)
    os.replace(_conf(root) + ".tmp", _conf(root))
