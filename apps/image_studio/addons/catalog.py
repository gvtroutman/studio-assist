#!/usr/bin/env python3
"""
studio_catalog - the Image Studio's Add-ons: LoRAs sorted by the models
they work with, a CivitAI catalog per model, and uninstalling.

A plugin catalog in the Jellyfin sense, but the "server version" a plugin
must match is the model: a LoRA is only offered with a model whose family it
suits (`studio_imagegen.lora_fits`). The window is `studio_images_ui.
AddonsWindow`; this module is its engine and has no tkinter.

- **Installed** is the LoRA library, split per model (`sorted_for`) into
  what fits and what has no family set, plus the LoRAs that fit none of the
  user's models (`fits_none`) - Qwen-Image-Edit's, say, which no form model
  can load.
- **Catalog** is CivitAI's search, filtered to the model's own `baseModel`
  names (`civitai.FAMILY_BASES`), one card per model for the version that
  suits it (`card`). Install is the LoRA importer's `import_link`: the file
  into a backend's LoRA folder on this PC, checked against CivitAI's hash.
- **Licenses**: every catalog card (CivitAI's and Hugging Face's) and every
  LoRA record carries a `license_group` (`imagegen.LICENSE_GROUPS`) and the
  window lists them grouped by it (`by_license`), most freely usable first,
  keeping the order they came in inside a group.
- **Thumbnails** are PNG, because Tk reads no JPEG and CivitAI serves JPEG.
  `to_png` converts through Windows' own System.Drawing in one PowerShell
  run per batch (stdlib only; a contained child, `studio_procs.spawn`).
  Only a picture CivitAI rates PG or PG-13 is ever shown (`safe_preview`).
- **Uninstall** sends the file to the Recycle Bin (`recycle`), never a hard
  delete, and removes the record. A LoRA whose file is only on another
  machine cannot be uninstalled from here; it is turned off instead, since
  a backend scan would bring its record straight back.
"""

import base64
import json
import os
import re
import subprocess
import sys
import tempfile

import apps.image_studio.addons.civitai as civitai
import apps.image_studio.imagegen as ig
import core.procs as procs

THUMB = 160                   # px on the long edge of a stored thumbnail
SAFE_LEVELS = (1, 2)          # CivitAI's nsfwLevel bits: PG, PG-13
ABOUT_MAX = 280               # chars of a card's description
CONVERT_TIMEOUT = 60


# ============================================================ installed

def _by_name(recs):
    return sorted(recs, key=lambda r: (not r.get("enabled", True), r["name"].lower()))


def sorted_for(lib, model):
    """The library's LoRAs for one model -> (fits, unknown), each enabled
    ones first, then by name. A LoRA for another family is in neither."""
    fits, unknown = [], []
    for rec in lib.all("loras"):
        ok = ig.lora_fits(rec, model)
        if ok:
            fits.append(rec)
        elif ok is None:
            unknown.append(rec)
    return _by_name(fits), _by_name(unknown)


def fits_none(lib):
    """LoRAs with a family that suits none of the library's models."""
    models = lib.all("models")
    return _by_name(r for r in lib.all("loras") if r.get("family")
                    and not any(ig.lora_fits(r, m) for m in models))


def where_installed(rec, backends, inventories):
    """(on, off): names of the backends whose LoRA list has this LoRA's file,
    and of those that answered without it. Backends not checked are in
    neither."""
    on, off = [], []
    for b in backends:
        inv = inventories.get(b["id"])
        if inv is None:
            continue
        (on if ig.lora_file(rec, b["id"]) in inv.get("loras", ()) else off).append(b["name"])
    return on, off


# ============================================================== catalog

def bases_for(model):
    """CivitAI `baseModel` names for a model's families, in a stable order."""
    out = []
    for fam in sorted(ig.model_families(model)):
        out += [b for b in civitai.FAMILY_BASES.get(fam, ()) if b not in out]
    return tuple(out)


def safe_preview(version, width=THUMB * 2):
    """The URL of the version's first still picture CivitAI rates PG or
    PG-13, asked for at `width` px, or "". Never a fallback to another."""
    for img in version.get("images") or []:
        if not isinstance(img, dict) or not img.get("url"):
            continue
        if img.get("type", "image") != "image" or img.get("nsfwLevel") not in SAFE_LEVELS:
            continue
        url = img["url"]
        if re.search(r"/(original=true|width=\d+)/", url):
            url = re.sub(r"/(original=true|width=\d+)/", "/width=%d/" % width, url, count=1)
        return url
    return ""


def by_license(cards):
    """Catalog cards or LoRA records in LICENSE_GROUPS order; a group keeps
    the order it came in."""
    return ig.by_license(cards, ig.lora_license)


def card(model, families):
    """One CivitAI search hit -> a catalog card for its newest version that
    suits one of `families`, or None when none does (CivitAI lists a model
    under every base model any of its versions was trained for)."""
    for version in model.get("modelVersions") or []:
        if not isinstance(version, dict):
            continue
        fam = civitai.family_of(version.get("baseModel"))
        if fam not in families:
            continue
        f = civitai.primary_file(version) or {}
        mid, vid = model.get("id"), version.get("id")
        if not (mid and vid and f.get("name")):
            continue
        creator = model.get("creator") if isinstance(model.get("creator"), dict) else {}
        stats = model.get("stats") if isinstance(model.get("stats"), dict) else {}
        about = civitai.plain_text(version.get("description") or model.get("description") or "")
        if len(about) > ABOUT_MAX:
            about = about[:ABOUT_MAX].rsplit(" ", 1)[0] + " …"
        words = [w.strip().strip(",") for w in version.get("trainedWords") or []
                 if isinstance(w, str) and w.strip()]
        group, terms = civitai.license_of(model)
        return {
            "model_id": mid, "version_id": vid,
            "name": (model.get("name") or f["name"]).strip(),
            "version": (version.get("name") or "").strip(),
            "creator": creator.get("username") or "",
            "base_model": version.get("baseModel") or "",
            "family": fam,
            "downloads": int(stats.get("downloadCount") or 0),
            "category": civitai.category_of(model.get("tags")) or "Other",
            "about": about.replace("\n", " "),
            "trigger": ", ".join(dict.fromkeys(words)),
            "file": f["name"],
            "size": int(float(f.get("sizeKB") or 0) * 1024),
            "sha256": ((f.get("hashes") or {}).get("SHA256") or "").lower(),
            "preview_url": safe_preview(version),
            "link": civitai.page_url(mid, vid),
            "download": version.get("downloadUrl") or f.get("downloadUrl") or "",
            "license_group": group,
            "license": terms,
        }
    return None


def search(client, model, query="", sort=civitai.SORTS[0], cursor=""):
    """One page of the catalog for `model` -> (cards, next cursor)."""
    bases = bases_for(model)
    if not bases:
        return [], ""
    families = ig.model_families(model)
    hits, nxt = client.search(bases, query, sort, cursor)
    cards = [c for c in (card(m, families) for m in hits) if c]
    return cards, nxt


# ============================================================ checkpoints
# Community checkpoints of the architectures whose pictures may be sold
# (imagegen.FAMILY_LICENSES "commercial"), keeping only the checkpoints whose
# own CivitAI license allows it too. Pony and Illustrious are SDXL underneath
# but carry their own licenses, so they are not searched.
OWN_LICENSE_BASES = ("Pony", "Illustrious")


def checkpoint_families():
    return [f for f, (g, _w) in ig.FAMILY_LICENSES.items()
            if g == "commercial" and civitai.FAMILY_BASES.get(f)]


def checkpoint_search(client, family, query="", sort=civitai.SORTS[0], cursor=""):
    """One page of commercially usable checkpoints of `family` -> (cards,
    next cursor, how many were left out for their license)."""
    bases = [b for b in civitai.FAMILY_BASES.get(family, ()) if b not in OWN_LICENSE_BASES]
    if not bases:
        return [], "", 0
    hits, nxt = client.search(bases, query, sort, cursor, types="Checkpoint")
    cards = [c for c in (card(m, {family}) for m in hits) if c]
    keep = [c for c in cards if c["license_group"] == "commercial"]
    return keep, nxt, len(cards) - len(keep)


def checkpoint_install(lib, client, c, folder, say=lambda t: None, stop=None):
    """A checkpoint downloaded into `folder` and added as a model -> the
    model record. It copies a model of the same family (workflow, encoders,
    defaults) with its main file swapped, so it needs one to copy."""
    like = next((m for m in lib.all("models") if m.get("family") == c["family"]), None) or \
        next((ig.clean_model(m) for m in ig._default_models()
              if m["family"] == c["family"]), None)
    if like is None:
        raise civitai.CivitAIError("No %s model in your library to base it on; add one in "
                                   "Models… first." % ig.FAMILIES.get(c["family"], c["family"]))

    def progress(done, total):
        say("Downloading %s: %d of %d MB" % (c["file"], done >> 20, max(total, done) >> 20))
    civitai.download(client, c["download"], folder, c["file"], c["sha256"], progress, stop)
    rec = dict(like, id=ig.unique_id(ig.slug(c["name"]), {m["id"] for m in lib.all("models")}),
               label=c["name"], values=dict(like.get("values") or {}, model=c["file"]),
               license_group="commercial", license="",
               notes="From CivitAI: %s\nLicense: %s" % (c["link"], c["license"]))
    rec.pop("backends", None)
    lib.save("models", lib.all("models") + [rec])
    return lib.get("models", rec["id"])


def installed_as(lib, c):
    """The library record a catalog card is already installed as, or None:
    the same file by hash, the same CivitAI version, or the same filename."""
    for rec in lib.all("loras"):
        if c["sha256"] and rec.get("sha256") == c["sha256"]:
            return rec
        if re.search(r"[?&]modelVersionId=%s(\D|$)" % c["version_id"], rec.get("source", "")):
            return rec
    return lib.lora_by_file(c["file"])


def install(lib, client, c, folder, say=lambda text: None, stop=None):
    """A card's LoRA downloaded into `folder` and filed in the library ->
    (record, added). The importer's own path, so the hash is checked and a
    re-install fills only empty fields."""
    return civitai.import_link(lib, client, c["link"], folder, say, stop)


def human_count(n):
    return ("%.1fM" % (n / 1e6) if n >= 1e6 else "%.0fk" % (n / 1e3) if n >= 1e4 else
            "%.1fk" % (n / 1e3) if n >= 1e3 else str(n))


# ============================================================ thumbnails

def _kind(data):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"GIF8":
        return "gif"
    return "other"


def _is_tk_readable(path):
    try:
        with open(path, "rb") as f:
            return _kind(f.read(8)) != "other"
    except OSError:
        return False


PS_TO_PNG = r"""
Add-Type -AssemblyName System.Drawing
$job = [Console]::In.ReadToEnd() | ConvertFrom-Json
foreach ($p in $job.pairs) {
  try {
    $img = [System.Drawing.Image]::FromFile($p.src)
    if ($img.PropertyIdList -contains 274) {
      # A phone's photo stands upright by its EXIF orientation (1-8), which
      # DrawImage ignores. One of the tag's two bytes is 0, whichever order.
      $v = $img.GetPropertyItem(274).Value
      $o = [Math]::Max([int]$v[0], [int]$v[1])
      $turn = @{2 = 'RotateNoneFlipX'; 3 = 'Rotate180FlipNone'; 4 = 'Rotate180FlipX';
                5 = 'Rotate90FlipX'; 6 = 'Rotate90FlipNone'; 7 = 'Rotate270FlipX';
                8 = 'Rotate270FlipNone'}
      if ($turn.ContainsKey($o)) { $img.RotateFlip([System.Drawing.RotateFlipType]$turn[$o]) }
    }
    $k =[Math]::Min(1.0, $job.side / [Math]::Max($img.Width, $img.Height))
    $w = [Math]::Max(1, [int]($img.Width * $k))
    $h = [Math]::Max(1, [int]($img.Height * $k))
    $bmp = New-Object System.Drawing.Bitmap $w, $h
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.DrawImage($img, 0, 0, $w, $h)
    $g.Dispose(); $img.Dispose()
    $bmp.Save($p.dest + '.part', [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Move-Item -Force -LiteralPath ($p.dest + '.part') -Destination $p.dest
  } catch { }
}
"""


def to_png(pairs, side=THUMB):
    """[(picture, png path)] converted, `side` px on the long edge, in one
    PowerShell run -> the png paths that now exist. Windows only; elsewhere,
    or when PowerShell fails, nothing (a card shows no picture)."""
    pairs = [(s, d) for s, d in pairs if s and d]
    if not pairs or sys.platform != "win32":
        return []
    script = base64.b64encode(PS_TO_PNG.encode("utf-16-le")).decode("ascii")
    data = json.dumps({"side": int(side), "pairs": [{"src": os.path.abspath(s),
                                                     "dest": os.path.abspath(d)}
                                                    for s, d in pairs]}).encode("utf-8")
    child = procs.spawn(["powershell", "-NoProfile", "-NonInteractive",
                         "-ExecutionPolicy", "Bypass", "-EncodedCommand", script],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, creationflags=procs.NO_WINDOW)
    try:
        child.proc.communicate(data, timeout=CONVERT_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError):
        pass
    finally:
        child.stop(0)
    return [d for _, d in pairs if os.path.isfile(d)]


def thumbnails(client, wanted, folder, stop=None):
    """{key: picture URL} -> {key: PNG path} under `folder`, cached by key.
    Best effort: a picture that will not come is left out."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(folder, exist_ok=True)
    out, convert, tmp, fetch = {}, [], [], []
    for key, url in wanted.items():
        dest = os.path.join(folder, "%s.png" % re.sub(r"[^\w.-]+", "_", str(key)))
        if os.path.isfile(dest):
            out[key] = dest
        elif url:
            fetch.append((key, url, dest))

    def get(job):
        if stop is not None and stop.is_set():
            return job, None
        try:
            with civitai._open_download(client, job[1]) as r:
                return job, r.read()
        except (civitai.CivitAIError, OSError):
            return job, None
    with ThreadPoolExecutor(max_workers=6) as pool:
        got = list(pool.map(get, fetch))
    for (key, _url, dest), data in got:
        if not data:
            continue
        if _kind(data) != "other":
            with open(dest, "wb") as f:
                f.write(data)
            out[key] = dest
            continue
        fd, src = tempfile.mkstemp(suffix=".img", dir=folder)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        tmp.append(src)
        convert.append((key, src, dest))
    try:
        made = set(to_png([(s, d) for _, s, d in convert]))
        out.update({k: d for k, _, d in convert if d in made})
    finally:
        for src in tmp:
            try:
                os.remove(src)
            except OSError:
                pass
    return out


def previews(recs, folder):
    """{record id: a picture Tk can show} for installed LoRAs: the preview
    itself when it is PNG or GIF, else a PNG copy made once under `folder`."""
    out, convert = {}, []
    for rec in recs:
        path = rec.get("preview") or ""
        if not path or not os.path.isfile(path):
            continue
        if _is_tk_readable(path):
            out[rec["id"]] = path
            continue
        dest = os.path.join(folder, "installed-%s.png" % re.sub(r"[^\w.-]+", "_", rec["id"]))
        if os.path.isfile(dest) and os.path.getmtime(dest) >= os.path.getmtime(path):
            out[rec["id"]] = dest
        else:
            convert.append((rec["id"], path, dest))
    if convert:
        os.makedirs(folder, exist_ok=True)
        made = set(to_png([(s, d) for _, s, d in convert]))
        out.update({k: d for k, _, d in convert if d in made})
    return out


# ============================================================= uninstall

def local_copies(rec, backends):
    """Paths of this LoRA's file in every backend's LoRA folder on this PC."""
    out = []
    for b in backends:
        folder = b.get("lora_dir") or ""
        if not folder or not os.path.isdir(folder):
            continue
        path = os.path.normpath(os.path.join(folder, ig.lora_file(rec, b["id"])))
        if path not in out and os.path.isfile(path):
            out.append(path)
    return out


def recycle(path):
    """`path` to the Recycle Bin (Windows' shell, undo allowed, no dialogs).
    Raises OSError when it did not go."""
    if sys.platform != "win32":
        raise OSError("The Recycle Bin is Windows'; remove %s by hand." % path)
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p),
                    ("lpszProgressTitle", wintypes.LPCWSTR)]
    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = (
        3, 0x4, 0x10, 0x40, 0x400)
    op = SHFILEOPSTRUCTW(wFunc=FO_DELETE, pFrom=os.path.abspath(path) + "\0",
                         fFlags=FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO
                         | FOF_NOERRORUI)
    err = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if err or op.fAnyOperationsAborted or os.path.exists(path):
        raise OSError("Windows would not move %s to the Recycle Bin (code %s)." % (path, err))


def users_of(lib, rec):
    """Names of the identities, styles and saved mixes that name this LoRA."""
    out = ["identity " + i["name"] for i in lib.all("identities") if i.get("lora") == rec["id"]]
    out += ["style " + s["name"] for s in lib.all("styles") if s.get("lora") == rec["id"]]
    out += ["preset " + p["name"] for p in lib.all("presets")
            if any(x.get("id") == rec["id"] for x in p.get("loras") or ())]
    return out


def uninstall(lib, rec, backends, send=recycle):
    """Every copy of the LoRA on this PC to the Recycle Bin, then its record
    out of the library -> the paths recycled. With no copy on this PC,
    raises ValueError and changes nothing (the caller turns it off)."""
    paths = local_copies(rec, backends)
    if not paths:
        raise ValueError("%s is not in a LoRA folder on this PC." % rec["name"])
    for path in paths:
        send(path)
    lib.save("loras", [r for r in lib.all("loras") if r["id"] != rec["id"]])
    return paths
