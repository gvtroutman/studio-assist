"""
Icons found on the web, for Preferences > Icons and a sidebar row's menu.

The search is Wikimedia Commons': free, no key, and it draws its SVG logos as
PNG thumbnails on its own servers - Tk 8.6 shows neither SVG nor JPEG, and the
stdlib-only rule bars a rasteriser. So only what arrives as PNG or GIF is
offered (a drawing's thumbnail always does), each with the licence Commons
gives it, and only from Wikimedia's own hosts: the addresses come back inside
the answer, and are not followed anywhere else.

    python core/icon_search.py "photoshop logo"
"""

if __package__ in (None, ""):  # run as a script: import from the checkout
    import os as _os, sys as _sys
    _sys.path[0] = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

import json
import sys
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "StudioAssist/1.0 (local desktop app)"   # Wikimedia asks for one
TIMEOUT = 15
LIMIT = 24                     # results asked for; some are dropped as photos
THUMB_PX = 128                 # asked for; Commons rounds up to its own sizes
MAX_BYTES = 2_000_000          # a thumbnail bigger than this is not an icon
# A drawing's thumbnail is a PNG; a JPEG is a photo, and Tk cannot show one.
SHOWN = {"image/svg+xml", "image/png", "image/gif"}
HOSTS = (".wikimedia.org", ".wikipedia.org")
MAGIC = (b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")


class SearchError(Exception):
    """Why a search or a download did not happen, in the user's words."""


def words_for(name, logo=True):
    """What the search box starts with: an app's logo, a button's icon."""
    return "%s %s" % (" ".join((name or "").split()), "logo" if logo else "icon")


def _open(url, opener):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return (opener or urllib.request.urlopen)(req, timeout=TIMEOUT)


def ours(url):
    """True for an https address on Wikimedia's own hosts."""
    try:
        part = urllib.parse.urlsplit(url or "")
    except ValueError:
        return False
    host = (part.hostname or "").lower()
    return part.scheme == "https" and host.endswith(HOSTS)


def search(words, limit=LIMIT, opener=None):
    """[{title, thumb, page, license}] for `words`, drawings first. [] when
    nothing matches; SearchError when Commons cannot be reached."""
    words = " ".join((words or "").split())
    if not words:
        return []
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": 2,
        "generator": "search", "gsrnamespace": 6, "gsrlimit": limit,
        "gsrsearch": "%s filetype:drawing|bitmap" % words,
        "prop": "imageinfo", "iiprop": "url|mime|extmetadata",
        "iiextmetadatafilter": "LicenseShortName", "iiurlwidth": THUMB_PX})
    try:
        with _open(API + "?" + query, opener) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise SearchError("Could not reach Wikimedia Commons (%s)." % e)
    pages = (data.get("query") or {}).get("pages") or []

    def rank(page):
        # Drawings first: a logo is nearly always an SVG, and the PNGs that
        # match an app's name are mostly screenshots of it.
        info = (page.get("imageinfo") or [{}])[0]
        return info.get("mime") != "image/svg+xml", page.get("index", 0)
    out = []
    for page in sorted(pages, key=rank):
        info = (page.get("imageinfo") or [{}])[0]
        thumb = info.get("thumburl")
        if info.get("mime") not in SHOWN or not ours(thumb):
            continue
        meta = info.get("extmetadata") or {}
        title = page.get("title") or ""
        out.append({
            "title": title.partition(":")[2] or title,
            "thumb": thumb,
            "page": info.get("descriptionurl") or "",
            "license": ((meta.get("LicenseShortName") or {}).get("value")
                        or "licence not given"),
        })
    return out


def fetch(url, opener=None):
    """The thumbnail's bytes: a PNG or GIF from Wikimedia, or SearchError."""
    if not ours(url):
        raise SearchError("Not a Wikimedia address: %s" % url)
    try:
        with _open(url, opener) as r:
            data = r.read(MAX_BYTES + 1)
    except OSError as e:
        raise SearchError("Could not download the picture (%s)." % e)
    if len(data) > MAX_BYTES:
        raise SearchError("The picture is too big for an icon.")
    if not data.startswith(MAGIC):
        raise SearchError("Wikimedia did not send a PNG or GIF.")
    return data


def main(argv):
    for hit in search(" ".join(argv[1:]) or "photoshop logo"):
        print("%-50s %-18s %s" % (hit["title"][:50], hit["license"][:18], hit["thumb"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
