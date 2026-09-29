#!/usr/bin/env python3
"""
apps.phone.server - Studio Assist on a phone: a chat with the local model and
the Image Studio's Generate, as one web page this PC serves over the tailnet.

    python apps/phone/server.py            # the tailnet and this PC only
    python apps/phone/server.py --lan      # also the home network, behind a passcode
    python apps/phone/server.py --check    # what it would serve and where; serves nothing

The phone is a browser and nothing more: the page is `page.html`, the
conversation lives in the phone's own storage, and everything that costs a GPU
happens where it already does - LM Studio on the LLM PC, ComfyUI on either
machine. This process is the go-between, and it is the desktop app's engine
with no window: `core.agent.LLM` streams the reply, `fit_model` loads the model
onto an empty card with a window that fits the conversation, and a picture is
an `apps.image_studio.imagegen.Studio` job - the same library, routing, passes
and History as the Image Studio tab, so a picture made on the phone is in the
tab's History when the user is back at the desk.

Who may ask. A tailnet address is one of the user's own signed-in devices, and
WireGuard has already authenticated it: those and this PC itself are served
with no question. With `--lan` the server also listens on every interface, and
a private address that is not on the tailnet must give the passcode once
(`Access`); anything else is refused. A request whose `Host` is not this PC's
address or name, or whose `Origin` is another site, is refused too - a page
open in the phone's browser must not be able to make pictures here.

Stdlib only, like the rest of the app.
"""

if __package__ in (None, ""):  # run as a script: import from the checkout
    import os as _os, sys as _sys
    _sys.path[0] = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

import argparse
import hashlib
import hmac
import http.server
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse

import apps.image_studio.imagegen as ig
import core.agent as eng
import core.doctor as doctor

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "page.html")
PORT = 8765
LOG = logging.getLogger("studio.phone")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

BODY_MAX = 2 * 1024 * 1024        # bytes of JSON a request may carry
MESSAGES_MAX = 400                # messages of one conversation
PROMPT_MAX = 4000                 # characters of a picture's prompt
GALLERY = 40                      # pictures the Pictures tab lists
JOBS_KEPT = 50
THUMB_SIDE = 480
CHAT_TEMPERATURE = 0.7            # a conversation, not tool calls (LLM's default is 0.2)

CHAT_PROMPT = """You are the local model of Studio Assist, a film and design studio's \
assistant, talking with the user on their phone. This conversation has no tools: you \
cannot open files, browse the web or change anything in an app, and must never say \
you did. Answer from what you know and say when you are unsure.
Replies are read on a small screen: short paragraphs, plain words, no preamble, no \
closing offers. Use Markdown only for lists and code."""

# The picture's shape, as the form's width and height. Sizes Z-Image and FLUX
# both take; the model's own default is the square.
SHAPES = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832)}

RECORD_ID = re.compile(r"^(\d{4})(\d{2})(\d{2})-\d{6}-[0-9a-f]{6}(-generated)?$")

_THUMB_PS = r"""
Add-Type -AssemblyName System.Drawing
$jpeg = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() |
  Where-Object { $_.MimeType -eq 'image/jpeg' }
$q = New-Object System.Drawing.Imaging.EncoderParameters 1
$q.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter(
  [System.Drawing.Imaging.Encoder]::Quality, [long]82)
foreach ($pair in $input) {
  $src, $dst = $pair -split '\|', 2
  try {
    $img = [System.Drawing.Image]::FromFile($src)
    $k = [Math]::Min(1.0, SIDE / [Math]::Max($img.Width, $img.Height))
    $w = [Math]::Max(1, [int]($img.Width * $k)); $h = [Math]::Max(1, [int]($img.Height * $k))
    $bmp = New-Object System.Drawing.Bitmap $w, $h
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.InterpolationMode = 'HighQualityBicubic'
    $g.Clear([System.Drawing.Color]::White)
    $g.DrawImage($img, 0, 0, $w, $h)
    $bmp.Save($dst, $jpeg, $q)
    $g.Dispose(); $bmp.Dispose(); $img.Dispose()
  } catch { }
}
"""


class Refused(Exception):
    """A request this server will not carry out, said in words the phone shows.
    `status` is the HTTP status it goes back with."""

    def __init__(self, text, status=400):
        super().__init__(text)
        self.status = status


def phone_dir():
    """Beside the settings file, like everything else the app keeps."""
    return os.path.join(doctor.data_dir(), "phone")


# ------------------------------------------------------------------ addresses

def tailscale_exe():
    return shutil.which("tailscale") or next(
        (p for p in (r"C:\Program Files\Tailscale\tailscale.exe",
                     r"C:\Program Files (x86)\Tailscale\tailscale.exe")
         if os.path.isfile(p)), None)


def _run(args, timeout=5):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout if r.returncode == 0 else ""


def on_tailnet(address):
    try:
        return ipaddress.ip_address(address) in eng.TAILNET
    except ValueError:
        return False


def tailnet_address():
    """This PC's own Tailscale address, or None when it has none: asked of
    the Tailscale CLI, else looked for among the addresses of this PC's name."""
    exe = tailscale_exe()
    if exe:
        for line in _run([exe, "ip", "-4"]).split():
            if on_tailnet(line.strip()):
                return line.strip()
    try:
        found = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return None
    return next((a[4][0] for a in found if on_tailnet(a[4][0])), None)


def tailnet_name():
    """This PC's MagicDNS name ("desktop-x.tail1234.ts.net"), or ""."""
    exe = tailscale_exe()
    if not exe:
        return ""
    try:
        me = (json.loads(_run([exe, "status", "--json"]) or "{}").get("Self") or {})
    except ValueError:
        return ""
    return str(me.get("DNSName") or "").rstrip(".").lower()


def lan_addresses():
    """This PC's private addresses that are not the tailnet's."""
    try:
        found = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    out = []
    for a in found:
        ip = a[4][0]
        if ip not in out and ipaddress.ip_address(ip).is_private and not on_tailnet(ip) \
                and not ipaddress.ip_address(ip).is_loopback:
            out.append(ip)
    return out


def own_names():
    """What a request's Host may name: this PC, by any of its names. An IP
    address is always taken - a rebinding attack arrives under a domain name."""
    names = {"localhost", socket.gethostname().lower()}
    full = tailnet_name()
    if full:
        names.update((full, full.split(".")[0]))
    return names


def host_of(header):
    """The host of a Host or Origin value, lower case, no port or brackets."""
    text = (header or "").strip().lower()
    if "://" in text:
        text = urllib.parse.urlsplit(text).netloc
    if text.startswith("["):
        return text[1:].split("]")[0]
    return text.rsplit(":", 1)[0] if text.count(":") == 1 else text


def is_address(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------- access

class Access:
    """Who is served. -> "yes", "passcode" (a private address, `--lan`, not
    yet paired) or "no".

    The passcode is six digits made once and kept in `phone.json`; a phone
    that gives it gets a cookie, and only the cookie's hash is kept. Five wrong
    guesses from one address close the door to it for `LOCK` seconds - six
    digits are a million guesses only if guessing is slow."""

    TRIES = 5
    LOCK = 600
    COOKIE = "studio_phone"

    def __init__(self, root=None, lan=False):
        self.path = os.path.join(root or phone_dir(), "phone.json")
        self.lan = lan
        self.lock = threading.Lock()
        self.wrong = {}                   # address -> [count, locked until]
        self.state = self._read()
        if lan and not self.state.get("passcode"):
            self.state["passcode"] = "%06d" % secrets.randbelow(10 ** 6)
            self._write()

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {}
        d = d if isinstance(d, dict) else {}
        return {"passcode": d.get("passcode") if isinstance(d.get("passcode"), str) else "",
                "paired": [h for h in d.get("paired") or [] if isinstance(h, str)][-20:]}

    def _write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.path)

    @property
    def passcode(self):
        return self.state.get("passcode") or ""

    @staticmethod
    def _hash(token):
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def check(self, address, cookie=""):
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return "no"
        if ip.is_loopback or ip in eng.TAILNET:
            return "yes"
        if not (self.lan and ip.is_private):
            return "no"
        if cookie and any(hmac.compare_digest(self._hash(cookie), h)
                          for h in self.state["paired"]):
            return "yes"
        return "passcode"

    def pair(self, address, passcode):
        """-> the cookie's token. Raises Refused for a wrong passcode."""
        now = time.time()
        with self.lock:
            count, until = self.wrong.get(address, (0, 0))
            if until > now:
                raise Refused("Too many wrong passcodes. Try again in %d minutes."
                              % max(1, int((until - now) / 60 + 0.5)), 429)
            if not (self.passcode and hmac.compare_digest(str(passcode or ""), self.passcode)):
                count += 1
                self.wrong[address] = ((0, now + self.LOCK) if count >= self.TRIES
                                       else (count, 0))
                raise Refused("That is not the passcode. It is shown in the Studio Assist "
                              "Phone window on the PC.", 401)
            self.wrong.pop(address, None)
            token = secrets.token_urlsafe(32)
            self.state["paired"] = (self.state["paired"] + [self._hash(token)])[-20:]
            self._write()
            return token


# ---------------------------------------------------------------- the engine

def words(text, limit):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "\u2026"


def clean_messages(raw):
    """The phone's conversation as the host takes it: user and assistant
    turns of plain text, in order. Anything else is dropped - the system
    prompt is this server's to write, never the page's."""
    out = []
    for m in (raw if isinstance(raw, list) else [])[-MESSAGES_MAX:]:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        text = m.get("content")
        if isinstance(text, str) and text.strip():
            out.append({"role": m["role"], "content": text})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def estimate(messages):
    """A conversation's cost in tokens before the host has counted it: prose
    runs about four characters a token; three keeps the guess on the high side."""
    return sum(len(m["content"]) + 12 for m in messages) // 3


def trimmed(messages, window):
    """-> (messages, dropped). The oldest exchanges leave first when the
    conversation outgrows the window the model was loaded with; the system
    prompt and the newest message always stay."""
    if not isinstance(window, int):
        return messages, 0
    room = window - eng.MIN_ROOM
    head, rest = messages[:1], messages[1:]
    dropped = 0
    while len(rest) > 1 and estimate(head + rest) > room:
        rest.pop(0)
        dropped += 1
        while len(rest) > 1 and rest[0]["role"] != "user":
            rest.pop(0)
            dropped += 1
    return head + rest, dropped


class Phone:
    """Everything the page can ask for, with no socket in it: the handler
    below is only HTTP around these methods, and the tests call them as they
    are. `llm` makes the client a reply streams from; `studio` is the Image
    Studio's engine (made on first use: it reads the library from disk)."""

    def __init__(self, host=None, studio=None, llm=None, root=None):
        self.host = host or eng.env_default("STUDIO_HOST", "AE_AGENT_HOST",
                                            fallback=eng.DEFAULT_HOST)
        self.root = root or phone_dir()
        self.llm = llm or (lambda model: eng.LLM(self.host, model,
                                                 temperature=CHAT_TEMPERATURE, timeout=600))
        self._studio = studio
        if studio is not None and studio.make_room is None:
            studio.make_room = self.make_room
        self.gpu = threading.Lock()       # one load or unload on the host at a time
        self.lock = threading.Lock()
        self.talking = {}                 # model -> replies streaming from it now
        self.jobs = {}                    # job id -> Job, the newest JOBS_KEPT
        self.records = {}                 # record path -> (mtime, summary or None)
        self.icon = None

    # ------------------------------------------------------------- models
    @property
    def studio(self):
        with self.lock:
            if self._studio is None:
                self._studio = ig.Studio(make_room=self.make_room)
            return self._studio

    def busy_models(self):
        with self.lock:
            return {m for m, n in self.talking.items() if n > 0}

    def make_room(self, backend=None):
        """Before a picture on a ComfyUI that shares the LLM PC's card: LM
        Studio's models off it, but for one a reply is streaming from now."""
        with self.gpu:
            eng.make_room(self.host, self.busy_models())

    def models(self):
        """-> {"reachable", "models": [{"id", "loaded", "sees"}], "model",
        "error"}: what the host serves for a conversation, what is on the
        card first, and the one a new chat would use."""
        reachable, loaded, ids, vision, error = eng.probe_models(self.host)
        if not reachable:
            return {"reachable": False, "models": [], "model": None,
                    "error": self.unreachable(error)}
        ids = [i for i in ids if "embed" not in i.lower()]
        ids.sort(key=lambda i: (i not in loaded, i.lower()))
        return {"reachable": True, "model": eng.pick_model(loaded, ids), "error": None,
                "models": [{"id": i, "loaded": i in loaded, "sees": i in vision}
                           for i in ids]}

    def unreachable(self, error=None):
        alive = eng.host_alive(self.host)
        where = eng.api_root(self.host)
        if alive is False:
            return ("The LLM PC is not answering at all: it is asleep, off, or off the "
                    "tailnet. Wake it and try again.")
        if alive:
            return ("The LLM PC is up, but LM Studio's server is not answering at %s. "
                    "Start the server in LM Studio." % where)
        return "Cannot reach LM Studio at %s (%s)." % (where, error or "no answer")

    # --------------------------------------------------------------- chat
    def chat(self, body, send):
        """One reply, streamed. `send(dict)` writes one event to the phone
        and raises when the phone has gone: {"note"} while a model loads,
        {"t"} for each piece of the reply, then {"done"} or {"error"}."""
        messages = clean_messages(body.get("messages"))
        if not messages or messages[-1]["role"] != "user":
            raise Refused("There is no message to answer.")
        reachable, loaded, ids, _vision, error = eng.probe_models(self.host)
        if not reachable:
            raise Refused(self.unreachable(error), 502)
        want = body.get("model") if isinstance(body.get("model"), str) else None
        model = eng.pick_model(loaded, ids, want)
        if not model:
            raise Refused("LM Studio has no model to talk to. Download one on the LLM PC.",
                          502)
        messages = [{"role": "system", "content": CHAT_PROMPT}] + messages
        with self.lock:
            self.talking[model] = self.talking.get(model, 0) + 1
        try:
            if model not in loaded:
                send({"note": "Loading %s\u2026" % model})
            elif self.rendering():
                send({"note": "A picture is being made on the same GPU; this reply may "
                              "be slow."})
            with self.gpu:
                # Onto an empty card, or it decodes at a third of its speed for
                # as long as it stays loaded (AGENTS.md, "A model loads onto an
                # empty card"). Nothing is unloaded when the model already fits.
                window, note = eng.fit_model(self.host, model, estimate(messages),
                                             exact=False, keep=self.busy_models())
            messages, dropped = trimmed(messages, window)
            if dropped:
                send({"note": "This conversation is longer than %s can hold: the oldest "
                              "%d messages were left out." % (model, dropped)})
            reply = self.llm(model).stream(messages, on_text=lambda t: send({"t": t}))
            send({"done": True, "model": model, "window": window,
                  "empty": not (reply.get("content") or "").strip()})
        finally:
            with self.lock:
                self.talking[model] -= 1

    def rendering(self):
        """Whether a picture is being made where LM Studio's models live."""
        with self.lock:
            jobs = list(self.jobs.values())
        return any(j.status not in ig.FINISHED and j.backend.get("shares_llm_gpu")
                   for j in jobs)

    # ----------------------------------------------------------- pictures
    def choices(self):
        """What the Pictures form offers, from the Image Studio's library."""
        lib = self.studio.lib
        self.studio.check_all(full=False)
        return {
            "models": [{"id": m["id"], "name": m["label"]} for m in lib.all("models")
                       if self.from_words(m)],
            "model": ig.default_settings()["model"],
            "styles": [{"id": s["id"], "name": s["name"]} for s in lib.all("styles")],
            "people": [{"id": p["id"], "name": p["name"]} for p in lib.all("identities")],
            "shapes": [{"id": k, "name": k.capitalize(), "width": w, "height": h}
                       for k, (w, h) in SHAPES.items()],
            "backends": [{"name": b["name"], "ok": bool((self.studio.health.get(b["id"])
                                                         or {}).get("ok")),
                          "detail": (self.studio.health.get(b["id"]) or {}).get("detail", "")}
                         for b in self.studio.backends() if b["enabled"]],
        }

    def from_words(self, model):
        """Whether a model makes a picture from the form's few fields. The
        family photo's workflow wants a Scene Builder layout of faces."""
        try:
            return not self.studio.workflow_loader(model["workflow"]).get("multi_identity")
        except (ig.TemplateError, KeyError):
            return False

    def generate(self, body):
        """Queue one picture from the form. -> its state. Routing asks the
        backends how they are: network I/O."""
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise Refused("Say what the picture should show.")
        if len(prompt) > PROMPT_MAX:
            raise Refused("That is too long for a picture: %d characters, and %d is the "
                          "most." % (len(prompt), PROMPT_MAX))
        lib = self.studio.lib
        s = ig.default_settings()
        s["scene"] = prompt
        model = body.get("model") or s["model"]
        if lib.get("models", model) is None:
            raise Refused("There is no model called %r in the Image Studio." % model)
        s["model"] = model
        style = body.get("style") or "none"
        if lib.get("styles", style) is not None:
            s["style"] = style
        person = body.get("person")
        if person:
            who = lib.get("identities", person)
            if who is None:
                raise Refused("There is no person called %r in the Image Studio." % person)
            s["identities"] = [{"id": who["id"], "strength": who["strength"]}]
        s["width"], s["height"] = SHAPES.get(body.get("shape"), SHAPES["square"])
        # The form's hands pass finds and redraws every hand after the picture
        # is made: minutes on the 3090, for a picture that may have none. On
        # the phone it is asked for, not assumed.
        s["hand_pass"] = body.get("hands") is True
        try:
            job = self.studio.submit(s)[0]
        except ig.ComfyError as e:
            raise Refused(str(e), 503)
        with self.lock:
            self.jobs[job.id] = job
            for old in list(self.jobs)[:-JOBS_KEPT]:
                if self.jobs[old].status in ig.FINISHED:
                    del self.jobs[old]
        return self.state(job)

    def job(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
        if job is None:
            raise Refused("That picture is not one this server is making. It may have "
                          "been restarted: look in Pictures.", 404)
        return job

    def state(self, job):
        done = job.status in ig.FINISHED
        record = job.record if done and job.status == "complete" else None
        return {"id": job.id, "status": job.status, "detail": job.detail,
                "progress": job.progress, "done": done,
                "seconds": round(job.elapsed(), 1), "backend": job.backend.get("name", ""),
                "prompt": words(job.settings.get("scene"), 200),
                "pictures": self.urls(record) if record else []}

    def cancel(self, jid):
        job = self.job(jid)
        self.studio.queue.cancel(job)
        return self.state(job)

    @staticmethod
    def urls(record):
        return [{"full": "/picture/%s/%d" % (record["id"], i + 1),
                 "thumb": "/thumb/%s/%d" % (record["id"], i + 1)}
                for i in range(len(record.get("images") or []))]

    # ------------------------------------------------------------ history
    def summary(self, path):
        """One History record as the gallery shows it, or None for one that
        is not a finished picture. Parsed once per change: a record carries
        its whole graph, and fifty of them are seconds to read."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        kept = self.records.get(path)
        if kept and kept[0] == mtime:
            return kept[1]
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            rec = None
        out = None
        if (isinstance(rec, dict) and isinstance(rec.get("settings"), dict)
                and RECORD_ID.match(str(rec.get("id") or ""))
                and (rec.get("finish") or {}).get("state") != "complete"
                and rec.get("images")):
            out = {"id": rec["id"], "created": rec.get("created", ""),
                   "prompt": words(rec["settings"].get("scene") or rec.get("prompt"), 200),
                   "model": (rec.get("model") or {}).get("label") or "",
                   "images": [p for p in rec["images"] if isinstance(p, str)]}
        self.records[path] = (mtime, out)
        return out

    def gallery(self, limit=GALLERY):
        """The newest pictures in the Image Studio's History, newest first,
        with a thumbnail made for each that has none."""
        root = self.studio.history.root
        out = []
        days = sorted(os.listdir(root), reverse=True) if os.path.isdir(root) else []
        for day in days:
            folder = os.path.join(root, day)
            if not os.path.isdir(folder):
                continue
            found = [self.summary(os.path.join(folder, n)) for n in os.listdir(folder)
                     if n.endswith(".json")]
            out.extend(sorted((r for r in found if r), key=lambda r: r["id"], reverse=True))
            if len(out) >= limit:
                break
        out = out[:limit]
        self.make_thumbs([(p, self.thumb_path(r["id"], i + 1))
                          for r in out for i, p in enumerate(r["images"])])
        return [{"id": r["id"], "created": r["created"], "prompt": r["prompt"],
                 "model": r["model"], "pictures": self.urls(r)} for r in out]

    def record(self, rid):
        m = RECORD_ID.match(rid or "")
        if not m:
            raise Refused("No such picture.", 404)
        path = os.path.join(self.studio.history.root, "%s-%s-%s" % m.group(1, 2, 3),
                            rid + ".json")
        rec = self.summary(path)
        if rec is None:
            raise Refused("No such picture.", 404)
        return rec

    def picture(self, rid, n, thumb=False):
        """-> (path, content type) of picture `n` (from 1) of a record. The
        phone names a record, never a path, and a record naming a file outside
        History is not served."""
        rec = self.record(rid)
        if not 1 <= n <= len(rec["images"]):
            raise Refused("No such picture.", 404)
        path = os.path.realpath(rec["images"][n - 1])
        root = os.path.realpath(self.studio.history.root)
        if os.path.commonpath([path, root]) != root or not os.path.isfile(path):
            raise Refused("No such picture.", 404)
        if thumb:
            small = self.thumb_path(rid, n)
            self.make_thumbs([(path, small)])
            if os.path.isfile(small):
                return small, "image/jpeg"
        kind = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp"}.get(os.path.splitext(path)[1].lower())
        if kind is None:
            raise Refused("No such picture.", 404)
        return path, kind

    def thumb_path(self, rid, n):
        return os.path.join(self.root, "thumbs", "%s_%d.jpg" % (rid, n))

    def make_thumbs(self, pairs):
        """The missing small copies, in one PowerShell run (stdlib Python
        cannot shrink a picture at any speed worth having). Without one the
        phone is sent the picture itself."""
        todo = [(s, d) for s, d in pairs if os.path.isfile(s) and not os.path.isfile(d)]
        if not todo or os.name != "nt":
            return
        os.makedirs(os.path.join(self.root, "thumbs"), exist_ok=True)
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 _THUMB_PS.replace("SIDE", str(THUMB_SIDE))],
                input="\n".join("%s|%s" % pair for pair in todo).encode("utf-8"),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90,
                creationflags=NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            pass

    def app_icon(self):
        """The app's own mark at the size a home screen wants, drawn by
        `make_icon` and kept: a second of pure Python, once."""
        if self.icon is None:
            path = os.path.join(self.root, "icon-180.png")
            try:
                with open(path, "rb") as f:
                    self.icon = f.read()
            except OSError:
                import make_icon
                from core.icons import png
                self.icon = png(make_icon.render(180), 180, 180)
                try:
                    os.makedirs(self.root, exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(self.icon)
                except OSError:
                    pass
        return self.icon

    def close(self):
        if self._studio is not None:
            self._studio.close()


# ----------------------------------------------------------------------- HTTP

MANIFEST = {"name": "Studio Assist", "short_name": "Studio", "display": "standalone",
            "start_url": "/", "background_color": "#16150f", "theme_color": "#16150f",
            "icons": [{"src": "/icon.png", "sizes": "180x180", "type": "image/png"}]}


class Handler(http.server.BaseHTTPRequestHandler):
    """HTTP around `Phone`. HTTP/1.0 on purpose: every answer ends with the
    connection, so a streamed reply needs no chunk framing and a phone that
    walks away is seen at the next write."""

    server_version = "StudioAssistPhone"
    sys_version = ""

    QUIET = re.compile(r"^/(api/jobs/[0-9a-f]{12}|thumb/.*|api/access)$")

    def log_message(self, fmt, *args):
        LOG.info("%s %s", self.client_address[0], fmt % args)

    def log_request(self, code="-", size="-"):
        # A picture under way is asked after every second; the window would
        # show nothing else.
        if not (str(code) == "200" and self.QUIET.match(self.path or "")):
            super().log_request(code, size)

    # ------------------------------------------------------------ answers
    def answer(self, status, body, kind, headers=()):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def say(self, value, status=200, headers=()):
        self.answer(status, json.dumps(value).encode("utf-8"),
                    "application/json; charset=utf-8", headers)

    def file(self, path, kind, keep=False):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=31536000, immutable" if keep
                         else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------- asking
    def cookie(self):
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == Access.COOKIE:
                return v
        return ""

    def admitted(self):
        """Whether this request is served at all; answers it when not."""
        host = host_of(self.headers.get("Host"))
        if not (is_address(host) or host in self.server.names):
            self.say({"error": "This server does not answer to that name."}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "null" and host_of(origin) != host:
            self.say({"error": "Asked from another site."}, 403)
            return False
        if origin == "null" and self.command == "POST":
            self.say({"error": "Asked from another site."}, 403)
            return False
        self.pass_ = self.server.access.check(self.client_address[0], self.cookie())
        if self.pass_ == "no":
            self.say({"error": "This phone is not on the studio's tailnet."}, 403)
            return False
        return True

    def body(self):
        if "json" not in (self.headers.get("Content-Type") or "").lower():
            raise Refused("Expected JSON.", 415)
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = -1
        if not 0 <= size <= BODY_MAX:
            raise Refused("That is too much to send at once.", 413)
        try:
            value = json.loads(self.rfile.read(size).decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            raise Refused("That was not JSON.")
        if not isinstance(value, dict):
            raise Refused("That was not a JSON object.")
        return value

    def guarded(self, work):
        try:
            if self.admitted():
                work()
        except Refused as e:
            self.say({"error": str(e)}, e.status)
        except (BrokenPipeError, ConnectionError):
            pass                          # the phone went away; nothing to tell it
        except Exception as e:
            doctor.log_error("Phone server, %s %s:\n%r" % (self.command, self.path, e))
            try:
                self.say({"error": "%s: %s" % (type(e).__name__, e)}, 500)
            except OSError:
                pass

    def do_GET(self):
        self.guarded(self.get)

    def do_POST(self):
        self.guarded(self.post)

    # ------------------------------------------------------------- routes
    def get(self):
        phone = self.server.phone
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/", "/index.html"):
            with open(PAGE, "rb") as f:
                return self.answer(200, f.read(), "text/html; charset=utf-8", [
                    ("Content-Security-Policy",
                     "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                     "script-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                     "frame-ancestors 'none'")])
        if path == "/manifest.webmanifest":
            return self.answer(200, json.dumps(MANIFEST).encode("utf-8"),
                               "application/manifest+json")
        if path in ("/icon.png", "/favicon.ico", "/apple-touch-icon.png"):
            return self.answer(200, phone.app_icon(), "image/png")
        if path == "/api/access":
            return self.say({"paired": self.pass_ == "yes"})
        if self.pass_ != "yes":
            raise Refused("Give the passcode first.", 401)
        if path == "/api/models":
            return self.say(phone.models())
        if path == "/api/choices":
            return self.say(phone.choices())
        if path == "/api/pictures":
            return self.say({"pictures": phone.gallery()})
        m = re.match(r"^/api/jobs/([0-9a-f]{12})$", path)
        if m:
            return self.say(phone.state(phone.job(m.group(1))))
        m = re.match(r"^/(picture|thumb)/([0-9a-z-]{1,40})/(\d{1,3})$", path)
        if m:
            found, kind = phone.picture(m.group(2), int(m.group(3)), m.group(1) == "thumb")
            return self.file(found, kind, keep=True)
        raise Refused("Nothing is here.", 404)

    def post(self):
        phone = self.server.phone
        path = urllib.parse.urlsplit(self.path).path
        body = self.body()
        if path == "/api/pair":
            token = self.server.access.pair(self.client_address[0], body.get("passcode"))
            return self.say({"paired": True}, headers=[(
                "Set-Cookie", "%s=%s; Max-Age=31536000; Path=/; HttpOnly; SameSite=Strict"
                % (Access.COOKIE, token))])
        if self.pass_ != "yes":
            raise Refused("Give the passcode first.", 401)
        if path == "/api/chat":
            return self.chat(body)
        if path == "/api/jobs":
            return self.say(phone.generate(body))
        m = re.match(r"^/api/jobs/([0-9a-f]{12})/cancel$", path)
        if m:
            return self.say(phone.cancel(m.group(1)))
        raise Refused("Nothing is here.", 404)

    def chat(self, body):
        """The reply as lines of JSON, each written as it is made. An error
        after the first line is one more line: the status has gone by then."""
        started = []

        def send(event):
            if not started:
                started.append(True)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
            self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
            self.wfile.flush()

        try:
            self.server.phone.chat(body, send)
        except ConnectionError:
            raise                         # the phone went away mid-reply
        except Exception as e:
            if not started and isinstance(e, Refused):
                raise
            LOG.warning("chat: %s", e)
            send({"error": str(e) or type(e).__name__})


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False           # a second copy must fail, not share the port

    def __init__(self, address, phone, access, names):
        super().__init__(address, Handler)
        self.phone, self.access, self.names = phone, access, names


def plan(lan=False, port=PORT):
    """-> ([addresses to listen on], [the links to open on the phone])."""
    tail = tailnet_address()
    name = tailnet_name()
    if lan:
        binds = ["0.0.0.0"]
        links = ["http://%s:%d" % (a, port) for a in lan_addresses()]
    else:
        binds = ["127.0.0.1"] + ([tail] if tail else [])
        links = []
    if tail:
        links.insert(0, "http://%s:%d" % (tail, port))
        if name:
            links.insert(1, "http://%s:%d" % (name.split(".")[0], port))
    return binds, links


def serve(phone, access, binds, port=PORT):
    """Start one listener per address, each on its own thread. -> [Server]."""
    names = own_names()
    servers = []
    for address in binds:
        s = Server((address, port), phone, access, names)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
    return servers


def main(argv=None):
    ap = argparse.ArgumentParser(description="Studio Assist on a phone: chat and pictures "
                                             "as a web page, over the tailnet.")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--lan", action="store_true",
                    help="also serve the home network, behind a passcode")
    ap.add_argument("--check", action="store_true",
                    help="say what would be served and where, and serve nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    binds, links = plan(args.lan, args.port)
    phone = Phone()
    access = Access(lan=args.lan)
    print("Studio Assist Phone")
    print()
    found = phone.models()
    print("  Chat      %s" % ("%s, %d models (%s)" % (
        eng.api_root(phone.host), len(found["models"]), found["model"])
        if found["reachable"] else found["error"]))
    for b in phone.choices()["backends"]:
        print("  Pictures  %s: %s" % (b["name"], b["detail"] if b["ok"] else
                                      "not answering (%s)" % words(b["detail"], 120)))
    print()
    if not links:
        print("  This PC has no Tailscale address, so only this PC can open it:")
        print("      http://127.0.0.1:%d" % args.port)
        print("  Start Tailscale here, or run with --lan for the home network.")
    else:
        print("  On the phone, open:")
        for link in links:
            print("      " + link)
        if on_tailnet(host_of(links[0])):
            print("  The phone needs the Tailscale app, signed in to the same account.")
    if args.lan:
        print("  Passcode for the home network: %s" % access.passcode)
    print()
    if args.check:
        return 0 if found["reachable"] else 1
    try:
        servers = serve(phone, access, binds, args.port)
    except OSError as e:
        print("  Could not listen on port %d (%s). Is it already running?"
              % (args.port, e))
        return 2
    print("  Serving. Close this window to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for s in servers:
            s.shutdown()
        phone.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
