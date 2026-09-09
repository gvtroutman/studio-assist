#!/usr/bin/env python3
"""
AE Agent - a chat window that drives After Effects with a local model.

Inference runs on the tailnet box; tools run here against the AE panel on 7777.
Nothing to install: Tkinter ships with Python, and the engine is stdlib only.
"""

import json
import os
import queue
import re
import socket
import sys
import threading
import time
import traceback

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox
except ImportError:
    sys.exit("Tkinter is missing from this Python install; reinstall Python with tcl/tk.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ae_agent as eng

# ------------------------------------------------------------------ appearance
BG = "#141413"
SIDE = "#1a1a18"
HEAD = "#1a1a18"
CARD = "#232321"
BORDER = "#302f2c"
TEXT = "#ecebe8"
MUTED = "#928d86"
FAINT = "#6b6862"
ACCENT = "#d97757"
ACCENT_DK = "#c26343"
OK = "#5fb87f"
WARN = "#e0a458"
ERRC = "#e0685c"

SIDEBAR_W = 236

EXAMPLES = [
    "What comps are in this project?",
    "Make a 1920x1080 title card, 5 seconds at 24fps",
    "Add a drop shadow to the text and fade it in over 12 frames",
]

CHAT_SYSTEM_PROMPT = eng.SYSTEM_PROMPT + """

This is a continuing conversation. The user may refer back to things you made
earlier - keep track of comp and layer ids you have seen so you do not re-derive
them. Answer questions directly without calling tools when no tool is needed."""

MAX_HISTORY = 40
MAX_STEPS = 25


def pretty_host(url):
    """100.127.17.38:1234 - the scheme and /v1 are noise in a 236px rail."""
    s = url.replace("https://", "").replace("http://", "").rstrip("/")
    return s[:-3].rstrip("/") if s.endswith("/v1") else s


def clip(s, n):
    """Truncate rather than let a label wrap mid-word."""
    return s if len(s) <= n else s[:n - 1] + "…"


def rounded(canvas, x1, y1, x2, y2, r, **kw):
    """Rounded rectangle - Tk's canvas has no primitive for it."""
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


class Chat(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AE Agent")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1180, int(sw * 0.72)), min(820, int(sh * 0.80))
        self.geometry("%dx%d+%d+%d" % (w, h, (sw - w) // 2, max(0, (sh - h) // 3)))
        self.minsize(820, 520)
        self.configure(bg=BG)

        self.q = queue.Queue()
        self.mcp = None
        self.llm = None
        self.tools = []
        self.messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        self.busy = False
        self.host = os.environ.get("AE_AGENT_HOST", eng.DEFAULT_HOST)
        self.want_model = os.environ.get("AE_AGENT_MODEL")
        self.state_ae = False
        self.state_model = None
        self.n_models = 0
        self._stream_open = False
        self._stream_buf = []
        self._asst_start = "1.0"

        self._fonts()
        self._build()
        self.after(40, self._drain)
        self._spawn(self._boot)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _fonts(self):
        self.f_ui = tkfont.Font(family="Segoe UI", size=10)
        self.f_bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.f_title = tkfont.Font(family="Segoe UI Semibold", size=12)
        self.f_small = tkfont.Font(family="Segoe UI", size=8)
        self.f_cap = tkfont.Font(family="Segoe UI", size=8, weight="bold")
        self.f_badge = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self.f_mono = tkfont.Font(family="Consolas", size=9)
        self.f_body = tkfont.Font(family="Segoe UI", size=11)
        self.f_body_b = tkfont.Font(family="Segoe UI", size=11, weight="bold")

    # ------------------------------------------------------------------- layout
    def _build(self):
        head = tk.Frame(self, bg=HEAD, height=52)
        head.pack(side="top", fill="x")
        head.pack_propagate(False)
        tk.Frame(self, bg=BORDER, height=1).pack(side="top", fill="x")

        tk.Label(head, text="AE Agent", bg=HEAD, fg=TEXT, font=self.f_title
                 ).pack(side="left", padx=(18, 10))
        self.lbl_status = tk.Label(head, text="starting", bg=HEAD, fg=MUTED,
                                   font=self.f_ui)
        self.lbl_status.pack(side="left")
        self.btn_new = tk.Button(head, text="New chat", command=self._on_new,
                                 font=self.f_ui, bg=CARD, fg=TEXT, relief="flat",
                                 activebackground=BORDER, activeforeground=TEXT,
                                 padx=12, pady=4, cursor="hand2", bd=0)
        self.btn_new.pack(side="right", padx=(6, 18))
        self.btn_fix = tk.Button(head, text="Start After Effects", command=self._on_fix,
                                 font=self.f_ui, bg=ACCENT, fg="#16150f", relief="flat",
                                 activebackground=ACCENT_DK, padx=12, pady=4,
                                 cursor="hand2", bd=0)

        main = tk.Frame(self, bg=BG)
        main.pack(side="top", fill="both", expand=True)

        side = tk.Frame(main, bg=SIDE, width=SIDEBAR_W)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Frame(main, bg=BORDER, width=1).pack(side="left", fill="y")
        self._build_sidebar(side)

        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # The composer is packed BEFORE the transcript on purpose. In Tk an
        # expanding sibling packed first claims the leftover space and shoves a
        # later fixed-size widget off the bottom edge - which is exactly how the
        # input used to disappear until the window was resized.
        composer = tk.Frame(right, bg=BG)
        composer.pack(side="bottom", fill="x", padx=18, pady=(8, 16))
        self._build_composer(composer)

        wrap = tk.Frame(right, bg=BG)
        wrap.pack(side="top", fill="both", expand=True)
        bar = tk.Scrollbar(wrap, bg=BG, troughcolor=BG, activebackground=FAINT,
                           highlightthickness=0, bd=0, width=11)
        bar.pack(side="right", fill="y")
        self.view = tk.Text(wrap, bg=BG, fg=TEXT, font=self.f_body, wrap="word", bd=0,
                            padx=22, pady=16, yscrollcommand=bar.set, state="disabled",
                            cursor="arrow", selectbackground="#3d3b37",
                            insertbackground=TEXT, highlightthickness=0)
        self.view.pack(side="left", fill="both", expand=True)
        bar.config(command=self.view.yview)
        self._tags()
        self._welcome()

    def _tags(self):
        v = self.view
        v.tag_configure("role_user", foreground=ACCENT, font=self.f_cap,
                        spacing1=18, spacing3=6)
        v.tag_configure("role_asst", foreground="#8fb0c9", font=self.f_cap,
                        spacing1=18, spacing3=6)
        v.tag_configure("user", background=CARD, lmargin1=16, lmargin2=16, rmargin=16,
                        spacing1=6, spacing3=10, borderwidth=0)
        v.tag_configure("asst", lmargin1=16, lmargin2=16, rmargin=60, spacing2=4,
                        spacing3=10)
        v.tag_configure("tool", foreground=FAINT, font=self.f_mono, lmargin1=22,
                        lmargin2=36, rmargin=16, spacing1=2, spacing3=2)
        v.tag_configure("err", foreground=ERRC, lmargin1=16, lmargin2=16, rmargin=40,
                        spacing3=10)
        v.tag_configure("sys", foreground=MUTED, lmargin1=16, lmargin2=16, spacing3=8)
        v.tag_configure("hint", foreground=FAINT, lmargin1=26, lmargin2=26, spacing3=5)
        # last: these win on font when combined with a block tag
        v.tag_configure("b", font=self.f_body_b)
        v.tag_configure("code", font=self.f_mono, foreground="#d7d1c9")

    def _cap(self, parent, text):
        tk.Label(parent, text=text, bg=SIDE, fg=FAINT, font=self.f_cap,
                 anchor="w").pack(fill="x", padx=18, pady=(18, 8))

    def _badge(self, parent, code, fg, bg, size=26):
        c = tk.Canvas(parent, width=size, height=size, bg=SIDE,
                      highlightthickness=0, bd=0)
        rounded(c, 1, 1, size - 1, size - 1, 7, fill=bg, outline=bg)
        c.create_text(size / 2, size / 2 + 1, text=code, fill=fg, font=self.f_badge)
        return c

    def _dot(self, parent, color, size=8, bg=SIDE):
        c = tk.Canvas(parent, width=size + 2, height=size + 2, bg=bg,
                      highlightthickness=0, bd=0)
        c.create_oval(1, 1, size, size, fill=color, outline=color)
        return c

    def _build_sidebar(self, side):
        self._cap(side, "ON THIS PC")
        apps = eng.detect_apps()
        for a in apps:
            row = tk.Frame(side, bg=SIDE)
            row.pack(fill="x", padx=14, pady=2)
            self._badge(row, a["code"], a["fg"], a["bg"]).pack(side="left")
            box = tk.Frame(row, bg=SIDE)
            box.pack(side="left", fill="x", expand=True, padx=(9, 0))
            tk.Label(box, text=a["name"], bg=SIDE, fg=TEXT, font=self.f_ui,
                     anchor="w").pack(fill="x")
            sub = a["version"] or "installed"
            if a["drivable"]:
                sub += "  ·  drivable"
            tk.Label(box, text=sub, bg=SIDE, fg=FAINT, font=self.f_small,
                     anchor="w").pack(fill="x")
            if a["drivable"]:
                self.dot_app = self._dot(row, WARN)
                self.dot_app.pack(side="right", padx=(4, 2))
        if not apps:
            tk.Label(side, text="no creative apps found", bg=SIDE, fg=FAINT,
                     font=self.f_small).pack(padx=18, anchor="w")

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(16, 0))
        self._cap(side, "CONNECTIONS")
        self.dot_host, self.lbl_host = self._conn_row(
            side, "Inference", pretty_host(self.host))
        self.dot_bridge, self.lbl_bridge = self._conn_row(
            side, "After Effects bridge", "127.0.0.1:7777")

    def _conn_row(self, side, title, detail):
        row = tk.Frame(side, bg=SIDE)
        row.pack(fill="x", padx=14, pady=3)
        dot = self._dot(row, FAINT)
        dot.pack(side="left", padx=(6, 0), pady=(3, 0), anchor="n")
        box = tk.Frame(row, bg=SIDE)
        box.pack(side="left", fill="x", expand=True, padx=(10, 0))
        tk.Label(box, text=title, bg=SIDE, fg=TEXT, font=self.f_ui,
                 anchor="w").pack(fill="x")
        # no wraplength: these are pre-clipped, and char wrapping split IPs mid-number
        lbl = tk.Label(box, text=detail, bg=SIDE, fg=FAINT, font=self.f_small,
                       anchor="w", justify="left")
        lbl.pack(fill="x")
        return dot, lbl

    def _set_dot(self, canvas, color):
        canvas.itemconfig(1, fill=color, outline=color)

    def _build_composer(self, composer):
        shell = tk.Frame(composer, bg=BORDER)
        shell.pack(fill="x")
        inner = tk.Frame(shell, bg=CARD)
        inner.pack(fill="x", padx=1, pady=1)
        # button first, then the expanding input - same rule as above
        self.btn_send = tk.Button(inner, text="Send", command=self._on_send,
                                  font=self.f_bold, bg=ACCENT, fg="#16150f",
                                  relief="flat", activebackground=ACCENT_DK,
                                  padx=18, pady=6, cursor="hand2", bd=0)
        self.btn_send.pack(side="right", padx=10, pady=10)
        self.input = tk.Text(inner, height=2, bg=CARD, fg=TEXT, font=self.f_body,
                             wrap="word", bd=0, padx=14, pady=11,
                             insertbackground=ACCENT, selectbackground="#3d3b37",
                             highlightthickness=0)
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Return>", self._on_return)
        self.input.focus_set()
        tk.Label(composer, text="Enter to send   ·   Shift+Enter for a new line",
                 bg=BG, fg=FAINT, font=self.f_small, anchor="w"
                 ).pack(fill="x", pady=(6, 0))

    def _welcome(self):
        self._write("Ready when you are. Try:\n", "sys")
        for e in EXAMPLES:
            self._write(e + "\n", "hint")

    # -------------------------------------------------------------- view writes
    def _write(self, text, tag):
        self.view.config(state="normal")
        self.view.insert("end", text, tag)
        self.view.config(state="disabled")
        self.view.see("end")

    def _role(self, name, tag):
        self._write("\n%s\n" % name, tag)

    _MD = re.compile(r"\*\*(.+?)\*\*|`([^`\n]+)`")

    def _insert_md(self, text, base):
        """Light markdown: **bold** and `code`. Models emit it whether asked or not."""
        pos = 0
        for m in self._MD.finditer(text):
            if m.start() > pos:
                self._write(text[pos:m.start()], base)
            if m.group(1) is not None:
                self._write(m.group(1), (base, "b"))
            else:
                self._write(m.group(2), (base, "code"))
            pos = m.end()
        if pos < len(text):
            self._write(text[pos:], base)

    def _set_status(self, text, color=MUTED):
        self.lbl_status.config(text=text, fg=color)

    def _show_fix(self, show):
        if show:
            self.btn_fix.pack(side="right", padx=4)
        else:
            self.btn_fix.pack_forget()

    # ----------------------------------------------------------------- threading
    def _spawn(self, fn, *a):
        threading.Thread(target=self._guard, args=(fn,) + a, daemon=True).start()

    def _guard(self, fn, *a):
        """No traceback ever reaches the user - it goes to the transcript as prose."""
        try:
            fn(*a)
        except Exception as e:
            self.q.put(("error", "%s: %s" % (type(e).__name__, e)))
            self.q.put(("trace", traceback.format_exc()))
            self.q.put(("idle", None))

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.after(40, self._drain)

    def _handle(self, kind, payload):
        if kind == "status":
            text, color, fixable = payload
            self._set_status(text, color)
            self._show_fix(fixable)
        elif kind == "conn":
            which, color, detail = payload
            dot, lbl = ((self.dot_host, self.lbl_host) if which == "host"
                        else (self.dot_bridge, self.lbl_bridge))
            self._set_dot(dot, color)
            if detail:
                lbl.config(text=detail)
            if which == "bridge" and hasattr(self, "dot_app"):
                self._set_dot(self.dot_app, color)
        elif kind == "sys":
            self._write(payload + "\n", "sys")
        elif kind == "tool":
            self._write("  " + payload + "\n", "tool")
        elif kind == "tool_result":
            self._write("     " + payload + "\n", "tool")
        elif kind == "stream_start":
            self._role("AE AGENT", "role_asst")
            self._asst_start = self.view.index("end-1c")
            self._stream_buf = []
            self._stream_open = True
        elif kind == "token":
            self._stream_buf.append(payload)
            self._write(payload, "asst")
        elif kind == "stream_end":
            if self._stream_open:
                raw = "".join(self._stream_buf).rstrip()
                self.view.config(state="normal")
                self.view.delete(self._asst_start, "end-1c")
                self.view.config(state="disabled")
                self._insert_md(raw + "\n", "asst")
                self._stream_buf = []
                self._stream_open = False
        elif kind == "error":
            if self._stream_open:
                self._stream_open = False
            self._write(payload + "\n", "err")
        elif kind == "trace":
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "ae_agent_error.log"), "a", encoding="utf-8") as f:
                f.write("\n---- %s ----\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                                payload))
        elif kind == "ready":
            self._write("Connected and warmed up - replies land in a few seconds.\n",
                        "sys")
        elif kind == "idle":
            self.busy = False
            self.btn_send.config(text="Send", state="normal")

    # --------------------------------------------------------------- preflight
    def _boot(self):
        self.q.put(("status", ("checking the inference host", MUTED, False)))
        ok, loaded, ids, err = eng.probe_models(self.host)
        if not ok:
            self.q.put(("status", ("inference host unreachable", ERRC, False)))
            self.q.put(("conn", ("host", ERRC,
                                 "%s\nunreachable" % pretty_host(self.host))))
            self.q.put(("error",
                        "Cannot reach the inference host at %s.\n"
                        "Check that the other PC is awake, Tailscale is up on both "
                        "ends, and LM Studio's server is started.\n(%s)"
                        % (self.host, err)))
            self.q.put(("idle", None))
            return
        model = eng.pick_model(loaded, ids, self.want_model)
        if not model:
            self.q.put(("status", ("host is serving no models", ERRC, False)))
            self.q.put(("conn", ("host", ERRC,
                                 "%s\nno models" % pretty_host(self.host))))
            self.q.put(("idle", None))
            return
        self.state_model, self.n_models = model, len(ids)
        self.llm = eng.LLM(self.host, model)
        self.q.put(("conn", ("host", OK, "%s\n%d models\n%s"
                             % (pretty_host(self.host), len(ids), clip(model, 24)))))

        self.q.put(("status", ("starting the After Effects bridge", MUTED, False)))
        try:
            self.mcp = eng.MCPClient("npx", ["-y", "@engine-room/after-effects-mcp"],
                                     quiet=True)
            self.mcp.initialize(timeout=75)
            allt = self.mcp.list_tools(timeout=45)
        except Exception as e:
            self.q.put(("status", ("bridge did not start", ERRC, False)))
            self.q.put(("conn", ("bridge", ERRC, "127.0.0.1:7777\nfailed to start")))
            self.q.put(("error",
                        "Could not start the After Effects bridge.\n"
                        "Usually this is npx on a cold cache, or Node missing from "
                        "PATH. Close this window and open it again - the second start "
                        "is normally quick. If it keeps failing, run this once in a "
                        "terminal to see the real error:\n"
                        "    npx -y @engine-room/after-effects-mcp --help\n\n(%s)" % e))
            self.q.put(("idle", None))
            return
        wanted = set()
        for g in eng.DEFAULT_GROUPS:
            wanted |= set(eng.GROUPS[g])
        self.tools = eng.to_openai_tools([t for t in allt if t["name"] in wanted])

        # Prefill dominates the first call - ~14k tokens of tool schemas takes about a
        # minute cold. Pay it here, at startup, against the exact prompt prefix a real
        # message will use, so the user's first question comes back in seconds.
        self.q.put(("status", ("warming up the model, about a minute", WARN, False)))
        try:
            self.llm.chat([{"role": "system", "content": CHAT_SYSTEM_PROMPT},
                           {"role": "user", "content": "Say ready."}],
                          self.tools, max_tokens=1)
        except Exception:
            pass  # warming is an optimisation; failing here is not fatal
        self._refresh_status()
        self.q.put(("ready", None))
        self.q.put(("idle", None))

    def _refresh_status(self):
        self.state_ae = eng.ae_running()
        n = len(self.tools)
        if self.state_ae:
            self.q.put(("status", ("connected", OK, False)))
            self.q.put(("conn", ("bridge", OK, "127.0.0.1:7777\n%d tools" % n)))
        else:
            self.q.put(("status", ("After Effects is not running", WARN, True)))
            self.q.put(("conn", ("bridge", WARN, "127.0.0.1:7777\nnot running")))

    def _on_fix(self):
        if self.busy:
            return
        self.busy = True
        self.btn_send.config(state="disabled")
        self._spawn(self._fix)

    def _fix(self):
        if not eng.ae_running():
            self.q.put(("sys", "Launching After Effects..."))
            self.q.put(("status", ("launching After Effects", WARN, False)))
            eng.launch_ae()
            for _ in range(60):
                if eng.ae_running():
                    self.q.put(("sys", "After Effects is connected."))
                    break
                time.sleep(2)
            else:
                self.q.put(("error",
                            "After Effects did not come up on port 7777 within two "
                            "minutes. If it is open, check that Window > Extensions "
                            "shows the ae-mcp panel."))
        self._refresh_status()
        self.q.put(("idle", None))

    # ------------------------------------------------------------------ sending
    def _on_return(self, ev):
        if ev.state & 0x0001:  # Shift+Enter = newline
            return None
        self._on_send()
        return "break"

    def _on_new(self):
        if self.busy:
            return
        self.messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        self.view.config(state="normal")
        self.view.delete("1.0", "end")
        self.view.config(state="disabled")
        self._welcome()

    def _on_send(self):
        if self.busy:
            return
        task = self.input.get("1.0", "end").strip()
        if not task:
            return
        if self.llm is None or self.mcp is None:
            self._write("Still starting up - give it a moment.\n", "err")
            return
        self.input.delete("1.0", "end")
        self._role("YOU", "role_user")
        self._write(task + "\n", "user")
        self.messages.append({"role": "user", "content": task})
        self.busy = True
        self.btn_send.config(text="...", state="disabled")
        self._set_status("working", WARN)
        self._spawn(self._turn)

    def _trim(self):
        if len(self.messages) > MAX_HISTORY + 1:
            keep = self.messages[1:][-MAX_HISTORY:]
            while keep and keep[0].get("role") == "tool":
                keep.pop(0)  # never open on an orphaned tool reply
            self.messages = self.messages[:1] + keep

    def _turn(self):
        for _ in range(MAX_STEPS):
            self._trim()
            started = {"v": False}

            def on_text(piece, s=started):
                if not s["v"]:
                    s["v"] = True
                    self.q.put(("stream_start", None))
                self.q.put(("token", piece))

            msg = self.llm.stream(self.messages, self.tools, on_text)
            self.q.put(("stream_end", None))
            self.messages.append(msg)

            calls = msg.get("tool_calls") or []
            if not calls:
                if not (msg.get("content") or "").strip():
                    self.q.put(("error", "The model returned an empty reply."))
                self._refresh_status()
                self.q.put(("idle", None))
                return

            for call in calls:
                fn = call["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    out = ("TOOL ERROR: arguments were not valid JSON (%s). "
                           "Re-issue with valid JSON." % e)
                    self.q.put(("tool", "%s  [bad arguments]" % name))
                else:
                    preview = json.dumps(args)
                    self.q.put(("tool", "%s %s" % (
                        name, preview[:110] + ("..." if len(preview) > 110 else ""))))
                    try:
                        out = eng.mcp_result_to_text(self.mcp.call_tool(name, args))
                    except Exception as e:
                        out = "TOOL ERROR: %s" % e
                    # collapse to one line - raw JSON's first line is often just "["
                    flat = " ".join(out.split())
                    self.q.put(("tool_result",
                                flat[:110] + ("..." if len(flat) > 110 else "")))
                self.messages.append({"role": "tool",
                                      "tool_call_id": call.get("id", name),
                                      "content": out})
        self.q.put(("sys", "Stopped - the agent hit its step limit for this turn."))
        self._refresh_status()
        self.q.put(("idle", None))

    def _quit(self):
        try:
            if self.mcp:
                self.mcp.close()
        except Exception:
            pass
        self.destroy()


_LOCK = None


def claim_single_instance(port=57733):
    """One app, one MCP server. A second copy would spawn a rival bridge."""
    global _LOCK
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        _LOCK = s
        return True
    except OSError:
        s.close()
        return False


def main():
    try:  # crisp text on a high-DPI display
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    if not claim_single_instance():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("AE Agent", "AE Agent is already running.\n\n"
                                        "Look for its window on the taskbar.")
        root.destroy()
        return
    Chat().mainloop()


if __name__ == "__main__":
    main()
