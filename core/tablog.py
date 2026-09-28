"""Each tab's own log: which tab a line of the activity log belongs to, and
the recent lines of every tab kept in memory for its Log window.

The activity log (`studio_doctor.start_activity_log`) said what the app did,
but not for whom: six tabs' tool calls, model requests and errors ran
together in one file. Every record now carries a `tab` - the app id of the
tab it happened in, or "-" for the window's own business - stamped by
`Stamp`, and `BOOK` keeps the last `KEEP` lines of each tab so a tab can show
its own without reading the file back.

A record gets its tab one of two ways:

- **Explicitly**, `extra={"tab": app_id}` - what `log()` does, and what the
  window's event pump uses, since it knows which tab every event is for.
- **From the thread.** A tab's work runs on threads `Chat._spawn` starts,
  and those run inside `working_for(app_id)`, so a module logging to
  `logging.getLogger("studio.<name>")` from there - the engine's tool calls
  and model requests - lands in the right tab without knowing tabs exist.
  A thread started from inside one does not inherit it; pass `current()` to
  it and enter `working_for` again, as `MCPClient` does for a bridge's stderr.

No tkinter: the tests and the CLI import this, and `--doctor` must work when
the window will not.
"""

import collections
import logging
import threading
import time

KEEP = 2000                       # lines kept per tab; the file has the rest
WINDOW = "-"                      # the tab of a line that belongs to no tab

_where = threading.local()


def current():
    """The tab this thread is working for, or None."""
    return getattr(_where, "tab", None)


class working_for:
    """`with working_for(app_id):` - records logged on this thread belong to
    that tab. Nests, and restores what was there, so a tab's work calling
    into the window's is never misfiled once it returns."""

    def __init__(self, tab):
        self.tab = tab

    def __enter__(self):
        self.before = current()
        _where.tab = self.tab
        return self

    def __exit__(self, *exc):
        _where.tab = self.before
        return False


def tab_of(sid):
    """A tab's app id from any of the ids events carry: the id itself, an
    `(app_id, generation)` event id, or None for the window."""
    if isinstance(sid, tuple):
        sid = sid[0] if sid else None
    return sid or None


LOG = logging.getLogger("studio.tab")


def log(tab, text, level=logging.INFO):
    """One line for `tab` (an app id, or None for the window)."""
    LOG.log(level, "%s", text, extra={"tab": tab or WINDOW})


class Stamp(logging.Filter):
    """Gives every record a `tab`, so a formatter can print `%(tab)s` and the
    book can file it. Put on a handler, not a logger: a logger's filters see
    only what is logged on that logger itself, never its children's."""

    def filter(self, record):
        if not getattr(record, "tab", None):
            record.tab = current() or WINDOW
        return True


class Book(logging.Handler):
    """The last `KEEP` lines of every tab, for its Log window. Each listener
    is called with `(tab, entry)` for each new line from whatever thread
    logged it - a window points one at its queue, never at a widget. More
    than one, because a second window (the tests build several) must not
    silence the first."""

    def __init__(self, keep=KEEP):
        super().__init__(logging.INFO)
        self.keep = keep
        self.tabs = {}
        self.listeners = []
        self.addFilter(Stamp())

    def listen(self, fn):
        if fn is not None and fn not in self.listeners:
            self.listeners.append(fn)

    def unlisten(self, fn):
        if fn in self.listeners:
            self.listeners.remove(fn)

    def emit(self, record):
        try:
            text = record.getMessage()
        except Exception:
            text = str(record.msg)
        entry = {"time": record.created, "level": record.levelname,
                 "source": record.name, "text": text}
        tab = record.tab
        with self.lock:
            lines = self.tabs.get(tab)
            if lines is None:
                lines = self.tabs[tab] = collections.deque(maxlen=self.keep)
            lines.append(entry)
        for listener in list(self.listeners):
            try:
                listener(tab, entry)
            except Exception:
                pass              # a closed window's queue costs the line, not the caller

    def lines(self, tab):
        with self.lock:
            return list(self.tabs.get(tab or WINDOW, ()))

    def clear(self, tab):
        with self.lock:
            self.tabs.pop(tab or WINDOW, None)


BOOK = Book()


def install(on_line=None):
    """Hang the book on the app's logger (once), and let a window listen;
    the window calls `BOOK.unlisten` when it goes. The logger is opened to
    INFO, which `start_activity_log` also does, so a window the tests build
    without a log file still fills its tabs' logs."""
    logger = logging.getLogger("studio")
    if BOOK not in logger.handlers:
        logger.addHandler(BOOK)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    BOOK.listen(on_line)
    return BOOK


def format_line(entry):
    """`12:04:31  WARN  agent  tool comfy_generate failed ...` - the Log
    window's line and its Copy's."""
    source = entry["source"]
    if source.startswith("studio."):
        source = source[len("studio."):]
    level = {"WARNING": "WARN", "CRITICAL": "CRIT"}.get(entry["level"], entry["level"])
    return "%s  %-5s %-6s %s" % (time.strftime("%H:%M:%S", time.localtime(entry["time"])),
                                 level, "" if source == "tab" else source, entry["text"])
