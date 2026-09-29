"""Checking GitHub for updates and pulling them. Split out of
core/chat.py (docs/CODEMAP.md) as a mixin."""
import threading
from tkinter import messagebox

import studio_update as updater

from core.chat import UPDATE_EVERY_MS, APP_NAME, ELLIPSIS, pretty_host, this_pc


class ChatUpdatesMixin:
    def _update_tick(self):
        """Look at GitHub now and again while the window is open. The header's
        Update button appears when there is something to pull."""
        if self.closing:
            return
        self._check_updates(quiet=True)
        self.update_timer = self.after(UPDATE_EVERY_MS, self._update_tick)

    def _check_updates(self, quiet=True):
        def work():
            try:
                st = updater.check()
            except Exception as e:        # a git that hangs past its timeout
                st = {"behind": 0, "ahead": 0, "commits": [], "upstream": "",
                      "problem": "Could not check GitHub: %s" % e}
            self.q.put(("update", None, (st, quiet)))
        threading.Thread(target=work, daemon=True).start()

    def _show_update(self, st, quiet):
        if st["behind"] and not self.updating:
            self.btn_update.set(text="Update (%d)" % st["behind"])
            self.btn_update.pack(side="right", padx=(6, 0), after=self.btn_hist)
        else:
            self.btn_update.pack_forget()
        if quiet:
            return
        if st["problem"]:
            messagebox.showwarning(APP_NAME, st["problem"], parent=self)
        elif st["behind"]:
            self._on_update(st)
        else:
            messagebox.showinfo(APP_NAME, "Up to date with %s." % st["upstream"],
                                parent=self)

    def _on_update(self, st=None):
        """Say what is new, then fast-forward. Only ever a fast-forward: work
        on this PC is never merged over (studio_update.py)."""
        if self.updating:
            return
        if st is None:
            self._check_updates(quiet=False)
            return
        new = st["commits"][:12]
        more = len(st["commits"]) - len(new)
        text = ("%d new change%s on GitHub (%s):\n\n%s%s\n\nUpdate now?"
                % (st["behind"], "" if st["behind"] == 1 else "s", st["upstream"],
                   "\n".join("\u2022 " + c for c in new),
                   "\n\u2026and %d more" % more if more > 0 else ""))
        if not messagebox.askyesno(APP_NAME, text, parent=self):
            return
        self.updating = True
        self.btn_update.set(text="Updating" + ELLIPSIS)

        def work():
            try:
                result = updater.pull()
            except Exception as e:
                result = (False, "The update failed: %s" % e)
            self.q.put(("updated", None, result))
        threading.Thread(target=work, daemon=True).start()

    def _updated(self, changed, msg):
        self.updating = False
        if not changed:
            self.btn_update.set(text="Update")
            messagebox.showwarning(APP_NAME, msg, parent=self)
            return
        self.btn_update.pack_forget()
        if messagebox.askyesno(APP_NAME, "Updated. This window is still running the "
                               "old version.\n\nRestart %s now?" % APP_NAME, parent=self):
            self.restart = True
            self._quit()

    def _about(self):
        messagebox.showinfo(
            APP_NAME,
            "%s\n\nOne tab per creative app, driven by a local model.\n\n"
            "Inference: %s\nBridges run here on %s."
            % (APP_NAME, pretty_host(self.host), this_pc()), parent=self)

    # ------------------------------------------------------------------- layout
