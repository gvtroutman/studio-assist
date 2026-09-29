"""Connect an MCP bridge by hand: the dialog and its registry writes.
Split out of core/chat.py (docs/CODEMAP.md) as a mixin."""
import os
import shutil
import tkinter as tk

import core.agent as eng

from core.chat import SUGGESTED_BRIDGES


class ChatBridgeDialogMixin:
    def _bridge_dialog(self, row=None, spec=None):
        """
        Connect any MCP stdio bridge as an app tab. `row` is a sidebar app with
        no bridge written here (the name, program and process are filled in);
        `spec` is an existing hand-entered bridge to edit. The dialog is built
        by this method and applied by `_save_bridge`, so a test can drive it
        without posting a window.
        """
        win = tk.Toplevel(self)
        win.title("Connect an MCP bridge")
        win.resizable(False, False)
        win.transient(self)
        self._skin(win, bg="bg")
        body = self._skin(tk.Frame(win), bg="bg")
        body.pack(fill="both", expand=True, padx=22, pady=18)

        name = (spec.name if spec else row["name"] if row else "")
        exe = (spec.exe_path if spec else (row or {}).get("exe") or "")
        probe = spec.probe if spec else ("process:" + os.path.basename(exe) if exe else "")
        command = " ".join([spec.command] + [
            ('"%s"' % a if " " in a else a) for a in spec.args]) if spec else ""
        if spec and " " in spec.command:
            command = '"%s"' % spec.command + command[len(spec.command):]

        self._cap(body, "BRIDGE", bg="bg").pack(fill="x", pady=(0, 8))
        intro = ("Any MCP server that speaks over stdio can drive a tab. Enter the command "
                 "line that starts it - the same one its README puts in an MCP client "
                 "config. Its tools and its own instructions are read when the tab opens.")
        lbl = tk.Label(body, text=intro, font=self.f_small, anchor="w", justify="left",
                       wraplength=self._px(460))
        self._skin(lbl, bg="bg", fg="muted")
        lbl.pack(fill="x", pady=(0, 12))

        fields = {}

        def field(key, label, value, hint):
            self._skin(tk.Label(body, text=label, font=self.f_ui, anchor="w"),
                       bg="bg", fg="text").pack(fill="x")
            var = tk.StringVar(value=value)
            entry = self._entry(body, var)
            entry.master.pack(fill="x", pady=(3, 2))
            h = tk.Label(body, text=hint, font=self.f_small, anchor="w", justify="left",
                         wraplength=self._px(460))
            self._skin(h, bg="bg", fg="faint")
            h.pack(fill="x", pady=(0, 10))
            fields[key] = var
            return entry

        first = field("name", "App", name, "What the tab and the sidebar call it.")
        cmd = field("command", "Command line", command,
              'e.g.  npx -y some-mcp-server   or   "C:\\path\\python.exe" server.py')
        field("exe", "Program (optional)", exe,
              "Path of the app's .exe, so the Start button can launch it and the tab "
              "gets its icon.")
        field("probe", "Running check (optional)", probe,
              "process:<Name.exe>, port:<number> or url:<http://...> - how to tell the "
              "app is up. Leave empty if the bridge itself is the only evidence.")
        if row and row["name"] in SUGGESTED_BRIDGES:
            tip = tk.Label(body, text=SUGGESTED_BRIDGES[row["name"]], font=self.f_small,
                           anchor="w", justify="left", wraplength=self._px(460))
            self._skin(tip, bg="bg", fg="muted")
            tip.pack(fill="x", pady=(0, 10))

        err = tk.Label(body, text="", font=self.f_small, anchor="w", justify="left",
                       wraplength=self._px(460))
        self._skin(err, bg="bg", fg="err")
        err.pack(fill="x")

        def save():
            values = {k: v.get() for k, v in fields.items()}
            problem = self._save_bridge(values, replacing=spec)
            if problem:
                err.config(text=problem)
            else:
                win.destroy()

        row_b = self._skin(tk.Frame(body), bg="bg")
        row_b.pack(fill="x", pady=(12, 0))
        ok = self._button(row_b, "Connect" if spec is None else "Save", save,
                          kind="accent")
        ok.pack(side="right")
        cancel = self._button(row_b, "Cancel", win.destroy)
        cancel.pack(side="right", padx=(0, 8))
        win.bind("<Return>", lambda ev: save())
        win.bind("<Escape>", lambda ev: win.destroy())
        (first if not name else cmd).focus_set()
        return win

    def _save_bridge(self, values, replacing=None):
        """
        Register a bridge from the dialog's values and open its tab. Returns a
        sentence for the dialog when the values will not do, else None. The
        command line is split the way a shell would; the program and the
        running check are optional.
        """
        name = (values.get("name") or "").strip()
        line = (values.get("command") or "").strip()
        exe = (values.get("exe") or "").strip().strip('"')
        probe = (values.get("probe") or "").strip()
        if not name:
            return "Give the app a name."
        if not line:
            return "Enter the command line that starts the bridge."
        owner = eng.APPS_BY_ID.get(eng.DRIVABLE.get(name))
        if owner is not None and not owner.custom:
            return "%s already has a bridge written into this program; give this one another name." % name
        try:
            command, args = eng.split_command(line)
        except ValueError as e:
            return "That command line does not parse: %s" % e
        if not (shutil.which(command) or os.path.isfile(command)):
            return "%r is not on PATH and is not a file. Use the full path." % command
        if exe and not os.path.isfile(exe):
            return "There is no program at %s." % exe
        if probe and probe.partition(":")[0] not in ("process", "port", "url"):
            return "The running check must start with process:, port: or url:."
        if replacing is not None:
            if replacing.id in self.sessions:
                self._close_tab(replacing.id)
            eng.remove_bridge(replacing.id)
        rec = {"name": name, "command": command, "args": args, "exe": exe, "probe": probe}
        if replacing is not None:
            rec["id"] = replacing.id
        spec = eng.bridge_from_record(rec)
        if spec is None:
            return "Those values do not make a bridge."
        if spec.id in self.sessions:
            self._close_tab(spec.id)      # an older entry with the same id
        eng.add_bridge(spec)
        self._remember_bridges()
        self.detected = eng.detect_apps()
        self._build_apps()
        self._add_tab(spec.id)
        return None

    def _forget_bridge(self, spec):
        if spec.id in self.sessions:
            self._close_tab(spec.id)
        eng.remove_bridge(spec.id)
        self._remember_bridges()
        self.detected = eng.detect_apps()
        self._build_apps()

    def _remember_bridges(self):
        self.prefs.set(bridges=[a.record() for a in eng.custom_bridges()])

    # ------------------------------------------------------------- preferences
