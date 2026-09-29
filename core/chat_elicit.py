"""A bridge's MCP elicitation, answered by the user: the approval card,
never by the model. Split out of core/chat.py (docs/CODEMAP.md) as a
mixin. `_elicit` itself runs off the Tk main thread - moved unchanged,
do not touch its threading."""
import threading
import tkinter as tk

import core.agent as eng

from core.chat import Pill


class ChatElicitMixin:
    def _elicit(self, s, params):
        """A bridge's elicitation, answered by the user - never by the model.
        Runs on the MCP client's thread: the form goes up through the queue
        and this waits for a click. A Stop, a closed tab or a New chat is the
        user cancelling, and the form is greyed with that said."""
        box = {"done": threading.Event()}
        self.q.put(("elicit", s.event_id, (params, box)))
        while not box["done"].wait(0.25):
            if s.closed or s.cancel.is_set():
                if box.setdefault("result", {"action": "cancel"})["action"] == "cancel":
                    self.q.put(("elicit_done", s.event_id, (box, "Stopped")))
                break
        return box.get("result") or {"action": "cancel"}

    def _show_elicit(self, s, payload):
        """The bridge's question as a card: its message, the diff or command a
        coding agent wants to apply, a note field, and a button per choice.
        A click answers the bridge directly; the form then stays, greyed, with
        what was chosen, so the transcript records every decision."""
        params, box = payload
        if "result" in box:
            return                                    # cancelled before it was drawn
        self._hide_hero(s)
        view = s.view
        shell, frame = self._form_card(view)
        wrap = self._px(560)
        shown = (params.get("_meta") or {}).get("studio/approval") or {}
        message = params.get("message") or "The bridge asks:"
        head = message.split("\n")[0] if shown.get("command") else message
        self._skin(tk.Label(frame, text=head, font=self.f_body, wraplength=wrap,
                            justify="left", anchor="w"),
                   bg="card", fg="text").pack(fill="x", pady=(0, self._px(6)))
        if shown.get("diff"):
            self._diff_box(frame, shown["diff"]).pack(fill="x", pady=(0, self._px(6)))
        elif shown.get("command"):
            self._skin(tk.Label(frame, text=shown["command"], font=self.f_mono,
                                wraplength=wrap, justify="left", anchor="w",
                                padx=self._px(8), pady=self._px(6)),
                       bg="bg", fg="code").pack(fill="x", pady=(0, self._px(6)))

        schema = params.get("requestedSchema") or {}
        fields = eng.elicit_fields(schema)
        choice = next((f for f in fields if f[2]), None)   # the field the buttons answer
        entries, checks, widgets = {}, {}, []
        for name, spec, choices in fields:
            if choices:
                continue
            title = spec.get("title") or name
            if spec.get("type") == "boolean":
                var = tk.BooleanVar(master=self, value=bool(spec.get("default")))
                box_ = tk.Checkbutton(frame, text=title, variable=var, font=self.f_body,
                                      anchor="w", cursor="hand2", bd=0,
                                      highlightthickness=0)
                self._skin(box_, bg="card", fg="text", selectcolor="bg",
                           activebackground="card", activeforeground="text")
                box_.pack(fill="x")
                checks[name] = var
                widgets.append(box_)
            else:
                self._skin(tk.Label(frame, text=title, font=self.f_small, anchor="w"),
                           bg="card", fg="faint").pack(fill="x")
                entry = tk.Entry(frame, font=self.f_ui, relief="flat",
                                 highlightthickness=1)
                self._skin(entry, bg="bg", fg="text", insertbackground="text",
                           highlightbackground="border", highlightcolor="accent")
                entry.pack(fill="x", pady=(0, self._px(6)), ipady=self._px(3))
                entries[name] = entry
                widgets.append(entry)

        def answer(value=None, action="accept"):
            if "result" in box:
                return
            content = {}
            if choice is not None and value is not None:
                content[choice[0]] = value
            for name, entry in entries.items():
                text = entry.get().strip()
                if text:
                    content[name] = text
            for name, var in checks.items():
                content[name] = bool(var.get())
            if action == "accept" and choice is not None and value is None \
                    and choice[0] in (schema.get("required") or []):
                return                                  # a choice is still owed
            res = {"action": action}
            if action == "accept":
                res["content"] = content
            box["result"] = res
            box["done"].set()
            label = dict(choice[2]).get(value, "") if choice and value is not None else ""
            self._settle_elicit(s, box, label or ("Stopped" if action == "cancel" else
                                                  "Declined" if action == "decline" else "Sent"))

        row = self._skin(tk.Frame(frame), bg="card")
        row.pack(fill="x", pady=(self._px(2), 0))
        if choice is not None:
            for value, label in choice[2]:
                kind = ("accent" if value == choice[2][0][0] else
                        "ghost" if str(value) in ("reject", "deny", "no") else "quiet")
                pill = self._button(row, label, lambda v=value: answer(v), kind=kind,
                                    bg="card", font=self.f_body, padx=self._px(14),
                                    pady=self._px(5), r=self._px(11))
                pill.pack(side="left", padx=(0, self._px(6)), pady=(0, self._px(2)))
                widgets.append(pill)
        if choice is None:
            send = self._button(row, "Send", answer, kind="accent", bg="card",
                                font=self.f_body, padx=self._px(14),
                                pady=self._px(5), r=self._px(11))
            send.pack(side="left", padx=(0, self._px(6)))
            widgets.append(send)
        stop = self._button(row, "Stop", lambda: answer(action="cancel"), kind="ghost",
                            bg="card", font=self.f_small, padx=self._px(9),
                            pady=self._px(3), r=self._px(9))
        stop.pack(side="right")
        widgets.append(stop)
        status = self._skin(tk.Label(frame, text="", font=self.f_small, anchor="w"),
                            bg="card", fg="faint")
        status.pack(fill="x")
        s.elicits.append({"box": box, "widgets": widgets, "status": status})
        self._place_form(view, shell)
        for entry in entries.values():
            entry.bind("<Return>", lambda _e: choice is None and answer())

    def _diff_box(self, parent, diff, max_lines=18):
        """A unified diff, read-only, added lines green and removed ones red.
        The long header OpenCode writes (Index:, ===, ---/+++ with the full
        path) is left out: the card already says which file."""
        lines = [l for l in diff.splitlines()
                 if not l.startswith(("Index:", "--- ", "+++ "))
                 and not (l and not l.strip("="))]
        wrap = self._skin(tk.Frame(parent), bg="bg")
        text = tk.Text(wrap, font=self.f_mono, wrap="none", relief="flat", bd=0,
                       height=min(max(len(lines), 1), max_lines), width=90,
                       padx=self._px(8), pady=self._px(6), highlightthickness=0,
                       cursor="arrow")
        self._skin(text, bg="bg", fg="code", insertbackground="bg")

        def colour():
            text.tag_configure("add", foreground=self.C["ok"])
            text.tag_configure("del", foreground=self.C["err"])
            text.tag_configure("hunk", foreground=self.C["faint"])

        colour()
        self._repaint_on_theme(text, colour)
        for l in lines:
            tag = ("add" if l.startswith("+") else "del" if l.startswith("-") else
                   "hunk" if l.startswith("@@") else ())
            text.insert("end", l + "\n", tag)
        text.config(state="disabled")
        if len(lines) > max_lines:
            bar = tk.Scrollbar(wrap, orient="vertical", command=text.yview)
            text.config(yscrollcommand=bar.set)
            bar.pack(side="right", fill="y")
        text.pack(side="left", fill="x", expand=True)
        return wrap

    def _settle_elicit(self, s, box, said):
        """Grey an answered form and say what was answered."""
        for form in list(s.elicits):
            if form["box"] is not box:
                continue
            s.elicits.remove(form)
            for w in form["widgets"]:
                try:
                    if isinstance(w, Pill):
                        w.set(state="disabled")
                    else:
                        w.config(state="disabled")
                except tk.TclError:
                    pass
            try:
                form["status"].config(text=said)
            except tk.TclError:
                pass

