"""A per-tab activity strip that stays visible while the transcript scrolls."""
import tkinter as tk

from core.activity import Activity, STATES, tool_presentation, activity_heading


class ActivityView:
    def __init__(self, parent, *, skin, button, selectable, font, mono, small, px):
        self.model = Activity()
        self.skin = skin
        self.pad, self.gap = px(16), px(8)
        self.font = font
        self.open = False
        self.frame = skin(tk.Frame(parent), bg="bg")
        header = skin(tk.Frame(self.frame), bg="bg")
        header.pack(fill="x", padx=px(16), pady=(px(8), px(4)))
        self.toggle = button(header, "Tool activity (0)", self._toggle, kind="ghost",
                             font=small, padx=px(8), pady=px(2))
        self.toggle.pack(side="right")
        self.label = skin(tk.Label(header, font=font, anchor="w", justify="left"), bg="bg", fg="muted")
        self.label.pack(side="left", fill="x", expand=True)
        self.label.bind("<Configure>", lambda e: self.label.config(wraplength=max(px(80), e.width)))
        self.body = skin(tk.Frame(self.frame), bg="bg")
        scroll = tk.Scrollbar(self.body, width=px(11), bd=0, highlightthickness=0)
        skin(scroll, bg="bg", troughcolor="bg", activebackground="faint")
        scroll.pack(side="right", fill="y")
        self.details = tk.Text(self.body, height=6, wrap="word", font=mono, state="disabled",
                               bd=0, highlightthickness=0, yscrollcommand=scroll.set)
        skin(self.details, bg="card", fg="text")
        selectable(self.details)
        self.details.pack(fill="both", expand=True)
        self.details.tag_configure("title", font=font, spacing1=px(8), spacing3=px(3))
        scroll.config(command=self.details.yview)

    def _toggle(self):
        self.open = not self.open
        if self.open:
            self.body.pack(fill="x", padx=self.pad, pady=(0, self.gap))
        else:
            self.body.pack_forget()

    def handle(self, kind, payload=None):
        if self.model.handle(kind, payload):
            self.paint()

    def paint(self):
        if self.model.active or self.model.count:
            # Pack the fixed-height strip before the expanding transcript.
            if not self.frame.winfo_manager():
                other = next((w for w in self.frame.master.pack_slaves() if w is not self.frame), None)
                self.frame.pack(side="top", fill="x", **({"before": other} if other else {}))
        else:
            self.frame.pack_forget()
        self.label.config(text=activity_heading(self.model))
        self.skin(self.label, bg="bg", fg="err" if self.model.interrupted else "muted")
        self.toggle.set(text="Tool activity (%d)" % self.model.count)
        old = self.details.yview()
        self.details.config(state="normal")
        self.details.delete("1.0", "end")
        if self.model.omitted:
            self.details.insert("end", "%d earlier calls omitted from this view.\n" % self.model.omitted)
        for item in self.model.items:
            title = tool_presentation(item["name"], item["arguments"])[0]
            self.details.insert("end", "%s · %s\n" % (STATES.get(item["status"], "Outcome unknown"), title), "title")
            self.details.insert("end", "Tool: %s\nArguments: %s\n" % (item["name"], item["arguments"]))
            if item["progress"]:
                self.details.insert("end", "Progress: " + item["progress"] + "\n")
            if item["text"]:
                self.details.insert("end", "Result: " + item["text"] + "\n")
        if not self.model.items:
            self.details.insert("end", "No tools called in this turn.")
        self.details.config(state="disabled")
        if old and old[1] < .99:
            self.details.yview_moveto(old[0])
        else:
            self.details.see("end")

    def clear(self):
        self.model = Activity()
        self.open = False
        self.body.pack_forget()
        self.paint()
