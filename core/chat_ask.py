"""studio_ask rendered as a form in the transcript, and the card chrome
(_form_card/_place_form) it shares with chat_elicit.py's bridge
questions. Split out of core/chat.py (docs/CODEMAP.md) as a mixin."""
import tkinter as tk

from core.chat import Pill, rounded


class ChatAskMixin:
    def _show_ask(self, s, asked):
        """studio_ask, as a form in the transcript: a button per option - boxes
        to tick when several may apply - and one for an answer of the user's
        own. A click is the user's next message; the form then stays, greyed,
        so the transcript still shows what was asked and chosen."""
        self._settle_ask(s)
        view = s.view
        shell, frame = self._form_card(view)
        wrap = self._px(520)
        self._skin(tk.Label(frame, text=asked["question"], font=self.f_body,
                            wraplength=wrap, justify="left", anchor="w"),
                   bg="card", fg="text").pack(fill="x", pady=(0, self._px(6)))
        buttons = []

        def answer(text):
            if self.sessions.get(s.id) is not s or s.busy:
                return
            self._send(s, text)

        if asked.get("multiple"):
            picks = []
            for option in asked["options"]:
                var = tk.BooleanVar(master=self, value=False)
                text = option["label"]
                if option.get("description"):
                    text += "  -  " + option["description"]
                box = tk.Checkbutton(frame, text=text, variable=var, font=self.f_body,
                                     anchor="w", justify="left", wraplength=wrap,
                                     cursor="hand2", bd=0, highlightthickness=0)
                self._skin(box, bg="card", fg="text", selectcolor="bg",
                           activebackground="card", activeforeground="text")
                box.pack(fill="x")
                picks.append((option["label"], var))
                buttons.append(box)

            def send_picks():
                chosen = [label for label, var in picks if var.get()]
                if chosen:
                    answer("; ".join(chosen))
            go = self._button(frame, "Send these", send_picks, kind="accent",
                              bg="card", font=self.f_body, padx=self._px(14),
                              pady=self._px(5), r=self._px(11))
            go.pack(anchor="w", pady=(self._px(6), 0))
            buttons.append(go)
        else:
            for option in asked["options"]:
                # A full-width row that is also a button: `anchor="w"` gives a
                # pill the packer's width and a left-read label that wraps.
                button = self._button(
                    frame, option["label"], lambda t=option["label"]: answer(t),
                    kind="option", bg="card", font=self.f_body, anchor="w",
                    padx=self._px(11), pady=self._px(5), r=self._px(10))
                button.pack(fill="x", pady=(0, self._px(2)))
                buttons.append(button)
                if option.get("description"):
                    self._skin(tk.Label(frame, text=option["description"], font=self.f_small,
                                        wraplength=wrap, justify="left", anchor="w"),
                               bg="card", fg="faint").pack(fill="x", padx=(10, 0),
                                                           pady=(0, self._px(4)))
        other = self._button(frame, "Something else…", self.input.focus_set,
                             kind="ghost", bg="card", font=self.f_small,
                             padx=self._px(9), pady=self._px(3), r=self._px(9))
        other.pack(anchor="w", pady=(self._px(4), 0))
        buttons.append(other)
        self._place_form(view, shell)
        s.ask_buttons = buttons

    def _form_card(self, view):
        """(shell, frame): a rounded card for a form in the transcript. The
        card is drawn, so its corners can be round; the form is a frame on top
        of it, inset clear of the curve. Same shape as a tab chip or a rail
        row - see `_make_tab`."""
        PAD, R = self._px(10), self._px(14)
        shell = tk.Canvas(view, highlightthickness=0, bd=0)
        self._skin(shell, bg="bg")
        frame = self._skin(tk.Frame(shell, padx=self._px(4), pady=self._px(2)),
                           bg="card")
        shell.create_window(PAD, PAD, window=frame, anchor="nw")

        def paint(_ev=None):
            w = frame.winfo_reqwidth() + 2 * PAD
            h = frame.winfo_reqheight() + 2 * PAD
            shell.config(width=w, height=h)
            shell.delete("card")
            fill = self.C["card"]
            rounded(shell, 0, 0, w, h, R, fill=fill, outline=fill, tags="card")
            shell.tag_lower("card")

        frame.bind("<Configure>", paint)
        self._repaint_on_theme(shell, paint)
        return shell, frame

    def _place_form(self, view, shell):
        view.config(state="normal")
        view.insert("end", "\n")
        view.window_create("end", window=shell, padx=self._px(4))
        view.insert("end", "\n")
        view.config(state="disabled")
        # The form has no height until Tk lays it out, so a scroll now stops
        # short of it; scroll again once it has one.
        view.see("end")
        view.after_idle(lambda: view.winfo_exists() and view.see("end"))

    def _settle_ask(self, s):
        """Grey the question form once it has been answered - by a click or by
        a typed message - so it cannot send a second answer."""
        for button in s.ask_buttons:
            try:
                # A Pill has to be repainted to look disabled; the tick boxes
                # beside it are still Tk's own and answer to config().
                if isinstance(button, Pill):
                    button.set(state="disabled")
                else:
                    button.config(state="disabled")
            except tk.TclError:
                pass
        s.ask_buttons = []

    # ------------------------------------------------ a bridge asks the user

