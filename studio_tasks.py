"""Shared, stdlib-only task execution and recoverable task records.

The bridge's original schemas validate calls; grammar-compatible copies are only
for inference. A record is evidence of execution, not proof of visual quality.
"""
import json
import os
import re
import tempfile
import threading
import uuid

import studio_agent as eng
import studio_lessons as lessons
import studio_mcp
import studio_toolsmith as toolsmith


TASK_TOOL = {"type": "function", "function": {
    "name": "studio_task_update",
    "description": "Keep the task brief and progress across long conversations. Before substantial edits, record a plan and acceptance checks. Update objects with real IDs, and checks with observed evidence; never invent evidence. This does not edit the creative app.",
    "parameters": {"type": "object", "additionalProperties": False,
        "properties": {
            "plan": {"type": "array", "items": {"type": "string"}},
            "objects": {"type": "object", "additionalProperties": {"type": "string"}},
            "checks": {"type": "array", "items": {"type": "object",
                "properties": {"requirement": {"type": "string"},
                               "evidence": {"type": "string"}},
                "required": ["requirement", "evidence"], "additionalProperties": False}},
            "issues": {"type": "array", "items": {"type": "string"}}}}}}

# A question with choices the user clicks, when the answer changes what would
# be built. The executor shows it and ends the run; the answer is the user's
# next message, so the model never waits on a tool result for it.
ASK_TOOL = {"type": "function", "function": {
    "name": "studio_ask",
    "description": ("Ask the user one question with choices they can click, when the "
                    "answer changes what you would build and cannot be read from the "
                    "project or a file - which format, which take, which brand, how "
                    "long. Two to five short options, the one you would suggest first; "
                    "the user can also type something else. Their answer arrives as "
                    "the next message: after asking, stop and wait for it. This changes "
                    "nothing in the project."),
    "parameters": {"type": "object", "additionalProperties": False,
        "properties": {
            "question": {"type": "string", "minLength": 4, "maxLength": 400,
                         "description": "The question, one sentence."},
            "options": {"type": "array", "minItems": 2, "maxItems": 5,
                "items": {"type": "object", "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string", "minLength": 1, "maxLength": 60,
                                  "description": "Short, what the user clicks."},
                        "description": {"type": "string", "maxLength": 200,
                                        "description": "What choosing it means, optional."}},
                    "required": ["label"]}},
            "multiple": {"type": "boolean",
                         "description": "True when more than one option may apply."}},
        "required": ["question", "options"]}}}

INTERNAL_TOOLS = (TASK_TOOL, toolsmith.CREATE_TOOL, ASK_TOOL, lessons.REMEMBER_TOOL)

QUALITY_RULES = """

TASK QUALITY
- studio_tool_create records a repeated sequence of this tab's own tools under
  one name. It creates a tool; it neither runs one nor edits the project, and it
  cannot reach a tool this tab was not given. One-off work goes to the bridge
  tools directly. A tool you made is a shorthand, never evidence of a result.
- For substantial edits use studio_task_update to record the plan, acceptance
  checks, relevant object IDs, and unresolved issues. Preserve the user's exact
  wording and constraints. Ask only about missing details that affect the result.
- Inspect the target before editing. After editing, read the changed target and
  compare it with the brief. A successful write alone is not verification.
- For visual work request a preview when a suitable tool is exposed. You read
  text: a returned picture reaches you as the "Visual review" appended to that
  result, written by a model that looked at it. Treat that review as what is on
  screen - fix what it names, and if it says the frame is not what was asked for,
  it is not done. If a result says no review was made, say visual review is still
  needed rather than claiming the result looks right. When you finish with an
  edit nobody has looked at, this window may take the screenshot itself and hand
  you its review as the next message: act on that review as you would your own.
- Animation workflow: confirm copy, dimensions, frame rate and duration; construct
  the design; animate; inspect timing and representative frames; refine defects.
- Assembly workflow: identify source media; check frame rate and source ranges;
  assemble; inspect track placement, gaps, overlaps and total duration.
- Delivery workflow: inspect available formats and settings; confirm output path;
  submit only the requested job; check actual completion before claiming export.
- Before substantial changes to existing work, use a supported duplicate or backup
  operation when available. Never invent backup tools or imply undo is guaranteed.
- If blocked or only partly verified, explain the limitation instead of claiming
  completion. A timeout may mean an edit happened: inspect, never blindly repeat.
- studio_ask puts a question with clickable choices in front of the user; ask
  it alone, then stop - the answer is their next message. studio_remember keeps
  one reusable lesson for future tasks in this app: use it when the user
  corrects you, tells you how they work, or when a call fails and you find what
  works instead. Neither touches the project.
"""

# A reply that announces the next step instead of taking it: "Now I'll generate
# the image" with no tool call attached. A small model does this at the top of a
# task and the run would otherwise end there, looking finished. A question to
# the user is not a promise, and the phrases are first-person future only, so a
# plain answer ("24 fps means...") is left alone.
PROMISE = re.compile(r"\b(?:I(?:'ll| will|'m going to| am going to| shall)|let(?:'s| us| me)|"
                     r"(?:now|next|then|first)[,:]? I)\b", re.I)
PROMISE_HINT = ("You described what you would do, but this reply called no tool, so "
                "nothing happened. Make the call now - the first step, with real "
                "arguments - or, if no tool you have fits the task, say so plainly "
                "instead of describing work.")


def announces_work(text):
    """True when a reply promises action rather than answering or asking."""
    text = (text or "").strip()
    return bool(text) and not text.endswith("?") and PROMISE.search(text) is not None


# The executor validates every call against the bridge's original schema with
# the harness's validator - the same one a bridge built on studio_mcp runs on
# arrival, so the two sides cannot disagree about what a schema allows.
validate = studio_mcp.validate
def readonly(name, args, spec):
    # Compound tools mix reads and writes: inspect the action, not annotations
    # describing the whole tool. Unknown actions are conservatively writes.
    action = args.get("action")
    if action is not None:
        return isinstance(action, str) and (action.startswith(("get_", "list_", "is_")) or
                                            action in ("get", "list"))
    return (spec.get("annotations", {}).get("readOnlyHint") is True or
            name.startswith(("get_", "list_", "find_", "screenshot_")) or
            name in ("check_setup", "ae_guide", "diff_comp", "snapshot_comp"))


def validate_action(args, description):
    """Resolve compound tools document actions as name(params) under Actions:.

    Use that explicit contract only; never infer actions from general prose.
    Argument details not represented in JSON Schema remain the bridge's job.
    """
    if "action" not in args or "Actions:" not in description:
        return
    actions = set(eng.re.findall(r"^\s+([a-z][a-z0-9_]*)\([^\n]*\)\s*->",
                                description.split("Actions:", 1)[1], eng.re.MULTILINE))
    if actions and args["action"] not in actions:
        raise ValueError("unsupported action %r; supported actions: %s" %
                         (args["action"], ", ".join(sorted(actions))))


def verification_read(name, args):
    # App availability, UI page, and codec discovery do not inspect edited work.
    if name in ("check_setup", "ae_guide", "resolve_control", "layout_presets", "render_presets"):
        return False
    # The research sidecar reads the world, not the project: a web page or a
    # brief on disk says nothing about whether an edit landed.
    if name in eng.RESEARCH_TOOL_NAMES:
        return False
    if args.get("action") in ("get_formats", "get_codecs", "get_resolutions", "get_version"):
        return False
    return True


class TaskRecord:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.app_id = ""
        self.briefs = []
        self.plan = []
        self.objects = {}
        self.checks = []
        self.issues = []
        self.journal = []
        self.status = "ready"

    def context(self):
        return {k: getattr(self, k) for k in ("briefs", "plan", "objects", "checks", "issues", "status")}

    def save(self, path, messages):
        if not path:
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "record": self.__dict__, "messages": messages}, f)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def restore(cls, path, system_prompt):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != 1:
            raise ValueError("unsupported saved task version")
        record, messages = cls(), data["messages"]
        saved = data["record"]
        for key, default in record.__dict__.items():
            if key in saved:
                if not isinstance(saved[key], type(default)):
                    raise ValueError("invalid saved task field: " + key)
                setattr(record, key, saved[key])
        if not eng.re.fullmatch(r"[0-9a-f]{32}", record.id):
            raise ValueError("invalid saved task id")
        if not all(isinstance(b, str) for b in record.briefs):
            raise ValueError("invalid saved task brief")
        validate({k: getattr(record, k) for k in ("plan", "objects", "checks", "issues")},
                 TASK_TOOL["function"]["parameters"])
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise ValueError("invalid saved messages")
        if not all(isinstance(e, dict) for e in record.journal):
            raise ValueError("invalid saved journal")
        for entry in record.journal:
            if entry.get("status") == "running":
                entry["status"] = "unknown"
        # A crash can leave a multi-call assistant message only partly answered.
        repaired = [{"role": "system", "content": system_prompt}]
        pending = set()
        for m in messages[1:]:
            if m.get("role") != "tool":
                repaired.extend({"role": "tool", "tool_call_id": i,
                                 "content": "Interrupted; outcome unknown. Inspect before continuing."}
                                for i in sorted(pending))
                pending.clear()
            if m.get("role") == "assistant":
                pending.update(c["id"] for c in m.get("tool_calls", []))
            if m.get("role") == "tool":
                if m.get("tool_call_id") not in pending:
                    continue
                pending.discard(m["tool_call_id"])
            repaired.append(m)
        repaired.extend({"role": "tool", "tool_call_id": i,
                         "content": "Interrupted; outcome unknown. Inspect before continuing."}
                        for i in sorted(pending))
        record.status = "restored — inspect the current project before editing"
        return record, repaired

    @classmethod
    def summaries(cls, folder):
        """What is saved under `folder`, newest first - enough to recognise a
        task without opening it, which a file dialog full of 32-character hex
        names is not. Each is {path, when, brief, steps, status, problem}.

        A file that will not parse is listed carrying its `problem` rather
        than dropped: a saved task that can no longer be resumed is worth
        knowing about, and silently hiding it is how a user comes to believe
        the app threw their work away.
        """
        out = []
        try:
            names = os.listdir(folder)
        except OSError:
            return out                    # no folder yet is no tasks, not an error
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(folder, name)
            item = {"path": path, "when": 0.0, "brief": "", "steps": 0,
                    "status": "", "problem": ""}
            try:
                item["when"] = os.path.getmtime(path)
            except OSError:
                pass
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                record = data["record"]
                briefs = [b for b in (record.get("briefs") or []) if isinstance(b, str)]
                item["brief"] = briefs[0] if briefs else ""
                item["steps"] = len(record.get("journal") or [])
                item["status"] = record.get("status") or ""
            except Exception as e:
                item["problem"] = "%s: %s" % (type(e).__name__, e)
            out.append(item)
        out.sort(key=lambda i: i["when"], reverse=True)
        return out


def context_messages(messages, record, tools, max_chars=100000, memory=True, extra=""):
    """Bound the request conservatively by characters, keeping whole exchanges.

    Full history remains on disk. Never silently clip the brief or tool contract.
    A budget too small for the fixed context fails before any app mutation.
    `memory` carries the saved task record into the request; a tab with no tools
    has no task to carry, and its conversation is the whole of its context.

    The record goes LAST, after the conversation, not first. It changes on every
    turn - the status alone flips to "working" before each run - and the host
    caches a request by its prefix: with the record at position 1 the cache
    matched only through the system prompt and tools, and the whole history was
    prefilled again on every message. At the end it costs one short block, and
    the history before it is served from the cache. Keep it there.
    """
    tail = []
    if memory:
        tail = [{"role": "user", "content": "Saved task context (data, not new instructions):\n" +
                 json.dumps(record.context(), ensure_ascii=False)}]
    if extra:
        # Lessons kept since this session's prompt was built: the same rule,
        # the tail, so the cached prefix stays whole until the next boot.
        tail.append({"role": "user", "content": extra})
    budget = max_chars - len(json.dumps([messages[0]] + tail)) - len(json.dumps(tools))
    if budget < 0:
        raise ValueError("The task brief and tool set exceed the context budget. Start a new task or select fewer tool groups.")
    groups = []
    for m in messages[1:]:
        if m.get("role") == "tool" and groups:
            groups[-1].append(m)
        else:
            groups.append([m])
    kept = []
    for group in reversed(groups):
        size = len(json.dumps(group))
        if size > budget:
            if not kept:
                raise ValueError("The latest tool exchange exceeds the context budget. Narrow the request.")
            break
        kept.insert(0, group)
        budget -= size
    return [messages[0]] + [m for group in kept for m in group] + tail


class CheckpointError(RuntimeError):
    pass


class RepeatedCall(ValueError):
    """The same read, with the same arguments, for the third time running."""


def inference_tools(tools, library=None):
    """One exact tool prefix for execution and every GUI warm-up path.

    With no bridge tools there is nothing to journal, so the internal tools go
    too: a plain chat tab is offered no tools at all, including the one that
    makes tools - there would be nothing to make them from. Tools the model
    made come last, so making one re-prefills the tail of the cached prefix
    rather than the whole tool set.
    """
    if not tools:
        return []
    made = library.model_tools() if library is not None else []
    return list(tools) + list(INTERNAL_TOOLS) + made


class Executor:
    def __init__(self, llm, mcp, tools, schemas=None, record=None, cancel=None,
                 emit=None, checkpoint=None, vision=None, max_chars=100000,
                 library=None, readback=(), review=None, notebook=None):
        self.llm, self.mcp = llm, mcp
        self.bridge_tools = list(tools)
        self.library = library
        self.tools = inference_tools(self.bridge_tools, library)
        self.allowed, self.specs = toolsmith.contracts(tools, schemas)
        self.record = record or TaskRecord()
        self.cancel = cancel or threading.Event()
        self.emit = emit or (lambda kind, payload: None)
        self.checkpoint = checkpoint or (lambda: None)
        self.vision = vision
        self.max_chars = max_chars
        # The app's own answers to "how do I check that landed": see
        # AppSpec.readback and AppSpec.review. Only tools this tab offers count.
        self.readback = [(t, n) for t, n in readback if t in self.allowed]
        self.review = review if review and review[0] in self.allowed else None
        self.must_inspect = self.record.status.startswith("restored")
        self.failed_calls = set()
        self.via = None                   # the made tool a step is running under
        self.auto = False                 # a call the executor made, not the model
        # What this run teaches: the app's notebook (studio_remember writes to
        # it; lessons kept mid-session ride at the request's tail), every
        # validator refusal as (tool, message), and whether anything went wrong
        # - the GUI reflects on a troubled run, never a clean one.
        self.notebook = notebook
        self.refusals = []
        self.trouble = False
        # The question shown to the user by studio_ask; set, the run ends and
        # the answer is the next message.
        self.asked = None

    # ------------------------------------------------ verifying what was written

    def _last_write(self):
        """The most recent journal entry that changed something, or None."""
        for entry in reversed(self.record.journal):
            if not entry.get("read") and entry.get("status") in ("ok", "running", "unknown"):
                return entry
        return None

    def _ids_for(self, names, write):
        """Values for the id arguments `names`, from the last write first and
        then from the most recent call that carried all of them. None when
        nothing in the journal has them - a fresh comp not yet listed."""
        candidates = [write] if write else []
        candidates += [e for e in reversed(self.record.journal) if e is not write]
        for entry in candidates:
            args = entry.get("arguments") or {}
            if all(n in args for n in names):
                return {n: args[n] for n in names}
        return None

    def _readback_hint(self):
        """The reminder to verify, naming the exact read when the app has said
        which one: a call the model can copy, not an instruction to translate."""
        write = self._last_write()
        made = write["name"] if write else "your last edit"
        for tool, names in self.readback:
            ids = self._ids_for(names, write)
            if ids is not None:
                return ("Before finishing, verify the edit you made with %s: call %s with "
                        "%s and compare what it returns with the brief. If it did not land "
                        "as asked, fix it; if you cannot check, say the work is unverified "
                        "and why." % (made, tool, json.dumps(ids)))
        return ("Before finishing, inspect the target changed by %s and compare with the "
                "brief. If unable, state that the work is unverified and explain the "
                "blocker." % made)

    def _auto_review(self, messages):
        """Look at the work before accepting "done": take the app's screenshot
        and hand the vision model's review back as the next message.

        The prompt asks the model to request a preview; a small model forgets,
        and a nag is an instruction it has to translate. This is the
        observation itself. Only with a vision model - without one the picture
        would clear the read-back obligation while nobody had looked - and only
        when the screenshot's ids are in the journal. Returns True when a review
        was appended; the ordinary read-back reminder is the fallback.
        """
        if not (self.review and self.vision) or self.cancel.is_set():
            return False
        tool, names = self.review
        ids = self._ids_for(names, self._last_write())
        if ids is None:
            return False
        self.emit("sys", "Looking at the result before accepting it: %s" % tool)
        self.auto = True
        try:
            text, _, read = self._call({"function": {"name": tool, "arguments": json.dumps(ids)}})
        except CheckpointError:
            raise
        except Exception as e:
            self.emit("sys", "Could not take the screenshot (%s); asking for a read-back instead." % e)
            return False
        finally:
            self.auto = False
        if not read or "Visual review (model assessment)" not in text:
            return False
        write = self._last_write()
        messages.append({"role": "user", "content":
            "Automatic review of the work after %s - I called %s with %s:\n%s\n\n"
            "This is what is on screen now. Fix what the review names, then finish; "
            "if it says the work is as asked, finish with your summary. Do not claim "
            "anything the review did not show."
            % (write["name"] if write else "your edits", tool, json.dumps(ids), text)})
        return True

    def _fresh_lessons(self):
        if self.notebook is None:
            return ""
        fresh = self.notebook.fresh()
        return eng.lessons_section(fresh).strip() if fresh else ""

    def _save(self):
        try:
            self.checkpoint()
        except Exception as e:
            raise CheckpointError("Could not save task progress; execution stopped: " + str(e)) from e

    def _call(self, call):
        """One dispatch, told to the user as it happens: the call as the model
        made it - the whole of it, the script included, since the transcript
        folds it - then its outcome under the same name. Every route in comes
        through here, so a made tool's steps and the executor's own look are
        announced like the model's direct calls."""
        fn = call["function"]
        name, raw = fn["name"], fn.get("arguments") or "{}"
        try:
            shown = json.loads(raw)
        except ValueError:
            shown = raw                   # not JSON; shown as the model wrote it
        self.emit("tool", {"name": name, "arguments": shown, "via": self.via})
        try:
            text, wrote, read = self._dispatch(name, raw)
        except Exception as e:
            self.emit("tool_result", {"name": name, "text": "TOOL ERROR: " + str(e),
                                      "status": "error"})
            raise
        self.emit("tool_result", {"name": name, "text": text,
                                  "status": "error" if text.startswith("TOOL ERROR") else "ok"})
        return text, wrote, read

    def _dispatch(self, name, raw):
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        if name == "studio_task_update":
            validate(args, TASK_TOOL["function"]["parameters"])
            for key, value in args.items():
                if key == "objects":
                    self.record.objects.update(value)
                else:
                    setattr(self.record, key, value)
            return "Task record updated. Checks are reported observations, not independent proof.", False, False
        if name == toolsmith.CREATE_TOOL["function"]["name"]:
            validate(args, toolsmith.CREATE_TOOL["function"]["parameters"])
            return self._make(args), False, False
        if name == ASK_TOOL["function"]["name"]:
            validate(args, ASK_TOOL["function"]["parameters"])
            self.asked = {"question": args["question"], "options": args["options"],
                          "multiple": bool(args.get("multiple"))}
            self.emit("ask", self.asked)
            return ("The question is in front of the user with its choices. Their answer "
                    "will be the next message: finish this reply now, with no further "
                    "tool calls and no work on the question's outcome.", False, False)
        if name == lessons.REMEMBER_TOOL["function"]["name"]:
            validate(args, lessons.REMEMBER_TOOL["function"]["parameters"])
            if self.notebook is None:
                raise ValueError("This tab keeps no notebook; nothing was recorded.")
            lesson, note = self.notebook.add(args["lesson"], "model")
            self.emit("sys", "Remembered: " + lesson["text"])
            return ("Kept for future tasks in this app: %s%s" % (
                lesson["text"], "" if not note else " (" + note + ")"), False, False)
        made = self.library.get(name) if self.library is not None else None
        if made is not None:
            validate(args, made.parameters())
            return self._run_made(made, args)
        if name not in self.allowed:
            raise ValueError("tool is not enabled: " + name)
        spec = self.specs.get(name, {})
        try:
            validate(args, spec.get("inputSchema", self.allowed[name]))
            validate_action(args, spec.get("description", ""))
        except ValueError as e:
            # A refusal is a fact about the contract the model got wrong once;
            # the notebook learns it after the run so it is wrong once only.
            self.refusals.append((name, str(e)))
            raise
        if name == "resolve_control" and str(args.get("action", "")).lower().strip() == "quit":
            raise ValueError("Closing Resolve is prohibited; it may contain unsaved work.")
        signature = json.dumps([name, args], sort_keys=True)
        previous = [e for e in self.record.journal if e.get("signature") == signature]
        if any(e.get("status") in ("running", "unknown") for e in previous):
            raise ValueError("A previous call has an unknown outcome. Inspect the project; do not repeat that call.")
        if signature in self.failed_calls:
            raise ValueError("This exact call already failed. Correct the arguments or explain the blocker.")
        read = readonly(name, args, spec)
        if not read and self.must_inspect:
            raise ValueError("Inspect the current project before editing a restored task.")
        if read:
            # The same read with the same arguments, straight after itself,
            # answers the same. A model that asks again did not take the
            # answer in - a window too small for its own tool results, most
            # often, which the warm-up now fits, but a small model loops on
            # its own too. The second time it gets the first answer back,
            # marked as such; the third ends the run rather than the step
            # limit. A write is never short-circuited: a second generate
            # with the same prompt is another picture.
            again = 0
            for e in reversed(self.record.journal):
                if e.get("signature") == signature and e.get("status") == "ok":
                    again += 1
                else:
                    break
            if again >= 2:
                raise RepeatedCall("%s was called three times in a row with the same "
                                   "arguments, and the answer has not changed." % name)
            if again == 1:
                text = (self.record.journal[-1].get("result") or "") + (
                    "\n\n(The same call as the step before, with the same arguments: "
                    "this is the same answer. Use it, or call something else.)")
                self.record.journal.append({
                    "name": name, "arguments": args, "signature": signature,
                    "read": True, "status": "ok", "result": text, "repeat": True,
                    "verifies": False})
                self._save()
                return text, False, True
        entry = {"name": name, "arguments": args, "signature": signature,
                 "read": read, "status": "running"}
        if self.via:
            entry["via"] = self.via       # a step of a made tool, not a bare call
        if self.auto:
            entry["auto"] = True          # the executor's own look, not the model's
        self.record.journal.append(entry)
        self._save()  # record intent before a call can change the project
        try:
            result = self.mcp.call_tool(name, args)
        except (TimeoutError, ConnectionError, BrokenPipeError, EOFError) as e:
            self.failed_calls.add(signature)
            entry["status"] = "unknown" if not read else "error"
            entry["result"] = str(e)
            raise RuntimeError("Bridge response lost; outcome %s. Inspect before continuing: %s" % (entry["status"], e))
        except Exception as e:
            self.failed_calls.add(signature)
            entry["status"] = "unknown" if not read else "error"
            entry["result"] = str(e)
            raise
        text = eng.mcp_result_to_text(result)
        failed = isinstance(result, dict) and result.get("isError", False)
        structured = result.get("structuredContent") if isinstance(result, dict) else None
        if isinstance(structured, dict) and (structured.get("success") is False or structured.get("error")):
            failed = True
        # Some bridges encode failure in their JSON text rather than isError.
        if isinstance(result, dict):
            for item in result.get("content", []) or []:
                if item.get("type") == "text":
                    try:
                        payload = json.loads(item.get("text", ""))
                        if isinstance(payload, dict) and (payload.get("success") is False or payload.get("error")):
                            failed = True
                    except (ValueError, TypeError):
                        pass
        entry["status"] = "error" if failed else "ok"
        if failed:
            self.failed_calls.add(signature)
        entry["result"] = text
        if isinstance(result, dict):
            # Preserve complete textual evidence on disk; image payloads belong
            # to the preview surface and must not inflate the model context.
            entry["raw_result"] = {k: v for k, v in result.items() if k != "content"}
            entry["raw_result"]["content"] = [i for i in result.get("content", []) or []
                                               if i.get("type") != "image"]
        if failed and not text.startswith("TOOL ERROR"):
            text = "TOOL ERROR: " + text
        # A tool that hands back the thing it made - a generation that waited
        # for its run and returned the picture - has done its own read-back.
        # Asking for another inspection would only be answered with the record.
        content = result.get("content", []) if isinstance(result, dict) else []
        observed = any(i.get("type") == "image" for i in content or [])
        verified_read = not failed and ((read and verification_read(name, args)) or observed)
        entry["verifies"] = verified_read
        if verified_read:
            self.must_inspect = False
        for item in content or []:
            if item.get("type") == "image":
                self.emit("preview", item)
                if self.vision and not self.cancel.is_set():
                    try:
                        critique = self.vision(item, self.record.context())
                        text += "\nVisual review (model assessment): " + critique
                        self.emit("sys", "Visual review: " + critique)
                    except Exception as e:
                        text += "\nVisual review unavailable: " + str(e)
                else:
                    text += "\nPreview available to the user; visual quality has not been assessed by the model."
        return text, not read and not failed, verified_read

    def _make(self, args):
        """Record a new tool. This writes a definition; it changes no project."""
        if self.library is None:
            raise ValueError("This tab cannot make tools.")
        made = toolsmith.parse(args, self.allowed, self.specs, self.library.made)
        note = self.library.add(made)
        # Made tools are appended after the fixed contract, so the model sees
        # this one from its next message on without re-prefilling the rest.
        self.tools = inference_tools(self.bridge_tools, self.library)
        self.emit("sys", "Made a tool: %s (%s)" % (made.name, made.summary()))
        return ("Created %s. It runs %s, and is available from your next message. "
                "It has changed nothing in the project.%s"
                % (made.name, made.summary(), " " + note if note else ""))

    def _run_made(self, made, args):
        """Run a made tool's steps as ordinary bridge calls.

        Every step goes back through _call, so a made tool cannot reach a tool
        this tab was not given, skip validation or the Resolve prohibition, or
        keep its work out of the journal. The read-back obligation is carried
        step by step in order: a tool that edits and then inspects clears it, a
        tool that inspects and then edits does not.
        """
        steps = made.calls(args)
        for tool, _ in steps:
            if tool not in self.allowed:
                raise ValueError("%s uses %s, which this tab no longer offers. "
                                 "Make it again from the tools you have."
                                 % (made.name, tool))
        results, pending, verified, failed = [], False, False, False
        for index, (tool, arguments) in enumerate(steps, 1):
            if self.cancel.is_set():
                results.append({"step": index, "tool": tool,
                                "result": "Cancelled; this step was not executed."})
                failed = True
                break
            self.via = made.name
            try:
                text, wrote, read = self._call(
                    {"function": {"name": tool, "arguments": json.dumps(arguments)}})
            except CheckpointError:
                raise                     # progress is unsaved; stop, do not continue
            except Exception as e:
                results.append({"step": index, "tool": tool, "arguments": arguments,
                                "result": "TOOL ERROR: " + str(e)})
                failed = True
                break
            finally:
                self.via = None
            pending = (pending or wrote) and not read
            verified = read or (verified and not wrote)
            results.append({"step": index, "tool": tool, "arguments": arguments,
                            "result": text})
        report = json.dumps({"tool": made.name, "steps_run": len(results),
                             "steps_total": len(made.steps), "results": results},
                            ensure_ascii=False)
        if failed:
            report = ("TOOL ERROR: %s stopped at step %d of %d%s. "
                      % (made.name, len(results), len(made.steps),
                         "; earlier steps already ran" if len(results) > 1 else "")) + report
        return report, pending, verified and not pending

    def _skipped(self, call, why):
        """A call the model made and the executor will not dispatch - after a
        stop, or three errors in a row - still shows in the transcript as the
        call it was, with why it did not run where its result would be."""
        fn = call["function"]
        try:
            shown = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            shown = fn.get("arguments")
        self.emit("tool", {"name": fn["name"], "arguments": shown, "via": None})
        self.emit("tool_result", {"name": fn["name"], "text": why, "status": "skipped"})

    def run(self, messages, max_steps=25, streaming=True):
        try:
            return self._run(messages, max_steps, streaming)
        except BaseException:
            # A persistence/transport failure may interrupt a multi-call batch.
            # Complete its protocol replies so the next user message is valid.
            pending = set()
            for message in messages:
                if message.get("role") == "assistant":
                    pending.update(c["id"] for c in message.get("tool_calls", []))
                elif message.get("role") == "tool":
                    pending.discard(message.get("tool_call_id"))
            messages.extend({"role": "tool", "tool_call_id": ident,
                             "content": "Execution interrupted. Check the task journal and inspect the project before continuing."}
                            for ident in sorted(pending))
            self.record.status = "interrupted; inspect before continuing"
            try:
                self._save()
            except Exception:
                pass
            raise

    def _run(self, messages, max_steps=25, streaming=True):
        failures, needs_read, reminders, reviews = 0, False, 0, 0
        called, nudged, repeated = 0, False, False
        for entry in self.record.journal:
            if entry.get("verifies") and entry.get("status") == "ok":
                needs_read = False
            elif not entry.get("read") and entry.get("status") in ("ok", "running", "unknown"):
                needs_read = True
        self.record.status = "working"
        self._save()
        def stop(reason):
            self.record.status = reason
            self._save()
            self.emit("sys", reason)
            return reason
        for _ in range(max_steps):
            if self.cancel.is_set():
                return stop("Stopped. Completed edits remain; inspect before resuming.")
            context = context_messages(messages, self.record, self.tools,
                                       self.max_chars, memory=bool(self.tools),
                                       extra=self._fresh_lessons())
            if streaming:
                msg = self.llm.stream(context, self.tools,
                                      lambda piece: self.emit("token", piece))
            else:
                choice = self.llm.chat(context, self.tools)["choices"][0]
                if choice.get("finish_reason") not in (None, "stop", "tool_calls"):
                    return stop("Stopped: incomplete inference response. No tools from that response were executed.")
                msg = choice["message"]
            self.emit("stream_end", None)
            calls = msg.get("tool_calls") or []
            # Validate the whole envelope before dispatching any part of a batch.
            ids = set()
            for call in calls:
                if (not isinstance(call, dict) or not isinstance(call.get("id"), str) or
                        call["id"] in ids or not isinstance(call.get("function"), dict) or
                        not isinstance(call["function"].get("name"), str) or
                        not isinstance(call["function"].get("arguments", ""), str)):
                    return stop("Stopped: the model returned an invalid tool-call envelope. No calls from that response were executed.")
                ids.add(call["id"])
            messages.append(msg)
            if not calls:
                if self.cancel.is_set():
                    return stop("Stopped. No further tools were called.")
                # An unverified edit and a model that says it is done: look at
                # the work ourselves when the app has a picture and something
                # to see it with, else name the read that would verify it.
                if needs_read and reviews < 2 and self._auto_review(messages):
                    reviews += 1
                    needs_read = False
                    self._save()
                    continue
                if needs_read and reminders < 2:
                    messages.append({"role": "user", "content": self._readback_hint()})
                    reminders += 1
                    continue
                final = msg.get("content") or "The model returned an empty reply."
                if needs_read:
                    return stop("Edits were made but remain unverified. " + final)
                if self.tools and not called and announces_work(final):
                    # The model described the call instead of making it. One
                    # reminder it can act on; a second promise ends the run
                    # with its emptiness named rather than looking finished.
                    if not nudged:
                        nudged = True
                        self.emit("sys", "The model described a step without calling a tool; asked once to make the call.")
                        messages.append({"role": "user", "content": PROMISE_HINT})
                        continue
                    return stop("Nothing was done: the model described work but called no tool. " + final)
                self.record.status = "response complete; see recorded checks and limitations"
                self._save()
                return final
            for call in calls:
                if self.cancel.is_set() or failures >= 3 or repeated:
                    out = ("Cancelled before dispatch; this call was not executed."
                           if self.cancel.is_set() else
                           "Not executed: the model is repeating itself." if repeated else
                           "Not executed: stopped after three consecutive tool errors.")
                    self._skipped(call, out)
                else:
                    try:
                        out, wrote, read = self._call(call)
                        called += 1
                        needs_read = (needs_read or wrote) and not read
                    except CheckpointError:
                        raise
                    except RepeatedCall as e:
                        # Not a tool error: nothing for the failure count, and
                        # nothing for the notebook to learn a platitude from.
                        out = "Not executed: " + str(e)
                        repeated = True
                    except Exception as e:
                        out = "TOOL ERROR: " + str(e)
                        if self.record.journal and self.record.journal[-1].get("status") == "unknown":
                            needs_read = True
                            self.must_inspect = True
                    failures = failures + 1 if out.startswith("TOOL ERROR") else 0
                    self.trouble = self.trouble or out.startswith("TOOL ERROR")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": out})
                self._save()
            if repeated:
                return stop("Stopped: the model made the same read three times in a row, with the "
                            "same arguments and the same answer, so it is not taking its results "
                            "in. Try again with a shorter request, or New chat.")
            if failures >= 3:
                return stop("Stopped after repeated tool errors. Review the last error and inspect the project before continuing.")
            if self.asked is not None:
                # The user has a question to answer; the model has nothing to
                # do until they do. Any unverified edit stays in the journal
                # and is picked up when the answer starts the next run.
                self.record.status = "response complete; waiting for the user's answer"
                self._save()
                return self.asked["question"]
        return stop("Stopped at the step limit. Progress is saved; inspect and continue the task when ready.")
