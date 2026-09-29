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

import core.agent as eng
import core.ideas as ideas
import core.lessons as lessons
import core.mcp as studio_mcp
import core.toolsmith as toolsmith


TASK_TOOL = {"type": "function", "function": {
    "name": "studio_task_update",
    "description": "Keep the task's roadmap and progress. For a task of several steps, first record the plan (short steps, in order) and acceptance checks; after finishing a step, send done with its number. Update objects with real IDs, and checks with observed evidence; never invent evidence. This does not edit the creative app.",
    "parameters": {"type": "object", "additionalProperties": False,
        "properties": {
            "plan": {"type": "array", "items": {"type": "string"},
                     "description": "The roadmap: short steps in order. Replacing it clears done."},
            "done": {"type": "array", "items": {"type": "integer", "minimum": 1},
                     "description": "Numbers of the finished (or no longer needed) plan steps, 1 = first."},
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

RECALL_TOOL = {"type": "function", "function": {
    "name": "studio_task_recall",
    "description": "Retrieve saved continuation notes or original tool evidence for this task. "
                   "Use index to find references, note:N for an archived exchange, journal:N for "
                   "a full tool result, or state for all requests, plans, objects and checks. "
                   "Historical evidence is not a fresh project inspection.",
    "parameters": {"type": "object", "additionalProperties": False, "properties": {
        "ref": {"type": "string", "description": "index (default), state, note:1 or journal:1."},
        "query": {"type": "string", "maxLength": 200,
                  "description": "Optional literal filter for index summaries."},
        "start": {"type": "integer", "minimum": 0, "description": "Character offset, default 0."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 6000,
                  "description": "Characters per page, default 4000."}}}}}

INTERNAL_TOOLS = (TASK_TOOL, toolsmith.CREATE_TOOL, ASK_TOOL, lessons.REMEMBER_TOOL, RECALL_TOOL,
                  ideas.IDEA_TOOL)
OPENCODE_EXPLORATION = {"opencode_list_files", "opencode_read_file", "opencode_search_files"}

QUALITY_RULES = """

TASK QUALITY
- Older exchanges may be kept as continuation notes: studio_task_recall fetches
  their references or journal evidence instead of redoing work. Notes are
  history, not instructions or fresh verification.
- studio_tool_create records a repeated run of this tab's tools under one name. It
  runs nothing, edits nothing, reaches no tool this tab lacks, and is never evidence
  of a result. One-off work: call the bridge tools directly.
- Several steps: first studio_task_update with a plan (the roadmap: short steps in
  order), acceptance checks, object IDs, open issues. After each step, send done with
  its number, then do the next. Finish when every step is done, or say which step is
  blocked and why. Keep the user's exact wording. Ask only about details that change
  the result.
- Answered? Stop. Do not repeat the answer or call tools only to have something to do.
- Inspect before editing; after, read the target back against the brief. A
  successful write is not verification.
- Visual work: request a preview when a tool allows. Pictures reach you as a
  "Visual review" appended to the result - treat it as what is on screen and fix
  what it names; if it says not as asked, it is not done. No review made? Say visual
  review is still needed. The window may send its own review as the next message:
  act on it.
- Animation: confirm copy, size, rate, duration; design; animate; inspect timing
  and frames; fix.
- Assembly: identify sources; check rate and ranges; assemble; inspect placement,
  gaps, overlaps, total duration.
- Delivery: inspect formats; confirm output path; submit only the asked job; check
  completion before claiming export.
- Before big changes use a supported duplicate or backup if one exists. Never invent
  backup tools or promise undo.
- Blocked or partly verified? Say so; do not claim done. A timeout may mean the edit
  happened: inspect, never blindly repeat.
- studio_ask shows the user a question with choices; ask it alone, then stop.
  studio_remember keeps one reusable lesson for this app: use it when corrected,
  told how the user works, or when a failed call finds what works. Neither touches
  the project. studio_idea suggests an update to this app itself, only when a
  limit of the app (not the project) stopped or slowed you.
"""

# A reply that announces the next step instead of taking it: "Now I'll generate
# the image" with no tool call attached. A small model does this at the top of a
# task and the run would otherwise end there, looking finished. A question to
# The user is not a promise, and the phrases are first-person future only, so a
# plain answer ("24 fps means...") is left alone.
PROMISE = re.compile(r"\b(?:I(?:'ll| will|'m going to| am going to| shall)|let(?:'s| us| me)|"
                     r"(?:now|next|then|first)[,:]? I)\b", re.I)
PROMISE_HINT = ("You described what you would do, but this reply called no tool, so "
                "nothing happened. Make the call now - the first step, with real "
                "arguments - or, if no tool you have fits the task, say so plainly "
                "instead of describing work.")


# Calls that note something down and change nothing in the project: an answer
# written beside only these is the answer (see Executor._run).
BOOKKEEPING = frozenset((TASK_TOOL["function"]["name"], lessons.REMEMBER_TOOL["function"]["name"],
                         toolsmith.CREATE_TOOL["function"]["name"], ideas.IDEA_TOOL["function"]["name"]))

ROADMAP_NUDGES = 2
ROADMAP_HINT = ("Your roadmap still has step %(n)d open: \"%(step)s\". Do it now. If it is "
                "already done or no longer needed, mark it with studio_task_update (done) "
                "and go on; if it is blocked, say which step and why.")


STEP_MARK = re.compile(r"(?:^|(?<=\s))(\d{1,2})[.)]\s+")


def numbered_steps(text):
    """The steps of a brief written as a numbered list - "1) this; 2) that" or
    one per line - in order, else []. Numbers must run 1, 2, 3...; anything
    else (a version, "3. place") is not a list."""
    marks = list(STEP_MARK.finditer(text or ""))
    want, starts = 1, []
    for m in marks:
        if int(m.group(1)) == want:
            starts.append(m)
            want += 1
    if len(starts) < 2:
        return []
    steps = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        step = " ".join(text[m.end():end].split()).rstrip(" ;,")
        if not step:
            return []
        steps.append(step[:200])
    return steps[:12]


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
        self.done = []
        self.objects = {}
        self.checks = []
        self.issues = []
        self.journal = []
        self.notes = []
        self.compacted_until = 1       # first history message not yet archived
        self.status = "ready"

    def next_step(self):
        """The number of the first plan step not done, else None."""
        for i in range(1, len(self.plan) + 1):
            if i not in self.done:
                return i
        return None

    def roadmap(self):
        """The plan as a checklist, ending with what to do next; "" without one."""
        if not self.plan:
            return ""
        nxt = self.next_step()
        lines = ["%s %d. %s" % ("[x]" if i in self.done else "[>]" if i == nxt else "[ ]", i, step)
                 for i, step in enumerate(self.plan, 1)]
        lines.append("Next: step %d." % nxt if nxt else
                     "Every step is done: check the work, then answer.")
        return "Roadmap:\n" + "\n".join(lines)

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
        validate({k: getattr(record, k) for k in ("plan", "done", "objects", "checks", "issues")},
                 TASK_TOOL["function"]["parameters"])
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise ValueError("invalid saved messages")
        if not all(isinstance(e, dict) for e in record.journal):
            raise ValueError("invalid saved journal")
        if (record.compacted_until < 1 or record.compacted_until > max(1, len(messages))
                or any(not isinstance(n, dict) or not isinstance(n.get("summary"), str)
                       or not isinstance(n.get("messages"), list) for n in record.notes)):
            raise ValueError("invalid saved continuation notes")
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
    if memory and any(t.get("function", {}).get("name") == "studio_task_recall" for t in tools):
        return continuation_context(messages, record, tools, max_chars, extra)
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


def _excerpt(text, limit):
    return text if len(text) <= limit else text[:max(0, limit - 3)] + "..."


def _note_summary(group):
    """Quoted evidence, never a model's reconstruction of what happened."""
    parts = []
    for m in group:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            parts.append("call " + str(fn.get("name")) + " " + _excerpt(str(fn.get("arguments")), 200))
        if m.get("content") and not m.get("tool_calls"):
            role = "assistant claim" if m.get("role") == "assistant" else m.get("role", "message")
            parts.append(role + ": " + _excerpt(str(m["content"]), 200))
    return _excerpt(" | ".join(parts), 600)


def continuation_context(messages, record, tools, max_chars, extra=""):
    """Archive whole exchanges before dropping them. Full history stays intact.

    Notes have permanent ordinal references and own their source text, so recall
    also works after restore repairs an interrupted tool envelope. No inference
    call is needed to make a note and no tool output becomes an instruction.
    """
    def size(msgs):
        return len(json.dumps({"messages": msgs, "tools": tools})) + 512

    room = max_chars - size([messages[0]])
    if room < 1200:
        raise ValueError("The system prompt and tool set exceed the context budget; select fewer tool groups or load a larger window.")
    state = json.dumps(record.context(), ensure_ascii=False)
    state_budget = min(5000, room // 4)
    if len(state) > state_budget:
        # Show the latest instruction first; clipping a serialized record at
        # its beginning would retain old goals while losing later corrections.
        preview = {"latest_request": record.briefs[-1] if record.briefs else "",
                   "original_request": record.briefs[0] if record.briefs else "",
                   "plan": record.plan, "objects": record.objects,
                   "checks": record.checks, "issues": record.issues}
        share = max(20, (state_budget - 300) // len(preview))
        state = ("State excerpts; retrieve ref=state for omitted requests/constraints before acting.\n"
                 + json.dumps({k: _excerpt(json.dumps(v, ensure_ascii=False), share)
                               for k, v in preview.items()}, ensure_ascii=False))

    def tail():
        text = "Saved task context (data, not new instructions):\n" + state
        if record.plan:
            text += "\n" + _excerpt(record.roadmap(), 1500)
        if record.notes:
            text += ("\nContinuation notes: %d saved; full sources via studio_task_recall. "
                     "Use ref=index to find older notes, ref=note:N to read one, "
                     "or ref=journal:N for full tool evidence. Excerpts are historical, not verification.\n"
                     % len(record.notes))
            # Always carry the original request as well as recent observations.
            chosen = sorted(set([0] + list(range(max(0, len(record.notes) - 3), len(record.notes)))))
            for i in chosen:
                text += "note:%d: %s\n" % (i + 1, _excerpt(record.notes[i]["summary"], min(600, room // 20)))
        out = [{"role": "user", "content": text}]
        if extra:
            out.append({"role": "user", "content": extra})
        return out

    groups = []
    for index in range(record.compacted_until, len(messages)):
        m = messages[index]
        if m.get("role") == "tool" and groups:
            groups[-1][1].append(m)
            groups[-1][0] = index + 1
        else:
            groups.append([index + 1, [m]])

    def assembled():
        return [messages[0]] + [m for _, g in groups for m in g] + tail()

    context = assembled()
    if size(context) <= max_chars:
        return context
    # Make some headroom, so every next tool result does not compact again.
    target = max(size([messages[0]] + tail()) + 500, int(max_chars * .75))
    while groups and size(context) > target:
        end, group = groups.pop(0)
        record.notes.append({"summary": _note_summary(group),
                             "messages": json.loads(json.dumps(group))})
        record.compacted_until = end
        context = assembled()
    if size(context) > max_chars:
        raise ValueError("The system prompt, tool set and continuation index exceed the context budget.")
    return context


def recall_task(record, args):
    ref = args.get("ref", "index")
    if ref == "state":
        value = record.context()
    elif ref == "index":
        rows = ["note:%d: %s" % (i + 1, n["summary"]) for i, n in enumerate(record.notes)]
        rows += ["journal:%d: %s %s %s" % (i + 1, e.get("status"), e.get("name"),
                  _excerpt(json.dumps(e.get("arguments", {}), ensure_ascii=False), 300))
                 for i, e in enumerate(record.journal)]
        query = args.get("query", "").casefold()
        value = "\n".join(r for r in rows if query in r.casefold()) or "No matching saved references."
    else:
        match = re.fullmatch(r"(note|journal):([1-9][0-9]*)", ref)
        if not match:
            raise ValueError("Use index, state, note:N or journal:N.")
        source = record.notes if match[1] == "note" else record.journal
        index = int(match[2]) - 1
        if index >= len(source):
            raise ValueError("No saved reference " + ref)
        value = source[index]
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    start, limit = args.get("start", 0), args.get("limit", 4000)
    page = text[start:start + limit]
    next_page = (" Next start=%d." % (start + len(page)) if start + len(page) < len(text) else " End.")
    return "Saved historical data %s.%s\n%s" % (ref, next_page, page)


class CheckpointError(RuntimeError):
    pass


class RepeatedCall(ValueError):
    """The same read, with the same arguments, for the third time running."""


class Cancelled(Exception):
    """Stop pressed while a reply was streaming. Raised from the token
    callback so it unwinds through the open response and closes it - the host
    stops generating - instead of the rest of the reply arriving after Stop."""


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
    return offered_tools(tools) + list(INTERNAL_TOOLS) + made


# Past this much schema JSON a tab's bridge tools are offered by reference: a
# one-line index and two meta tools, with the full schema fetched on demand
# into the history. The After Effects bridge is ~100k chars (~29k tokens) of
# schema that would otherwise sit in every request's prefix and window.
LAZY_CHARS = 16000
SCHEMA_TOOL_NAME = "studio_tool_schema"
CALL_TOOL_NAME = "studio_tool_call"


def lazy(tools):
    return len(json.dumps(tools or [])) > LAZY_CHARS


def _gist(description):
    """A description's first sentence, capped: enough to pick a tool by."""
    text = " ".join((description or "").split())
    cut = re.search(r"[.!?](\s|$)", text)
    text = text[:cut.start() + 1] if cut else text
    return text if len(text) <= 110 else text[:107].rstrip() + "..."


def offered_tools(tools):
    """The bridge tools as inference sees them: whole, or by reference.

    Deterministic from the tool list, so the cached prefix stays one prefix."""
    tools = list(tools)
    if not lazy(tools):
        return tools
    index = "\n".join("- %s: %s" % (t["function"]["name"], _gist(t["function"].get("description")))
                      for t in tools)
    return [
        {"type": "function", "function": {
            "name": SCHEMA_TOOL_NAME,
            "description": ("Fetch the full description and parameter schema of this app's "
                            "tools, by name. Do this before a tool's first use in a "
                            "conversation; the schema stays in the history after that. "
                            "Reads nothing from the app."),
            "parameters": {"type": "object", "additionalProperties": False,
                           "properties": {"names": {"type": "array", "items": {"type": "string"}}},
                           "required": ["names"]}}},
        {"type": "function", "function": {
            "name": CALL_TOOL_NAME,
            "description": ("Call one of this app's tools by name, with arguments matching its "
                            "schema (get it with %s first). The tools:\n%s" % (SCHEMA_TOOL_NAME, index)),
            "parameters": {"type": "object", "additionalProperties": False,
                           "properties": {"name": {"type": "string"},
                                          "arguments": {"type": "object"}},
                           "required": ["name", "arguments"]}}},
    ]


class Executor:
    def __init__(self, llm, mcp, tools, schemas=None, record=None, cancel=None,
                 emit=None, checkpoint=None, vision=None, max_chars=100000,
                 library=None, readback=(), review=None, notebook=None, tab=""):
        self.llm, self.mcp = llm, mcp
        self.tab = tab                    # the app's name, on the ideas studio_idea adds
        self.ideas_added = 0
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
        self._context_checked = False
        self._output_tokens = None
        self._journal_start = len(self.record.journal)
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

    def _settle(self):
        """A render may have sent the host's models away (`eng.YieldGPU`);
        they come back here, when the model is next needed - after the user
        has the picture and the vision model has looked at it. Asked before
        that, the host would load the model just in time, at its default
        window."""
        eng.settle(self.mcp)

    def _fit_context_budget(self):
        """Budget against the loaded window after GPU settlement, once per run.

        Characters are an estimate, not a tokenizer. Use 2.5 per input token
        (below the measured ~3.5) and reserve output space on the request too.
        Fakes and third-party clients retain the caller's explicit budget.
        """
        if self._context_checked or not isinstance(self.llm, eng.LLM):
            return
        self._context_checked = True
        loaded, _ = eng.context_window(self.llm.base_url, self.llm.model)
        if isinstance(loaded, int) and loaded > 0:
            self._output_tokens = min(4096, max(256, loaded // 4))
            self.max_chars = min(self.max_chars, int((loaded - self._output_tokens) * 2.5))
        else:
            self._output_tokens = 4096
            self.max_chars = min(self.max_chars, 32000)

    def _handoff_due(self):
        if "opencode_ask" not in self.allowed:
            return False
        recent = self.record.journal[self._journal_start:]
        if any(e.get("status") == "ok" and e.get("name") in
               ("opencode_ask", "opencode_wait", "opencode_get_session")
               for e in recent):
            return False
        return sum(e.get("name") in OPENCODE_EXPLORATION for e in recent) >= 3

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
        if name == CALL_TOOL_NAME:
            # By-reference call: shown, journalled and validated as the tool
            # it names, exactly as a direct call would be.
            outer = json.loads(raw)
            if not isinstance(outer, dict) or not isinstance(outer.get("name"), str):
                raise ValueError("%s needs a tool `name` and its `arguments`" % CALL_TOOL_NAME)
            name, raw = outer["name"], json.dumps(outer.get("arguments") or {})
        try:
            shown = json.loads(raw)
        except ValueError:
            shown = raw                   # not JSON; shown as the model wrote it
        self.emit("tool", {"name": name, "arguments": shown, "via": self.via})
        try:
            try:
                text, wrote, read = self._dispatch(name, raw)
            except ValueError as e:
                # A deferred tool called wrong: hand back its schema with the
                # refusal, so the correction costs one step, not two.
                if lazy(self.bridge_tools) and name in self.allowed:
                    raise ValueError("%s\n%s" % (e, self._schemas([name])))
                raise
        except Exception as e:
            self.emit("tool_result", {"name": name, "text": "TOOL ERROR: " + str(e),
                                      "status": "error"})
            raise
        self.emit("tool_result", {"name": name, "text": text,
                                  "status": "error" if text.startswith("TOOL ERROR") else "ok"})
        return text, wrote, read

    def _schemas(self, names):
        """Full schemas for deferred tools, as the model would have seen them."""
        by_name = {t["function"]["name"]: t["function"] for t in self.bridge_tools}
        out = []
        for n in names:
            fn = by_name.get(n)
            if fn is None:
                out.append("%s: no such tool in this tab." % n)
            else:
                out.append("%s: %s\nparameters: %s" % (n, fn.get("description", ""),
                                                      json.dumps(fn["parameters"], separators=(",", ":"))))
        return "\n\n".join(out)

    def _dispatch(self, name, raw):
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        if name == SCHEMA_TOOL_NAME and lazy(self.bridge_tools):
            names = args.get("names")
            if not isinstance(names, list) or not names:
                raise ValueError("names must be a non-empty list of tool names")
            return self._schemas(names), False, False
        if name == "studio_task_recall":
            validate(args, RECALL_TOOL["function"]["parameters"])
            return recall_task(self.record, args), False, False
        if name == "studio_task_update":
            validate(args, TASK_TOOL["function"]["parameters"])
            old_plan = list(self.record.plan)
            if "plan" in args and args["plan"] != old_plan:
                self.record.done = []         # a new roadmap starts unticked
            for key, value in args.items():
                if key == "objects":
                    self.record.objects.update(value)
                elif key != "done":
                    setattr(self.record, key, value)
            if "done" in args:
                self.record.done = sorted(set(self.record.done) |
                                          {n for n in args["done"] if n <= len(self.record.plan)})
            reply = "Task record updated. Checks are reported observations, not independent proof."
            if "plan" in args or "done" in args:
                self.emit("roadmap", {"plan": list(self.record.plan), "done": list(self.record.done)})
                reply += "\n" + self.record.roadmap()
            return reply, False, False
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
            scope = args.get("scope")
            if isinstance(self.notebook, lessons.Stack):
                lesson, note = self.notebook.add(args["lesson"], "model", scope)
            else:
                lesson, note = self.notebook.add(args["lesson"], "model")
            self.emit("sys", "Remembered: " + lesson["text"])
            return ("Kept for future tasks in this app: %s%s" % (
                lesson["text"], "" if not note else " (" + note + ")"), False, False)
        if name == ideas.IDEA_TOOL["function"]["name"]:
            validate(args, ideas.IDEA_TOOL["function"]["parameters"])
            if self.ideas_added >= ideas.MODEL_IDEAS_PER_RUN:
                raise ValueError("Enough ideas from one task; nothing added. Finish the task.")
            reply = ideas.Ideas().suggest(args["idea"], self.tab)
            if reply.startswith("Added"):
                self.ideas_added += 1
                self.emit("sys", "Idea for Studio Assist added: " + " ".join(args["idea"].split()))
            return reply, False, False
        made = self.library.get(name) if self.library is not None else None
        if made is not None:
            validate(args, made.parameters())
            return self._run_made(made, args)
        if name not in self.allowed:
            raise ValueError("tool is not enabled: " + name)
        if name in OPENCODE_EXPLORATION and self._handoff_due():
            raise ValueError("The three-read exploration budget is used. Call opencode_ask with "
                             "the user's request and constraints, or answer/ask a focused question. "
                             "For a review, explicitly ask OpenCode to inspect without edits.")
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
            result = self.mcp.call_tool(name, args, cancel=self.cancel)
        except eng.Cancelled as e:
            # The Stop the model's own loop is about to notice anyway
            # (`self.cancel.is_set()`, checked between calls); this is the
            # same stop, just caught early enough to end a call that would
            # otherwise block for minutes. Not a tool error.
            entry["status"] = "unknown" if not read else "error"
            entry["result"] = str(e)
            raise
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
        text += "\nSaved evidence: journal:%d (studio_task_recall)." % len(self.record.journal)
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
            result = self._run(messages, max_steps, streaming)
            # A run can end on a render (a Stop, a question) with the model
            # still away; the reflection and the next turn talk to it.
            self._settle()
            return result
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
        roadmap_nudges = 0
        handoff_reminded = False
        context_retried = False
        self._journal_start = len(self.record.journal)
        self._context_checked = False
        for entry in self.record.journal:
            if entry.get("verifies") and entry.get("status") == "ok":
                needs_read = False
            elif not entry.get("read") and entry.get("status") in ("ok", "running", "unknown"):
                needs_read = True
        self.record.status = "working"
        # A brief written as numbered steps is the roadmap: seen live, the
        # model worked such a list without ever recording one, so nothing
        # held it to the steps. Only when no roadmap is still open.
        steps = numbered_steps(self.record.briefs[-1] if self.record.briefs else "")
        if steps and self.record.next_step() is None:
            self.record.plan, self.record.done = steps, []
            self.emit("roadmap", {"plan": list(steps), "done": []})
        self._save()
        def stop(reason):
            self.record.status = reason
            self._save()
            self.emit("sys", reason)
            return reason
        def stopped(prefix):
            if not any(not e.get("read") and e.get("status") != "skipped"
                       for e in self.record.journal):
                return prefix + " No edit-capable tools were run."
            return prefix + " Earlier operations may have changed the project; inspect before resuming."
        for _ in range(max_steps):
            if self.cancel.is_set():
                return stop(stopped("Stopped."))
            # This tab delegates coding. A small model can instead spend the
            # whole run reading file beginnings. Pause exploration after three
            # reads; answering or delegating a read-only review remain available.
            if self._handoff_due() and not handoff_reminded:
                handoff_reminded = True
                messages.append({"role": "user", "content":
                    "You have inspected several files without handing work to OpenCode. "
                    "If the user requested a code change, call opencode_ask now with their "
                    "exact request and constraints; ask OpenCode to read AGENTS.md, locate "
                    "the implementation and test it. You do not need to identify every "
                    "function first. Workspace exploration tools are now paused until the "
                    "handoff. If the user only requested information or review, answer from "
                    "the evidence, ask a focused question, or delegate inspection explicitly "
                    "without edits. Do not turn a review into an edit request."})
            self._settle()
            self._fit_context_budget()
            active_tools = [t for t in self.tools if not (self._handoff_due() and
                            t["function"]["name"] in OPENCODE_EXPLORATION)]
            note_count = len(self.record.notes)
            context = context_messages(messages, self.record, active_tools,
                                       self.max_chars, memory=bool(self.tools),
                                       extra=self._fresh_lessons())
            if len(self.record.notes) > note_count:
                self._save()
                self.emit("sys", "Saved %d continuation notes; earlier evidence is available through task recall."
                          % (len(self.record.notes) - note_count))
            limits = {"max_tokens": self._output_tokens} if self._output_tokens else {}
            try:
                if streaming:
                    def on_text(piece):
                        if self.cancel.is_set():
                            raise Cancelled()
                        self.emit("token", piece)
                    msg = self.llm.stream(context, active_tools, on_text, **limits)
                else:
                    choice = self.llm.chat(context, active_tools, **limits)["choices"][0]
                    if choice.get("finish_reason") == "length":
                        raise eng.ContextLimitError("The model's reply reached its length limit.")
                    if choice.get("finish_reason") not in (None, "stop", "tool_calls"):
                        return stop("Stopped: incomplete inference response. No tools from that response were executed.")
                    msg = choice["message"]
            except Cancelled:
                self.emit("stream_end", None)
                return stop(stopped("Stopped mid-reply."))
            except eng.ContextLimitError:
                self.emit("stream_end", None)
                if context_retried:
                    return stop("The model still could not finish its response. Task evidence is saved; "
                                "increase the model's response/context limit before continuing. "
                                "No tools from either incomplete reply were executed.")
                context_retried = True
                self.max_chars = int(self.max_chars * .7)
                self.emit("sys", "The model reached its response/context limit. Reducing context and "
                          "retrying once from saved evidence; no partial tool calls were executed.")
                continue
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
                step = self.record.next_step()
                if (step and roadmap_nudges < ROADMAP_NUDGES and self.asked is None
                        and not final.rstrip().endswith("?")):
                    # Stopped with roadmap steps open: name the next one. Twice
                    # at most - a model that stops a third time has a reason,
                    # and its reply says it.
                    roadmap_nudges += 1
                    self.emit("sys", "Roadmap step %d is still open; asked the model to "
                                     "carry on." % step)
                    messages.append({"role": "user", "content": ROADMAP_HINT % {
                        "n": step, "step": self.record.plan[step - 1]}})
                    continue
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
                    except eng.Cancelled as e:
                        # Not a tool error: the bridge call was interrupted by
                        # The user's own Stop, already reflected in self.cancel.
                        out = "Cancelled: " + str(e)
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
            # An answer written alongside nothing but a note to the task
            # record is the answer: asked for another step, a small model
            # writes the same reply again, once or twice, and the user reads
            # it repeated. A reply that promises more work still continues.
            # The same holds beside a lesson or a made tool: seen live, a
            # finished model called one after another, answer rewritten each
            # time, until the user stopped it. Open roadmap steps continue.
            final = (msg.get("content") or "").strip()
            if (final and not needs_read and not failures and not self.cancel.is_set()
                    and not announces_work(final) and self.record.next_step() is None
                    and all(c["function"]["name"] in BOOKKEEPING for c in calls)):
                self.record.status = "response complete; see recorded checks and limitations"
                self._save()
                return final
            if self.asked is not None:
                # The user has a question to answer; the model has nothing to
                # do until they do. Any unverified edit stays in the journal
                # and is picked up when the answer starts the next run.
                self.record.status = "response complete; waiting for the user's answer"
                self._save()
                return self.asked["question"]
        return stop("Stopped at the step limit. Progress is saved; inspect and continue the task when ready.")
