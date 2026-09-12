"""Tools the model writes for itself: fixed sequences of its own bridge tools.

A made tool is *data, never code*. It names tools this tab has already been
given and fills their arguments from its own declared inputs; it cannot name
anything else, and every step is dispatched through the executor's ordinary
path, so schema validation, the action contract, the journal, the read-back
guards and the Resolve prohibition all still apply. There is no eval here and
there must never be one - this runs on the workstation driving the user's live
project.

Made tools are appended after the fixed tool contract, so adding one
invalidates only the tail of LM Studio's cached prefix rather than the whole
tool set. Keep them last.
"""
import json
import os
import re
import tempfile
import time

import studio_agent as eng


NAME_PATTERN = r"^[a-z][a-z0-9_]{2,47}$"
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]{0,31})\}")
SAMPLES = {"string": "sample", "number": 1.5, "integer": 1, "boolean": True}
MAX_STEPS_PER_TOOL = 12

CREATE_TOOL = {"type": "function", "function": {
    "name": "studio_tool_create",
    "description": (
        "Make a new reusable tool for this app out of tools you already have, when the "
        "same sequence of calls keeps coming up. Steps run in order and may only name "
        "tools in this tab's list. In a step's arguments, {input_name} is replaced by "
        "the value of that input when the tool runs; a string that is exactly "
        "\"{input_name}\" keeps the input's type. Every input must be used by some step. "
        "The new tool is available from your next message on. This creates a tool; it "
        "does not run one and does not change the project."),
    "parameters": {"type": "object", "additionalProperties": False,
        "required": ["name", "description", "steps"],
        "properties": {
            "name": {"type": "string", "pattern": NAME_PATTERN,
                     "description": "lower_snake_case, unused by any existing tool"},
            "description": {"type": "string", "minLength": 20, "maxLength": 600,
                            "description": "What it does and when to use it, for your own later use."},
            "inputs": {"type": "array", "items": {"type": "object",
                "required": ["name", "type", "description"], "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "pattern": NAME_PATTERN},
                    "type": {"type": "string", "enum": ["string", "number", "integer", "boolean"]},
                    "description": {"type": "string"}}}},
            "steps": {"type": "array", "minItems": 1, "maxItems": MAX_STEPS_PER_TOOL,
                "items": {"type": "object", "required": ["tool", "arguments"],
                    "additionalProperties": False,
                    "properties": {"tool": {"type": "string"},
                                   "arguments": {"type": "object"}}}},
            "replace": {"type": "boolean",
                        "description": "Required to overwrite a tool you made earlier."}}}}}


def relax(schema):
    """The same schema with value constraints dropped, structure intact.

    Creation checks a step's arguments with sample values standing in for the
    inputs. A sample cannot satisfy an enum or a pattern, so a failure under
    these keywords says nothing about the template; a structural failure - a
    missing required key, an unknown key, the wrong type - always does.
    """
    if not isinstance(schema, dict):
        return schema
    drop = ("enum", "const", "pattern", "format", "minimum", "maximum",
            "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
            "maxLength", "minItems", "maxItems", "uniqueItems", "not", "if",
            "then", "else")
    out = {k: v for k, v in schema.items() if k not in drop}
    for key in ("properties", "$defs", "definitions", "patternProperties"):
        if isinstance(out.get(key), dict):
            out[key] = {k: relax(v) for k, v in out[key].items()}
    for key in ("items", "additionalProperties", "additionalItems"):
        if isinstance(out.get(key), dict):
            out[key] = relax(out[key])
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if isinstance(out.get(key), list):
            out[key] = [relax(v) for v in out[key]]
    return out


def names_used(value, found=None):
    found = set() if found is None else found
    if isinstance(value, str):
        found.update(PLACEHOLDER.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            names_used(item, found)
    elif isinstance(value, list):
        for item in value:
            names_used(item, found)
    return found


def render(value, args):
    """Substitute declared inputs into a step's argument template.

    A string that is exactly one placeholder keeps the input's type - integers
    stay integers, which is the difference between a layer id and a string that
    looks like one. Anything else is textual substitution.
    """
    if isinstance(value, dict):
        return {k: render(v, args) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, args) for v in value]
    if not isinstance(value, str):
        return value
    whole = PLACEHOLDER.fullmatch(value)
    if whole and whole.group(1) in args:
        return args[whole.group(1)]
    def one(match):
        name = match.group(1)
        if name not in args:
            return match.group(0)
        item = args[name]
        return json.dumps(item) if not isinstance(item, str) else item
    return PLACEHOLDER.sub(one, value)


class MadeTool:
    """One tool the model wrote, as stored on disk and as offered back to it."""

    def __init__(self, data):
        self.name = data["name"]
        self.description = data["description"]
        self.inputs = [dict(i) for i in data.get("inputs", [])]
        self.steps = [dict(s) for s in data["steps"]]
        self.created = float(data.get("created") or time.time())

    def parameters(self):
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {i["name"]: {"type": i["type"],
                                             "description": i["description"]}
                                 for i in self.inputs}}
        if self.inputs:                   # an empty `required` upsets some converters
            schema["required"] = [i["name"] for i in self.inputs]
        return schema

    def summary(self):
        return " then ".join(s["tool"] for s in self.steps)

    def tool(self):
        return {"type": "function", "function": {
            "name": self.name,
            "description": (self.description.strip() + "\n\nMade in this tab from "
                            + self.summary() + "."),
            "parameters": eng.sanitize_schema(self.parameters())}}

    def calls(self, args):
        """The bridge calls this tool stands for, with the inputs filled in."""
        return [(s["tool"], render(s["arguments"], args)) for s in self.steps]

    def to_json(self):
        return {"version": 1, "name": self.name, "description": self.description,
                "inputs": self.inputs, "steps": self.steps, "created": self.created}


def contracts(tools, schemas=None):
    """The two views of a tab's tools: what may be called, and how it validates.

    `allowed` is the exposed set keyed by name; `specs` prefers the bridge's
    original schema over the sanitized copy inference sees. One derivation for
    the executor and for reloading a library, so the two cannot drift.
    """
    allowed = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    specs = {t["function"]["name"]: {"name": t["function"]["name"],
             "inputSchema": t["function"]["parameters"],
             "description": t["function"].get("description", "")} for t in tools}
    specs.update({t["name"]: t for t in (schemas or []) if t["name"] in allowed})
    return allowed, specs


def reserved(allowed):
    """Every name a made tool must not take: the bridge's and the internal ones."""
    names = set(allowed)
    names.add(CREATE_TOOL["function"]["name"])
    names.add("studio_task_update")
    return names


def parse(data, allowed, specs, existing=()):
    """Turn a validated studio_tool_create payload into a MadeTool, or explain why not.

    `allowed` is this tab's enabled bridge tools - the only tools a made tool may
    name. Every rejection here is one the model can act on.
    """
    name = data["name"]
    if name in reserved(allowed):
        raise ValueError("%r is already the name of a tool you have. Choose another name." % name)
    if name.startswith("studio_"):
        raise ValueError("names starting with 'studio_' are reserved for this app's own tools")
    if name in existing and not data.get("replace"):
        raise ValueError("you already made a tool called %r; pass replace true to overwrite it" % name)

    inputs = data.get("inputs", [])
    declared = [i["name"] for i in inputs]
    if len(set(declared)) != len(declared):
        raise ValueError("two inputs share a name")

    used = set()
    for index, step in enumerate(data["steps"], 1):
        tool = step["tool"]
        if tool not in allowed:
            raise ValueError("step %d names %r, which is not a tool you have. Steps may only "
                             "use the tools in this tab's list." % (index, tool))
        unknown = names_used(step["arguments"]) - set(declared)
        if unknown:
            raise ValueError("step %d uses %s, which is not one of this tool's inputs"
                             % (index, ", ".join("{%s}" % u for u in sorted(unknown))))
        used |= names_used(step["arguments"])
    missing = [d for d in declared if d not in used]
    if missing:
        raise ValueError("no step uses " + ", ".join("{%s}" % m for m in missing)
                         + "; every input must appear in a step's arguments")

    made = MadeTool(dict(data, created=time.time()))
    check(made, specs, allowed)
    return made


def check(made, specs, allowed):
    """Prove each step's arguments fit its bridge tool before the tool exists.

    Sample inputs stand in for the real ones, so this catches a template that is
    structurally wrong - a missing required argument, a misspelled key, the
    wrong type - and deliberately does not fail on a value a sample cannot
    satisfy. Values are validated for real against the original schema on every
    call, in the executor, like any other tool.
    """
    from studio_tasks import validate, validate_action
    samples = {i["name"]: SAMPLES[i["type"]] for i in made.inputs}
    for index, (step, (tool, args)) in enumerate(zip(made.steps, made.calls(samples)), 1):
        spec = specs.get(tool, {})
        schema = spec.get("inputSchema", allowed[tool])
        try:
            validate(args, relax(schema))
        except ValueError as e:
            raise ValueError("step %d does not fit %s: %s" % (index, tool, e))
        if not names_used(step["arguments"].get("action", "")):
            # A hard-coded action can be checked now; one that arrives as an
            # input is checked on every call, where the real value is known.
            try:
                validate_action(args, spec.get("description", ""))
            except ValueError as e:
                raise ValueError("step %d: %s" % (index, e))


class Library:
    """The tools made for one app, on disk beside its settings and tasks.

    Best-effort in both directions, like the preferences file: a wrecked or
    unwritable file costs one made tool, never the app, and everything read back
    is re-validated before it can be offered to the model again.
    """

    def __init__(self, app_id, directory=None):
        self.app_id = app_id
        self.dir = directory
        self.made = {}
        self.problems = []

    @classmethod
    def for_app(cls, app_id, base=None):
        if base is None:
            base = os.path.dirname(os.path.abspath(
                os.environ.get("STUDIO_SETTINGS") or
                os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"),
                             "StudioAssistant", "settings.json")))
        return cls(app_id, os.path.join(base, "tools", app_id))

    def ordered(self):
        return sorted(self.made.values(), key=lambda m: (m.created, m.name))

    def model_tools(self):
        return [m.tool() for m in self.ordered()]

    def get(self, name):
        return self.made.get(name)

    def load(self, allowed=None, specs=None):
        """Read the app's made tools back. Returns what could not be read."""
        from studio_tasks import validate
        self.made, self.problems = {}, []
        if not self.dir or not os.path.isdir(self.dir):
            return self.problems
        for filename in sorted(os.listdir(self.dir)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.dir, filename)
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("version") != 1:
                    raise ValueError("unsupported made-tool version")
                payload = {k: data[k] for k in ("name", "description", "steps")}
                payload["inputs"] = data.get("inputs", [])
                validate(payload, CREATE_TOOL["function"]["parameters"])
                if payload["name"] + ".json" != filename:
                    raise ValueError("name does not match its file")
                made = MadeTool(dict(data, created=data.get("created")))
                if allowed is not None:
                    # The tab's tool groups may have changed since it was made.
                    check(made, specs or {}, allowed)
                    for step in made.steps:
                        if step["tool"] not in allowed:
                            raise ValueError("uses %s, which this tab no longer offers"
                                             % step["tool"])
                self.made[made.name] = made
            except Exception as e:
                self.problems.append("%s: %s" % (filename, e))
        return self.problems

    def path(self, name):
        return os.path.join(self.dir, name + ".json") if self.dir else None

    def add(self, made):
        """Keep the tool, then try to persist it. Returns a note, or None."""
        self.made[made.name] = made
        path = self.path(made.name)
        if not path:
            return "It lasts for this session only; there is nowhere to save it."
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(made.to_json(), f, indent=1)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except Exception as e:
            return "It lasts for this session only; it could not be saved: %s" % e
        return None

    def remove(self, name):
        self.made.pop(name, None)
        path = self.path(name)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                return False
        return True
