"""The console-mode CLI: elicitation at a terminal, one agent turn, the
interactive REPL, and the argparse entry point. Split out of
core/agent.py (docs/CODEMAP.md); nothing calls back into this from the
rest of the engine, so core/agent.py keeps its own
`if __name__ == "__main__":` guard and only imports `main` from here."""
import argparse
import json
import os
import re
import sys

import core.procs as studio_procs

from core.agent import (
    log, DEFAULT_HOST, DEFAULT_MODEL, LLM, YieldGPU, to_openai_tools,
    probe_models, resolve_vision, resolve_draft, loaded_instances,
    make_room, give_back, fit_model, estimate_tokens,
    read_studio_brief, studio_brief_path, about_section, research_client, Router,
    DEFAULT_APP, TABS, TABS_BY_ID, get_app, split_command, add_bridge, BridgeSpec,
)

def ask_at_terminal(asked, answer=input):
    """A studio_ask question at the console: numbered choices, a number or
    numbers (or free text) back. Returns the reply as the user's next message."""
    print("\n" + asked["question"])
    for n, option in enumerate(asked["options"], 1):
        desc = option.get("description") or ""
        print("  %d. %s%s" % (n, option["label"], "  - " + desc if desc else ""))
    print("  (a number%s, or type something else)"
          % (", or several separated by commas" if asked.get("multiple") else ""))
    try:
        reply = answer("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return answer_text(asked, reply)


def elicit_fields(schema):
    """[(name, spec, choices)] of an elicitation's requestedSchema, in order.
    `choices` is [(value, label)] for an enum field, else None."""
    out = []
    for name, spec in ((schema or {}).get("properties") or {}).items():
        choices = None
        if isinstance(spec.get("oneOf"), list):
            choices = [(o.get("const"), o.get("title") or str(o.get("const")))
                       for o in spec["oneOf"] if isinstance(o, dict) and "const" in o]
        elif isinstance(spec.get("enum"), list):
            names = spec.get("enumNames") or []
            choices = [(v, names[i] if i < len(names) else str(v))
                       for i, v in enumerate(spec["enum"])]
        out.append((name, spec, choices))
    return out


def elicit_at_terminal(params, answer=input):
    """A bridge asking the user something (MCP elicitation), at the console:
    the message, any diff or command it carried, then each field. Enter on a
    choice leaves it unanswered; "c" cancels the whole request."""
    shown = ((params.get("_meta") or {}).get("studio/approval") or {})
    print("\n" + (params.get("message") or "The bridge asks:"))
    if shown.get("diff"):
        print(shown["diff"].rstrip())
    schema = params.get("requestedSchema") or {}
    required = set(schema.get("required") or [])
    content = {}
    try:
        for name, spec, choices in elicit_fields(schema):
            title = spec.get("title") or name
            if choices:
                for n, (_, label) in enumerate(choices, 1):
                    print("  %d. %s" % (n, label))
                reply = answer("%s (number, or c to cancel)> " % title).strip()
                if reply.lower() == "c":
                    return {"action": "cancel"}
                if reply.isdigit() and 1 <= int(reply) <= len(choices):
                    content[name] = choices[int(reply) - 1][0]
                elif name in required:
                    return {"action": "decline"}
            elif spec.get("type") == "boolean":
                reply = answer("%s (y/n)> " % title).strip().lower()
                content[name] = reply.startswith("y")
            else:
                reply = answer("%s> " % title).strip()
                if reply:
                    content[name] = reply
    except (EOFError, KeyboardInterrupt):
        print()
        return {"action": "cancel"}
    return {"action": "accept", "content": content}


def answer_text(asked, reply):
    """What the user's pick becomes in the conversation: the labels chosen, in
    The user's words, or their own text when it was not a pick."""
    labels = [o["label"] for o in asked["options"]]
    picks = []
    for piece in re.split(r"[,\s]+", reply.strip()):
        if piece.isdigit() and 1 <= int(piece) <= len(labels):
            picks.append(labels[int(piece) - 1])
        elif piece:
            picks = []
            break
    if picks and (asked.get("multiple") or len(picks) == 1):
        return "; ".join(picks)
    return reply.strip()


def run_agent(llm, mcp, tools, task, system_prompt, max_steps=25, quiet=False,
              schemas=None, library=None, vision=None, readback=(), review=None,
              notebook=None, app_name="", answer=input, makes_pictures=False):
    """One task, start to finish. `system_prompt` is final - see AppSpec.cli_prompt.

    A studio_ask question is put to the console and its answer continues the
    same task; a troubled run ends with the notebook learning from it, as the
    GUI's does. `makes_pictures` is the app's (AppSpec.makes_pictures): a
    picture it returns is checked against the brief, not reviewed for flaws.
    """
    from core.tasks import Executor, TaskRecord
    import core.lessons as studio_lessons
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": task}]
    record = TaskRecord()
    record.briefs.append(task)
    def emit(kind, payload):
        if kind == "tool":
            log("  %s %s" % (payload["name"], json.dumps(payload["arguments"])[:160]), quiet)
        elif kind == "tool_result":
            log("     " + " ".join(payload["text"].split())[:180], quiet)
        elif kind == "sys":
            log("  " + str(payload), quiet)
    look = None
    if vision:
        look = vision.check if makes_pictures else vision.review
    while True:
        executor = Executor(llm, mcp, tools, schemas=schemas, record=record, emit=emit,
                            library=library, vision=look,
                            readback=readback, review=review, notebook=notebook)
        result = executor.run(messages, max_steps, streaming=False)
        note = getattr(llm, "draft_note", None)
        if note:
            llm.draft_note = None
            log("  " + note, quiet)
        if notebook is not None:
            for lesson in learn_from_run(executor, messages, notebook, llm, app_name):
                log("  lesson kept: " + lesson, quiet)
        if executor.asked is None:
            return result
        reply = ask_at_terminal(executor.asked, answer)
        if not reply:
            return result
        messages.append({"role": "user", "content": reply})
        record.briefs.append(reply)


def learn_from_run(executor, messages, notebook, llm, app_name):
    """What one run leaves in the notebook: every validator refusal, and - when
    the run had trouble or the brief was a correction - one reflected lesson.
    Returns the texts kept. Never raises: a lesson is worth nothing if it costs
    the task's result."""
    import core.lessons as studio_lessons
    kept = []
    try:
        for lesson in notebook.learn_refusals(executor.refusals):
            kept.append(lesson["text"])
        brief = executor.record.briefs[-1] if executor.record.briefs else ""
        stated = studio_lessons.explicit_lesson(brief)
        if stated:
            lesson, note = notebook.add(stated, "user")
            if note != "already kept":
                kept.append(lesson["text"])
        if executor.trouble or studio_lessons.looks_like_correction(brief):
            text = studio_lessons.reflect(llm, app_name, messages)
            if text:
                lesson, note = notebook.add(text, "review")
                if note != "already kept":
                    kept.append(lesson["text"])
    except Exception as e:
        log("  could not learn from this run: %s" % e, True)
    return kept


def converse(llm, mcp, tools, app, args, schemas=None):
    """The task given on the command line, or an interactive session if none.

    The CLI shares the GUI's library of made tools: same app, same directory
    beside the settings file, so a tool made in a tab is offered here too.
    """
    import core.toolsmith as toolsmith
    import core.lessons as studio_lessons
    notebook = studio_lessons.for_app(app)
    problem = notebook.load()
    if problem:
        log("  could not read this app's lessons - " + problem, args.quiet)
    elif notebook.lessons:
        log("  %d lesson(s) from earlier work" % len(notebook.lessons), args.quiet)
    system_prompt = app.cli_prompt(read_studio_brief(), notebook.brief())
    if app.drivable:
        import core.appinfo as studio_appinfo
        base = os.path.dirname(studio_brief_path())
        system_prompt += about_section(studio_appinfo.render(studio_appinfo.refresh(app, base)))
    library = None
    if tools:
        library = toolsmith.Library.for_app(app.id)
        allowed, specs = toolsmith.contracts(tools, schemas)
        for problem in library.load(allowed, specs):
            log("  not offering a made tool - " + problem, args.quiet)
    import core.tasks as tasks
    # The window the model is loaded with has to hold this briefing, these
    # tools and a conversation; LM Studio's default does not. The GUI fits it
    # after its warm-up, exactly; the CLI has no warm-up, so from an estimate.
    # A load goes onto an empty card, as the GUI's does (fit_model's `keep`).
    if getattr(llm, "base_url", None):
        _, note = fit_model(llm.base_url, llm.model,
                            estimate_tokens(system_prompt, tasks.inference_tools(tools, library)),
                            exact=False, keep=())
        if note:
            log("  " + note, args.quiet)
    if args.task:
        print(run_agent(llm, mcp, tools, " ".join(args.task), system_prompt,
                        args.max_steps, args.quiet, schemas=schemas, library=library,
                        vision=getattr(args, "vision", None),
                        readback=app.readback, review=app.review,
                        notebook=notebook, app_name=app.name,
                        makes_pictures=app.makes_pictures))
        return 0

    print("studio_agent [%s] - interactive. Ctrl-C or 'exit' to quit.\n" % app.name)
    prompt = "%s> " % app.id
    while True:
        try:
            task = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if task.lower() in ("exit", "quit"):
            return 0
        if not task:
            continue
        try:
            print("\n" + run_agent(llm, mcp, tools, task, system_prompt,
                                   args.max_steps, args.quiet, schemas=schemas,
                                   library=library, vision=getattr(args, "vision", None),
                                   readback=app.readback, review=app.review,
                                   notebook=notebook, app_name=app.name,
                                   makes_pictures=app.makes_pictures)
                  + "\n")
        except Exception as e:
            print("error: %s\n" % e, file=sys.stderr)


def env_default(*names, fallback=None):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return fallback


def _interrupt():
    raise KeyboardInterrupt


def main():
    p = argparse.ArgumentParser(
        description="Local LLM agent for creative apps. Inference on the tailnet, "
                    "tools on this PC.")
    p.add_argument("task", nargs="*", help="what to do; omit for an interactive session")
    p.add_argument("--app", default=DEFAULT_APP, choices=sorted(TABS_BY_ID),
                   help="which app to drive, or 'chat' for no app at all "
                        "(default: %(default)s)")
    p.add_argument("--host", default=env_default("STUDIO_HOST", "AE_AGENT_HOST",
                                                 fallback=DEFAULT_HOST),
                   help="OpenAI-compatible base URL (default: %(default)s)")
    p.add_argument("--model", default=env_default("STUDIO_MODEL", "AE_AGENT_MODEL"),
                   help="model id on the host (default: the app's preferred small model "
                        "if served, else %s)" % DEFAULT_MODEL)
    p.add_argument("--draft", default=None, metavar="MODEL",
                   help="draft model for speculative decoding, or 'off' (default: "
                        "STUDIO_DRAFT_MODEL, else a small model of the same family the "
                        "host serves, else none)")
    p.add_argument("--groups", default=None,
                   help="tool groups to expose; app-specific, see --list-groups")
    p.add_argument("--all-tools", action="store_true", help="expose every tool the app has")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--list-tools", action="store_true", help="print exposed tools and exit")
    p.add_argument("--list-groups", action="store_true",
                   help="print the tool groups for every app and exit")
    p.add_argument("--mcp", metavar="COMMAND",
                   help="drive any MCP stdio bridge instead of a registry app: the "
                        "command line that starts it, quoted as one argument")
    p.add_argument("--name", default="the app",
                   help="with --mcp, what to call the app the bridge drives")
    p.add_argument("--quiet", action="store_true", help="hide the step trace")
    a = p.parse_args()
    # Ctrl+Break and SIGTERM end the run the way Ctrl+C does, through the
    # `finally` that closes the bridge, rather than by killing the process.
    studio_procs.on_shutdown(_interrupt)
    if a.mcp:
        command, args = split_command(a.mcp)
        add_bridge(BridgeSpec(a.name, command, args, id="mcp"))
        a.app = "mcp"

    if a.list_groups:
        for app in TABS:
            print("%s (--app %s)" % (app.name, app.id))
            if app.custom:
                print("    groups are learned from the bridge when it starts; see --list-tools")
            for g, names in app.groups.items():
                mark = "*" if g in app.default_groups else " "
                print("  %s %-10s %s" % (mark, g, ", ".join(names)))
            print()
        print("* = on by default")
        return 0

    app = get_app(a.app)
    if app.panel:
        p.error("%s is a window in a tab of the app, with no model and no bridge; "
                "open it there" % app.name)
    _, loaded, ids, vision_ids, _ = probe_models(a.host)
    if not a.model:
        # The GUI does the same: an app may prefer a small model the host serves.
        a.model, note = app.model_for(ids, DEFAULT_MODEL)
        if note:
            log("  " + note, a.quiet)
    a.vision, note = resolve_vision(a.host, a.model, vision_ids, loaded)
    log("  " + (note or "vision: " + a.vision.model), a.quiet)
    if a.draft:
        os.environ["STUDIO_DRAFT_MODEL"] = a.draft
    a.draft, note = resolve_draft(a.model, ids)
    if note:
        log("  " + note, a.quiet)
    elif a.draft:
        log("  draft: %s (speculative decoding)" % a.draft, a.quiet)
    # The vision model is not loaded here, ahead of the executing model: the
    # host gives the card to whichever loads first. It loads just in time, on
    # the first picture - after `converse` has fitted the executing model.
    if not app.bridged:
        # No bridge to start and no tools to expose: the model on its own.
        if a.groups or a.all_tools:
            p.error("%s has no bridge, so there are no tool groups to choose" % app.name)
        if a.list_tools:
            print("%s has no bridge and exposes no tools." % app.name)
            return 0
        llm = LLM(a.host, a.model, a.temperature, draft=a.draft)
        log("  model: %s @ %s\n" % (a.model, a.host), a.quiet)
        return converse(llm, None, [], app, a, schemas=[])

    log(". connecting to the %s bridge..." % app.name, a.quiet)
    mcp = app.connect(quiet=a.quiet)
    mcp.on_elicit = elicit_at_terminal     # a bridge's questions come to this console
    try:
        info = mcp.initialize()
        srv = info.get("serverInfo", {})
        log("  bridge up: %s %s" % (srv.get("name", "?"), srv.get("version", "")), a.quiet)

        all_tools = mcp.list_tools()
        if app.custom:
            # Nothing was known about this bridge until now; its groups and its
            # briefing come from what it just answered.
            app.learn(all_tools, mcp.instructions)
        groups = [g.strip() for g in (a.groups or ",".join(app.default_groups)).split(",")
                  if g.strip()]
        for g in groups:
            if g not in app.groups:
                p.error("unknown group %r for %s; pick from %s"
                        % (g, app.name, ", ".join(app.groups)))
        if a.all_tools:
            wanted, chosen = None, all_tools
        else:
            wanted = app.tool_names(groups)
            chosen = [t for t in all_tools if t["name"] in wanted]

        log("  %d of %d tools exposed%s" % (len(chosen), len(all_tools),
            "" if a.all_tools else " (groups: %s)" % ",".join(groups)), a.quiet)

        if a.list_tools:
            for t in sorted(chosen, key=lambda x: x["name"]):
                print("%-26s %s" % (t["name"], (t.get("description") or "").split("\n")[0][:90]))
            missing = (wanted - {t["name"] for t in all_tools}) if wanted else set()
            if missing:
                print("\nnot served by this bridge: %s" % ", ".join(sorted(missing)))
            return 0

        if app.research:
            # The sidecar: this PC's files and the web, in process, beside the
            # bridge. Its tools go after the bridge's and its schemas with them.
            sidecar = research_client()
            extra = sidecar.list_tools()
            chosen = list(chosen) + extra
            mcp = Router(mcp, sidecar)
            log("  + %d research tools (files and the web)" % len(extra), a.quiet)
        tools = to_openai_tools(chosen)
        llm = LLM(a.host, a.model, a.temperature, draft=a.draft)
        log("  model: %s @ %s\n" % (a.model, a.host), a.quiet)
        if app.gpu_tools:
            # As in the window: the whole GPU to the render, the model back after.
            def room():
                ctx = next((c for _, c in loaded_instances(a.host, a.model)), None)
                gone, err = make_room(a.host, set())
                if gone or err:
                    log("  . made room on the GPU: unloaded %s%s" % (
                        ", ".join(gone) or "nothing", " (%s)" % err if err else ""), a.quiet)
                return ctx
            def back(ctx):
                # The vision model that looked at the picture goes first: a
                # model loads into an empty card in 3 s, beside another in 9-16.
                make_room(a.host, {a.model})
                err = give_back(a.host, a.model, ctx)
                if err:
                    log("  . could not reload %s: %s" % (a.model, err), a.quiet)
            mcp = YieldGPU(mcp, app.gpu_tools, room, back)

        return converse(llm, mcp, tools, app, a, schemas=chosen)
    finally:
        mcp.close()
