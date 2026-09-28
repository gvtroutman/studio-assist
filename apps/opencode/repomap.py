"""
OpenCode's structure map: an MCP server (stdio) that OpenCode always starts
(core/agent.py opencode_config, server "repo" - OpenCode names the tools
repo_map and repo_find). The local model otherwise greps blind or reads whole
files to learn what is where; this hands it the outline - classes, functions,
methods and constants with their line ranges - so its next read is one
function at the right offset.

Built fresh on every call from the files on disk, so it is right in a task's
worktree after edits too. Python is read with `ast`; JavaScript and other
text files by a few line patterns. Read-only; stdlib only.
"""
import ast
import os
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import core.mcp as studio_mcp

SKIP_DIRS = {".git", ".claude", "__pycache__", "node_modules", ".venv", "venv", "env",
             ".mypy_cache", ".pytest_cache", "dist", "build", ".idea", ".vscode"}
CODE_EXT = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}
TEXT_EXT = CODE_EXT | {".md", ".json", ".toml", ".cfg", ".ini", ".txt", ".bat", ".ps1", ".sh"}
MAX_CHARS = 12000     # one answer; the model's window is small
FIND_CAP = 40

JS_PATTERNS = [
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"),
]
MD_HEADING = re.compile(r"^(#{1,3})\s+(.*)")


class MapError(Exception):
    pass


def _resolve(path):
    p = os.path.abspath(os.path.expanduser(path or "."))
    if not os.path.exists(p):
        raise MapError("No such file or folder: %s" % path)
    return p


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def _walk(root):
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            yield os.path.join(dirpath, name)


# ------------------------------------------------------------------ outline

def outline_py(text):
    """[(depth, kind, name, first, last)] for a Python source."""
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return [(0, "error", "does not parse: %s (line %s)" % (e.msg, e.lineno), e.lineno or 1, e.lineno or 1)]
    out = []

    def first_line(node):
        decos = getattr(node, "decorator_list", None) or []
        return min([node.lineno] + [d.lineno for d in decos])

    def visit(body, depth):
        for node in body:
            if isinstance(node, ast.ClassDef):
                out.append((depth, "class", node.name, first_line(node), node.end_lineno))
                visit(node.body, depth + 1)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                if depth and args and args[0] in ("self", "cls"):
                    args = args[1:]
                out.append((depth, "def", "%s(%s)" % (node.name, ", ".join(args)),
                            first_line(node), node.end_lineno))
            elif depth == 0 and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.isupper() and len(t.id) > 1:
                        out.append((0, "const", t.id, node.lineno, node.end_lineno))
    visit(tree.body, 0)
    return out


def outline_js(lines):
    out = []
    for i, line in enumerate(lines, 1):
        for pat in JS_PATTERNS:
            m = pat.match(line)
            if m:
                depth = min(2, (len(line) - len(line.lstrip())) // 2)
                kind = "class" if "class" in line.split(m.group(1))[0] else "def"
                out.append((depth, kind, m.group(1), i, None))
                break
    return out


def outline_md(lines):
    return [(len(m.group(1)) - 1, "#", m.group(2).strip(), i, None)
            for i, line in enumerate(lines, 1) for m in [MD_HEADING.match(line)] if m]


def outline(path):
    lines = _lines(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return lines, outline_py("\n".join(lines))
    if ext in CODE_EXT:
        return lines, outline_js(lines)
    if ext == ".md":
        return lines, outline_md(lines)
    return lines, []


def _span(first, last):
    return "%d-%d" % (first, last) if last and last != first else "%d" % first


def file_map(path, rel=None):
    lines, items = outline(path)
    head = "%s  (%d lines)" % (rel or path, len(lines))
    if not items:
        return head + "\n  (no outline for this kind of file; read it with offset and limit)"
    rows = [head]
    for depth, kind, name, first, last in items:
        rows.append("%s%s %s  L%s" % ("  " * (depth + 1), kind, name, _span(first, last)))
    return "\n".join(rows)


def folder_map(root, depth):
    """Every text file under `root` with its line count and, to `depth`, its
    top-level names: 0 = files only, 1 = + classes and functions, 2 = + methods."""
    rows = []
    for path in _walk(root):
        ext = os.path.splitext(path)[1].lower()
        if ext not in TEXT_EXT:
            continue
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        try:
            lines, items = outline(path) if ext in CODE_EXT and depth else (_lines(path), [])
        except OSError:
            continue
        rows.append("%s  (%d lines)" % (rel, len(lines)))
        for d, kind, name, first, last in items:
            if d < depth and kind != "const":
                rows.append("%s%s %s  L%s" % ("  " * (d + 1), kind, name.split("(")[0], _span(first, last)))
    return rows


def _clip(rows, hint):
    text, out = 0, []
    for r in rows:
        if text + len(r) + 1 > MAX_CHARS:
            out.append("... %d more lines cut. %s" % (len(rows) - len(out), hint))
            break
        out.append(r)
        text += len(r) + 1
    return "\n".join(out)


# -------------------------------------------------------------------- tools

def t_map(args):
    p = _resolve(args.get("path"))
    if os.path.isfile(p):
        return studio_mcp.result(_clip(file_map(p).splitlines(),
                                       "Map a narrower file or use repo_find."))
    depth = int(args.get("depth", 1))
    rows = folder_map(p, depth)
    if not rows:
        return studio_mcp.result("No source or text files under %s." % p)
    return studio_mcp.result(_clip(rows, "Map a subfolder, or depth 0 for files only."))


def t_find(args):
    name = (args.get("name") or "").strip()
    if not name:
        raise MapError("Give the name of a function, class, method or constant.")
    root = _resolve(args.get("path"))
    files = [root] if os.path.isfile(root) else list(_walk(root))
    base = root if os.path.isdir(root) else os.path.dirname(root)
    hits = []
    for path in files:
        if os.path.splitext(path)[1].lower() not in CODE_EXT:
            continue
        try:
            _, items = outline(path)
        except OSError:
            continue
        rel = os.path.relpath(path, base).replace(os.sep, "/")
        parents = []
        for depth, kind, label, first, last in items:
            parents[depth:] = [label.split("(")[0]]
            if label.split("(")[0] == name:
                owner = ".".join(parents[:depth])
                hits.append("%s:%s  %s %s%s" % (rel, _span(first, last), kind,
                                                (owner + ".") if owner else "", label))
    if not hits:
        return studio_mcp.result("No definition of %r found. It may be made at run time; "
                                 "grep for it instead." % name)
    more = "" if len(hits) <= FIND_CAP else "\n... %d more." % (len(hits) - FIND_CAP)
    return studio_mcp.result("\n".join(hits[:FIND_CAP]) + more)


PATH = {"type": "string", "description": "A folder or file, absolute or relative to the "
        "project folder. Default: the project folder."}
TOOLS = [
    ("map", t_map,
     "The project's structure, from the files on disk. On a folder: every source and doc "
     "file with its line count and its classes and functions with line ranges. On one "
     "file: its full outline - classes, methods with their arguments, functions, "
     "constants, markdown headings - each with the lines it spans. Use it BEFORE reading "
     "a file, then read only the lines you need with offset and limit.",
     {"type": "object", "properties": {
         "path": PATH,
         "depth": {"type": "integer", "minimum": 0, "maximum": 2,
                   "description": "Folders only. 0 files, 1 + classes and functions "
                                  "(default), 2 + methods."}},
      "additionalProperties": False}),
    ("find", t_find,
     "Where a function, class, method or constant is defined: file, line range and the "
     "class it belongs to. Exact name, no parentheses. Faster and surer than grep for "
     "'def name'.",
     {"type": "object", "properties": {
         "name": {"type": "string", "description": "The exact name, e.g. fit_window."},
         "path": PATH},
      "required": ["name"], "additionalProperties": False}),
]

SERVER = studio_mcp.Server(
    "studio-repo-map", "1.0",
    studio_mcp.tools_from_table(TOOLS, read_only={n for n, *_ in TOOLS},
                                **{n: {"open_world": False} for n, *_ in TOOLS}),
    errors=(MapError, OSError, ValueError),
    instructions="The project's structure: map a folder or file to see what is where "
                 "with line ranges; find a name to see where it is defined. Read-only.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        os.chdir(sys.argv[1])
        sys.argv = sys.argv[:1]
    sys.exit(studio_mcp.main(SERVER))
