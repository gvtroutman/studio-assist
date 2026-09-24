#!/usr/bin/env python3
"""Files and folders the user attaches to a message, described for a model.

Split out of `studio_chat` because none of it is about the window: it reads
headers, sizes and directory listings and returns sentences. **Nothing here
imports tkinter**, so it can be exercised on a machine with no display, which
is most of what these functions are worth testing for - `image_dims` reads
five formats' headers by hand, and `attachment_note` copies files into a
container's workspace.
"""

import os
import shutil
import socket

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff")
ATTACH_TYPES = [("All files", "*.*"), ("Pictures", " ".join("*" + e for e in IMAGE_EXTS))]
PREVIEWABLE = (".png", ".gif")            # what Tk 8.6 can decode without PIL
ATTACH_LIMIT = 200_000_000                # bytes; a picture, not a video
LIST_LIMIT = 40                           # folder entries named in the brief


def is_picture(path):
    """Whether an attachment is a picture - the ones the vision model is
    asked about and the transcript tries to show."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def image_dims(path):
    """(width, height) from the file header, or None. PNG, GIF and JPEG only -
    the formats a camera, a screenshot or an export actually produces - and
    read without decoding, so a 200 MB TIFF costs nothing to attach."""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
            if head[:6] in (b"GIF87a", b"GIF89a"):
                return (int.from_bytes(head[6:8], "little"), int.from_bytes(head[8:10], "little"))
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        return None
                    if marker[1] in (0xD8, 0x01) or 0xD0 <= marker[1] <= 0xD7:
                        continue
                    size = int.from_bytes(f.read(2), "big")
                    if marker[1] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        sof = f.read(5)
                        return (int.from_bytes(sof[3:5], "big"), int.from_bytes(sof[1:3], "big"))
                    f.seek(size - 2, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def describe_folder(path):
    """A folder as a heading and a bounded listing: what is in it, so the
    model can name a file without a tool call, and how much was left out."""
    try:
        names = sorted(os.listdir(path), key=str.lower)
    except OSError as e:
        return "%s (folder, unreadable: %s) at %s" % (os.path.basename(path) or path,
                                                     e.strerror or e, path)
    files = [n for n in names if os.path.isfile(os.path.join(path, n))]
    dirs = [n for n in names if os.path.isdir(os.path.join(path, n))]
    head = "%s (folder, %d files, %d folders) at %s" % (
        os.path.basename(path) or path, len(files), len(dirs), path)
    shown = [n + "/" for n in dirs] + files
    lines = ["    " + n for n in shown[:LIST_LIMIT]]
    if len(shown) > LIST_LIMIT:
        lines.append("    ... and %d more" % (len(shown) - LIST_LIMIT))
    return "\n".join([head] + lines)


def describe_attachment(path):
    """One line a text model can act on: name, size, dimensions when it is a
    picture, and the path every bridge on this PC opens files by. A folder
    gets its listing."""
    if os.path.isdir(path):
        return describe_folder(path)
    ext = os.path.splitext(path)[1].lstrip(".").upper() or "file"
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    detail = ["%.1f MB" % (size / 1e6) if size >= 1e6 else "%d KB" % max(1, size // 1000)]
    dims = image_dims(path)
    if dims:
        detail.insert(0, "%d x %d" % dims)
    return "%s (%s %s) at %s" % (os.path.basename(path), ", ".join(detail), ext, path)


def attachment_note(paths, app):
    """The paragraph appended to the brief when files or folders are attached.
    Bridges on this PC take the path as it is; the OpenCode container sees
    only its workspace, so attachments are copied in and named by the path
    the container will see."""
    if not paths:
        return ""
    lines = []
    if getattr(app, "container", False):
        folder = os.path.join(app.workspace, "attachments")
        os.makedirs(folder, exist_ok=True)
        for p in paths:
            name = os.path.basename(os.path.normpath(p))
            dest = os.path.join(folder, name)
            if os.path.abspath(dest) != os.path.abspath(p):
                if os.path.isdir(p):
                    shutil.copytree(p, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(p, dest)
            lines.append("- %s (copied into the workspace; the container sees it as "
                         "/workspace/attachments/%s)" % (describe_attachment(dest), name))
        head = "Attached files and folders, copied into the workspace:"
    else:
        head = ("Attached files and folders - on this PC; tools that take a path "
                "(import, place, upload, open, read_file) take these paths as written:")
        lines = ["- " + describe_attachment(p) for p in paths]
    return "\n\n" + head + "\n" + "\n".join(lines)


def this_pc():
    """The machine name, for the sidebar heading - this is the PC being driven."""
    try:
        return socket.gethostname().upper()
    except Exception:
        return "THIS PC"
