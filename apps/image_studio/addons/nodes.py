"""Install only bundled, explicitly supported ComfyUI add-ons. Stdlib only."""

import ast
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import uuid

ROOT = Path(__file__).resolve().parents[3]
_LOCK = threading.Lock()
CATALOG = {
    "studio_matchtone": {
        "title": "Match repaired areas to the photo",
        "purpose": "Matches the colors and contrast of a redrawn patch to the original photograph.",
        "improves": "Helps Fix a spot repairs blend in instead of looking pinker, brighter or flatter than their surroundings.",
        "integration": "Used by the existing Fix a spot workflow. Requires NumPy and PyTorch in ComfyUI; no extra model download.",
        "node": "StudioMatchTone",
    },
    "studio_dwpose": {
        "title": "Pose from a photo",
        "purpose": "Finds body, hand and face pose points in a reference picture.",
        "improves": "Lets Scene Builder fit its mannequin to a photographed pose, reducing manual posing.",
        "integration": "Used by Pose from a photo. Requires OpenCV, NumPy, ONNX Runtime and the yolox_l.onnx and dw-ll_ucoco_384.onnx models in a ComfyUI dwpose model folder.",
        "node": "StudioDWPoseKeypoints",
    },
    "studio_facepaste": {
        "title": "Blend a real reference face",
        "purpose": "Aligns a face from a reference photo with the generated face and blends its color and edges.",
        "improves": "Helps preserve the person's actual facial details in the existing face-paste workflow.",
        "integration": "Requires InsightFace, antelopev2 models, ONNX Runtime, OpenCV, NumPy and PyTorch in ComfyUI. Installs the node code; these dependencies and models must already be available.",
        "node": "StudioFacePaste",
    },
}


def candidates():
    return [dict(info, addon_id=key, kind="Supported add-on", why=info["improves"],
                 details=info["integration"], importable=False)
            for key, info in CATALOG.items()]


def install(addon_id, comfy_folder):
    """Atomic single-file install with a retained backup, confined to custom_nodes.

    No fetched URLs, package commands, agent output or arbitrary paths can select
    the code. Existing unknown files in an add-on folder are left alone.
    """
    if addon_id not in CATALOG:
        raise ValueError("This add-on has no supported installer.")
    root = Path(comfy_folder).resolve(strict=True)
    if not (root / "main.py").is_file() or not (root / "folder_paths.py").is_file():
        raise ValueError("Choose the ComfyUI folder containing main.py and folder_paths.py.")
    source = ROOT / "comfy_nodes" / addon_id / "__init__.py"
    data = source.read_bytes()
    tree = ast.parse(data, filename=str(source))
    mappings = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS" for t in node.targets)]
    if not mappings or CATALOG[addon_id]["node"] not in {
            key.value for node in mappings if isinstance(node.value, ast.Dict)
            for key in node.value.keys if isinstance(key, ast.Constant)}:
        raise ValueError("The bundled add-on does not contain its expected node.")
    with _LOCK:
        target_dir = root / "custom_nodes" / addon_id
        target = target_dir / "__init__.py"
        # Resolve junctions / symlinks before any write, including a linked file.
        for path in (root / "custom_nodes", target_dir, target):
            if path.resolve() != path:
                raise ValueError("The add-on destination uses a link or junction. Choose a direct ComfyUI folder.")
        if target.is_file() and target.read_bytes() == data:
            return {"path": str(target), "backup": "", "unchanged": True}
        target_dir.mkdir(parents=True, exist_ok=True)
        previous = target.read_bytes() if target.exists() else None
        backup = ""
        if previous is not None:
            # Non-.py extension keeps ComfyUI from loading the backup as a node.
            backup = str(target_dir / ("__init__.py." + uuid.uuid4().hex + ".bak"))
            with open(backup, "xb") as stream:
                stream.write(previous)
        tmp = None
        replaced = False
        try:
            with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as stream:
                tmp = stream.name
                stream.write(data)
            os.replace(tmp, target)
            replaced = True
            if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(data).digest():
                raise OSError("Installed file verification failed.")
        except OSError:
            if replaced:
                if previous is None:
                    target.unlink()
                else:
                    target.write_bytes(previous)
            raise
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        return {"path": str(target), "backup": backup, "unchanged": False}
