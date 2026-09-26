"""
studio_dwpose - a ComfyUI node that finds people's poses in a picture, for
Studio Assist's Scene Builder ("Pose from a photo...").

It answers with the points, not a drawing: DWPose's 133 COCO-WholeBody points
per person (17 body, 6 feet, 68 face, 21 per hand), in the picture's pixels,
as JSON text in the run's history (`ui.text`) and as a STRING output. The
Scene Builder fits the mannequin's controls to them.

Written here rather than installed (comfyui_controlnet_aux) because it needs
nothing ComfyUI's venv lacks - onnxruntime, OpenCV and numpy - and the two
model files, from https://huggingface.co/yzd-v/DWPose, in a `dwpose` model
folder (D:/ComfyUI-models/dwpose on the 5090, named in extra_model_paths.yaml):

    yolox_l.onnx           finds the people (YOLOX-L, COCO; class 0 is a person)
    dw-ll_ucoco_384.onnx   the points in each person's box (RTMPose, SimCC)

The pre- and post-processing follow DWPose's own onnxdet.py / onnxpose.py.
The source of truth is comfy_nodes/studio_dwpose in the Studio Assist repo;
copy it into ComfyUI's custom_nodes and restart ComfyUI.
"""

import json
import os

import numpy as np

import folder_paths

FOLDER = "dwpose"
DETECTOR = "yolox_l.onnx"
ESTIMATOR = "dw-ll_ucoco_384.onnx"
DET_SIZE = (640, 640)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)   # RGB
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))

_SESSIONS = {}


def _session(name):
    """One onnxruntime session per model file, kept for the next run. On
    the CPU on purpose: about a second a picture, no VRAM taken from the
    picture being made, and the 5090's onnxruntime-gpu 1.30 wants CUDA 13
    DLLs its torch (cu128) does not ship, so CUDA would only fail noisily."""
    if name in _SESSIONS:
        return _SESSIONS[name]
    path = folder_paths.get_full_path(FOLDER, name)
    if not path:
        raise FileNotFoundError(
            "%s is not in any '%s' model folder. Download it from "
            "https://huggingface.co/yzd-v/DWPose" % (name, FOLDER))
    import onnxruntime as ort
    s = _SESSIONS[name] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s


# ------------------------------------------------------------ the people
def _nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thr]
    return keep


def detect_people(bgr, score_thr=0.3):
    """-> [(x0, y0, x1, y1, score)] of every person, best first."""
    import cv2
    h, w = bgr.shape[:2]
    r = min(DET_SIZE[0] / h, DET_SIZE[1] / w)
    padded = np.full((DET_SIZE[0], DET_SIZE[1], 3), 114, dtype=np.uint8)
    small = cv2.resize(bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    padded[:small.shape[0], :small.shape[1]] = small
    x = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None]
    s = _session(DETECTOR)
    out = s.run(None, {s.get_inputs()[0].name: x})[0][0]
    grids, strides = [], []
    for stride in (8, 16, 32):
        hs, ws = DET_SIZE[0] // stride, DET_SIZE[1] // stride
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((hs * ws, 1), stride))
    grids, strides = np.concatenate(grids), np.concatenate(strides)
    out[:, :2] = (out[:, :2] + grids) * strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides
    score = out[:, 4] * out[:, 5]                 # objectness x "person"
    boxes = np.stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2,
                      out[:, 0] + out[:, 2] / 2, out[:, 1] + out[:, 3] / 2], 1) / r
    good = score > score_thr
    boxes, score = boxes[good], score[good]
    if not len(score):
        return []
    keep = _nms(boxes, score, 0.45)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return [tuple(float(v) for v in boxes[i]) + (float(score[i]),) for i in keep]


# ------------------------------------------------------------ the points
def estimate(rgb, box):
    """-> 133 [x, y, score] for the person in `box`, in picture pixels."""
    import cv2
    s = _session(ESTIMATOR)
    _, _, mh, mw = s.get_inputs()[0].shape
    x0, y0, x1, y1 = box[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sw, sh = (x1 - x0) * 1.25, (y1 - y0) * 1.25
    if sw > sh * mw / mh:                          # the box to the model's aspect
        sh = sw * mh / mw
    else:
        sw = sh * mw / mh
    m = np.array([[mw / sw, 0, -(cx - sw / 2) * mw / sw],
                  [0, mh / sh, -(cy - sh / 2) * mh / sh]], dtype=np.float32)
    crop = cv2.warpAffine(rgb, m, (int(mw), int(mh)), flags=cv2.INTER_LINEAR)
    x = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
    sx, sy = s.run(None, {s.get_inputs()[0].name: np.ascontiguousarray(x)})
    sx, sy = sx[0], sy[0]                          # (133, W*2), (133, H*2)
    ix, iy = sx.argmax(1), sy.argmax(1)
    vx, vy = sx.max(1), sy.max(1)
    split = sx.shape[1] / mw
    px = ix / split / mw * sw + cx - sw / 2
    py = iy / split / mh * sh + cy - sh / 2
    val = np.minimum(vx, vy)
    return [[round(float(a), 1), round(float(b), 1), round(float(c), 3)]
            for a, b, c in zip(px, py, val)]


def find_poses(image, max_people=8):
    """A ComfyUI IMAGE (1, H, W, 3 float 0..1) -> the JSON the node answers."""
    rgb = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    h, w = rgb.shape[:2]
    people = []
    for box in detect_people(bgr)[:max_people]:
        people.append({"box": [round(v, 1) for v in box[:4]], "score": round(box[4], 3),
                       "points": estimate(rgb, box)})
    return json.dumps({"width": w, "height": h, "format": "coco_wholebody_133",
                       "people": people})


class StudioDWPoseKeypoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)},
                "optional": {"max_people": ("INT", {"default": 8, "min": 1, "max": 32})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("keypoints",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Studio Assist"

    def run(self, image, max_people=8):
        text = find_poses(image, max_people)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {"StudioDWPoseKeypoints": StudioDWPoseKeypoints}
NODE_DISPLAY_NAME_MAPPINGS = {"StudioDWPoseKeypoints": "DWPose keypoints (Studio Assist)"}
