"""WithAnyone generation for Studio Assist. Runs only inside ComfyUI.

The installer supplies the pinned upstream FLUX implementation under vendor/.
Text encoding and VAE decoding use ComfyUI's stock nodes. This node owns the
identity model for one call and releases it even on cancellation or failure.
"""
import gc
import json
import os

import cv2
import folder_paths
import numpy as np
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from transformers import AutoProcessor, SiglipVisionModel

import comfy.model_management as mm
from comfy.utils import ProgressBar


def siglip_models():
    return sorted({name for root in folder_paths.get_folder_paths("diffusers")
                   if os.path.isdir(root) for name in os.listdir(root)
                   if os.path.isfile(os.path.join(root, name, "config.json"))})


class StudioWithAnyone:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "model": (folder_paths.get_filename_list("diffusion_models"),),
            "identity_model": (folder_paths.get_filename_list("diffusion_models"),),
            "siglip": (siglip_models(),),
            "face1": ("IMAGE",),
            "boxes": ("STRING", {"default": "[]", "multiline": True}),
            "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "guidance": ("FLOAT", {"default": 4.0, "min": 0, "max": 10}),
            "siglip_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 1}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
        }, "optional": {"face2": ("IMAGE",), "face3": ("IMAGE",), "face4": ("IMAGE",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Studio Assist"

    def generate(self, conditioning, model, identity_model, siglip, face1, boxes,
                 width, height, steps, guidance, siglip_weight, seed,
                 face2=None, face3=None, face4=None):
        from .vendor.withanyone.flux.pipeline import WithAnyonePipeline
        from .vendor.util import extract_moref

        refs = [x for x in (face1, face2, face3, face4) if x is not None]
        regions = json.loads(boxes)
        if len(regions) != len(refs):
            raise ValueError("WithAnyone needs one face position for each reference.")
        for box in regions:
            if (len(box) != 4 or not all(np.isfinite(v) and 0 <= v <= 1 for v in box)
                    or (box[2] - box[0]) * width < 4 or (box[3] - box[1]) * height < 4):
                raise ValueError("WithAnyone face positions must be inside the frame and at least 4 pixels wide and tall.")
        if not conditioning or "pooled_output" not in conditioning[0][1]:
            raise ValueError("WithAnyone needs FLUX text conditioning.")
        roots = folder_paths.get_folder_paths("insightface")
        root = next((r for r in roots if os.path.isfile(os.path.join(
            r, "models", "antelopev2", "glintr100.onnx"))), None)
        if root is None:
            raise ValueError("Install antelopev2 under ComfyUI/models/insightface/models/antelopev2.")
        siglip_path = next((os.path.join(r, siglip) for r in
                           folder_paths.get_folder_paths("diffusers")
                           if os.path.isdir(os.path.join(r, siglip))), None)
        if siglip_path is None:
            raise ValueError("The WithAnyone SigLIP model is missing.")

        pipeline = vision = analyser = None
        mm.unload_all_models()
        mm.soft_empty_cache()
        try:
            analyser = FaceAnalysis(name="antelopev2", root=root,
                                    allowed_modules=["detection", "recognition"],
                                    providers=["CPUExecutionProvider"])
            analyser.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.4)
            crops, embeddings = [], []
            for index, ref in enumerate(refs, 1):
                mm.throw_exception_if_processing_interrupted()
                picture = Image.fromarray((ref[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
                found = analyser.get(cv2.cvtColor(np.array(picture), cv2.COLOR_RGB2BGR))
                if len(found) != 1:
                    raise ValueError("Reference %d has %d detected faces. Crop it to show only that person." % (index, len(found)))
                crops.append(extract_moref(picture, {"bboxes": [found[0].bbox]}, 1)[0])
                embeddings.append(torch.from_numpy(found[0].embedding.copy()))
            processor = AutoProcessor.from_pretrained(siglip_path, local_files_only=True)
            vision = SiglipVisionModel.from_pretrained(siglip_path, local_files_only=True)
            pixels = processor(images=crops, return_tensors="pt").pixel_values
            siglip_embeddings = vision(pixels).last_hidden_state.unsqueeze(1)
            del vision
            vision = None
            device = mm.get_torch_device()
            mm.throw_exception_if_processing_interrupted()
            pipeline = WithAnyonePipeline(
                "flux-dev", folder_paths.get_full_path_or_raise("diffusion_models", identity_model),
                device, offload=True, only_lora=True, no_lora=True,
                flux_path=folder_paths.get_full_path_or_raise("diffusion_models", model))
            pipeline.model.to(device)

            class Progress:
                def __init__(self):
                    self.bar = ProgressBar(steps)

                def update(self, count):
                    mm.throw_exception_if_processing_interrupted()
                    self.bar.update(count)

            result = pipeline(
                txt=conditioning[0][0], vec=conditioning[0][1]["pooled_output"], prompt="",
                width=width, height=height, guidance=guidance, num_steps=steps, seed=seed,
                ref_imgs=crops, arcface_embeddings=torch.stack(embeddings),
                siglip_embeddings=siglip_embeddings,
                bboxes=[[[int(b[0]*width), int(b[1]*height), int(b[2]*width), int(b[3]*height)]
                         for b in regions]],
                id_weight=1.0-siglip_weight, siglip_weight=siglip_weight, pbar=Progress())
            return ({"samples": result.cpu()},)
        finally:
            # The upstream model is not a Comfy ModelPatcher and /free cannot
            # release it. Never leave it in a cached node output or on self.
            if pipeline is not None:
                pipeline.model.to("cpu")
            del pipeline, vision, analyser
            gc.collect()
            mm.soft_empty_cache()


NODE_CLASS_MAPPINGS = {"StudioWithAnyone": StudioWithAnyone}
NODE_DISPLAY_NAME_MAPPINGS = {"StudioWithAnyone": "Family photo (WithAnyone)"}
