"""Capability-gated workflow vocabulary; no second executor or inference client.

Inspection adapters inherit the enabled bridge's exact argument contract. Other
workflows stay discoverable but uncallable until a real implementation exists.
"""
from copy import deepcopy


INSPECTIONS = {
    "inspect_project": "get_project_summary",
    "inspect_comp": "get_comp",
    "inspect_layer": "get_layer_full",
}

PENDING = {
    "inspect_audio": "Needs an audio metadata/analysis provider; layer metadata alone cannot inspect sound.",
    "get_transcript": "Needs a timestamped transcript source or remote transcription provider.",
    "find_matching_media": "Needs a searchable media index and a matching implementation.",
    "place_media_from_transcript": "Needs timestamped transcript segments, verified source ranges, and an app placement adapter.",
    "apply_miter_style": "Needs a versioned Miter style reference and an app adapter.",
    "apply_ellwood_style": "Needs a versioned Ellwood style reference and an app adapter.",
    "build_red_card": "Needs the approved red-card template, copy, and a construction adapter.",
    "sync_animation_to_speech": "Needs speech timestamps and an animation timing adapter.",
    "verify_comp": "Needs explicit acceptance checks and an evaluator; a successful read is not a visual pass.",
    "verify_timing": "Needs expected cue times, observed animation/media times, and frame tolerances.",
}

CAPABILITY_TOOL = {"type": "function", "function": {
    "name": "studio_workflow_capabilities",
    "description": "List implemented workflow tools and blockers for planned workflows in this tab. This only reports capabilities; it does not inspect or edit the project.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}}


def adapters(tools):
    """Derive only from enabled tools, never the broader bridge catalogue."""
    enabled = {tool["function"]["name"]: tool for tool in tools}
    result = {}
    for name, target in INSPECTIONS.items():
        if target in enabled and name not in enabled:
            tool = deepcopy(enabled[target])
            tool["function"]["name"] = name
            tool["function"]["description"] = (
                "Workflow inspection via " + target + ". Returns bridge observations, "
                "not a verification verdict. Original bridge contract follows:\n" +
                tool["function"].get("description", ""))
            result[name] = (target, tool)
    return result


def model_tools(tools):
    return list(tools) + [value[1] for value in adapters(tools).values()] + [CAPABILITY_TOOL]


def capabilities(tools):
    available = adapters(tools)
    result = {}
    for name, target in INSPECTIONS.items():
        result[name] = ({"available": True, "bridge_tool": target} if name in available else
                        {"available": False, "reason": "No enabled adapter for this tab (requires " + target + ")."})
    result.update({name: {"available": False, "reason": reason}
                   for name, reason in PENDING.items()})
    return result
