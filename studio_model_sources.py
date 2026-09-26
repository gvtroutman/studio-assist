"""Local model-source settings, separate from generation prompts and jobs."""

import json
import os
from urllib.parse import urlsplit

import studio_civitai as civitai

SOURCES = {
    "huggingface": ("Hugging Face", "huggingface.co", "HF_TOKEN"),
    "civitai": ("Civitai", "civitai.com", civitai.TOKEN_ENV),
}


def path(root, source):
    SOURCES[source]
    return os.path.join(root, source + "-links.json")


def load(root, source):
    try:
        with open(path(root, source), encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    links = data.get("links", [])
    data["links"] = [x for x in links if isinstance(x, str)] if isinstance(links, list) else []
    data["token"] = (civitai.load_token(root) if source == "civitai" else
                     os.environ.get(SOURCES[source][2], str(data.get("token", "")))).strip()
    return data


def save(root, source, links, token):
    _name, domain, env = SOURCES[source]
    clean = []
    for link in links:
        link = link.strip()
        if not link:
            continue
        try:
            url = urlsplit(link)
            valid = (url.scheme == "https" and url.hostname in (domain, "www." + domain)
                     and not url.username and not url.password)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Use an https://%s model link on each line." % domain)
        if link not in clean:
            clean.append(link)
    data = {"links": clean}
    if source == "civitai":
        if not os.environ.get(env):
            civitai.save_token(root, token)
    else:
        # Do not copy a token supplied by the environment into a settings file.
        if not os.environ.get(env):
            data["token"] = token.strip()
        else:
            try:
                with open(path(root, source), encoding="utf-8") as stream:
                    previous = json.load(stream)
                if isinstance(previous, dict) and isinstance(previous.get("token"), str):
                    data["token"] = previous["token"]
            except (OSError, ValueError):
                pass
    os.makedirs(root, exist_ok=True)
    target = path(root, source)
    with open(target + ".tmp", "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
    os.replace(target + ".tmp", target)
