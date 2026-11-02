"""Export the remote service without installing anything on the workstation."""
import hashlib
import json
from pathlib import Path
import zipfile


def export(destination):
    root = Path(__file__).resolve().parents[2]
    files = ['apps/__init__.py', 'apps/voice_studio/__init__.py',
             'apps/voice_studio/catalog.py', 'apps/voice_studio/client.py',
             'apps/voice_studio/remote.py', 'apps/voice_studio/worker.py',
             'apps/voice_studio/setup_remote.ps1', 'apps/voice_studio/start_remote.ps1',
             'core/__init__.py', 'core/procs.py', 'docs/voice-studio.md']
    manifest = {}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        readme = ('STUDIO ASSIST VOICE — REMOTE SETUP\n\n'
                  'Extract this ZIP on the LLM PC. Keep the folder after setup.\n'
                  'Install Python 3.11. VoxCPM also needs shared FFmpeg DLLs on PATH.\n'
                  'Read docs/voice-studio.md for requirements and optional engines.\n\n'
                  'From this extracted folder in PowerShell:\n'
                  'Set-ExecutionPolicy -Scope Process Bypass\n'
                  '.\\apps\\voice_studio\\setup_remote.ps1 -Python python -Engines voxcpm,qwen,omnivoice\n'
                  '.\\apps\\voice_studio\\start_remote.ps1\n\n'
                  'In Studio Assist: + > Voice Studio. Enter the URL and the token\n'
                  'from voice-token.txt, then Connect. Allow remote model downloads\n'
                  'for the first generation. Audition takes before choosing one.\n\n'
                  'The bundle contains no models or credentials. Local protocol and\n'
                  'GUI tests use synthetic audio; live engine setup and speech quality\n'
                  'still need verification on your LLM PC.\n').encode('utf8')
        archive.writestr('README.txt', readme)
        manifest['README.txt'] = hashlib.sha256(readme).hexdigest()
        for name in files:
            data = (root / name).read_bytes()
            archive.writestr(name, data)
            manifest[name] = hashlib.sha256(data).hexdigest()
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    return str(destination)
