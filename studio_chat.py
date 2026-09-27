"""Starts Studio Assist. The Start Menu shortcut runs this file; the app is core/chat.py."""
import runpy

if __name__ == "__main__":
    runpy.run_module("core.chat", run_name="__main__", alter_sys=True)
