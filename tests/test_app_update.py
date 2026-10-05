"""core/app_update.py: the release a sidebar row shows, and its updater."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.agent as eng
import core.app_update as app_update

ADOBE = r"C:\Program Files\Adobe"
AE_2026 = ADOBE + r"\Adobe After Effects 2026\Support Files\AfterFX.exe"
ME_BETA = ADOBE + r"\Adobe Media Encoder (Beta)\Adobe Media Encoder (Beta).exe"


def version(number):
    return mock.patch.object(app_update.appinfo, "file_version", return_value=number)


class TestRelease(unittest.TestCase):
    def test_the_build_number_is_dropped(self):
        self.assertEqual(app_update.short("26.5.0.89"), "26.5.0")
        self.assertEqual(app_update.short("5.1"), "5.1")
        self.assertEqual(app_update.short(None), "")

    def test_the_other_installs_are_named_after_the_one_the_row_runs(self):
        with version("26.5.0.89"):
            self.assertEqual(app_update.release_line(AE_2026, ["2026", "Beta"]),
                             "26.5.0 + Beta")
            self.assertEqual(app_update.release_line(AE_2026, ["2026"]), "26.5.0")
        with version("27.1.0.5"):
            self.assertEqual(app_update.release_line(ME_BETA, ["2026", "Beta"]),
                             "27.1.0 Beta + 2026")
            self.assertEqual(app_update.release_line(ME_BETA, ["Beta"]), "27.1.0 Beta")

    def test_no_version_resource_means_no_claim(self):
        with version(None):
            self.assertEqual(app_update.release_line(AE_2026, ["2026"]), "")

    def test_a_folder_named_for_its_release_adds_nothing(self):
        exe = r"C:\Program Files\Blender Foundation\Blender 5.1\blender-launcher.exe"
        with version("5.1.1"):
            self.assertEqual(app_update.release_line(exe), "5.1.1")

    def test_read_does_not_take_a_bridges_word_for_an_install(self):
        row = {"name": "Some DAW", "id": "some-daw", "exe": AE_2026, "version": "bridge"}
        with version("1.2.3"), mock.patch.object(app_update, "slow_release",
                                                 return_value="") as slow:
            self.assertEqual(app_update.read(row), "")
        slow.assert_called_once()
        with version("26.5.0"):
            self.assertEqual(app_update.read(dict(row, version="2026, Beta")), "26.5.0 + Beta")

    def test_comfyui_is_asked_and_silence_costs_only_the_number(self):
        row = {"name": "ComfyUI", "id": "comfyui", "exe": None, "version": ""}
        page = mock.MagicMock()
        page.__enter__.return_value.read.return_value = \
            b'{"system": {"comfyui_version": "0.37.4"}}'
        with mock.patch("urllib.request.urlopen", return_value=page) as got:
            self.assertEqual(app_update.slow_release(row), "0.37.4")
        self.assertTrue(got.call_args[0][0].endswith("/system_stats"))
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            self.assertEqual(app_update.slow_release(row), "")

    def test_opencode_says_its_number_among_other_words(self):
        row = {"name": "OpenCode", "id": "opencode", "exe": None, "version": "server"}
        with mock.patch.object(eng, "opencode_exe", return_value="opencode.exe"), \
                mock.patch.object(app_update.appinfo, "command_version",
                                  return_value="opencode 1.18.32"):
            self.assertEqual(app_update.slow_release(row), "1.18.32")


class TestPlan(unittest.TestCase):
    def test_every_adobe_product_detection_knows_goes_to_creative_cloud(self):
        import core.agent_bridges as bridges
        self.assertEqual({p[2] for p in bridges.PRODUCTS}, set(app_update.ADOBE))
        step = app_update.plan({"name": "Photoshop", "id": "photoshop"})
        self.assertIn("Creative Cloud", step["label"])
        with mock.patch("os.path.isfile", return_value=False):
            step = app_update.plan({"name": "Photoshop", "id": "photoshop"})
        self.assertTrue(step["open"].startswith("https://"))

    def test_blender_goes_through_winget_when_there_is_one(self):
        row = {"name": "Blender", "id": None}
        with mock.patch("shutil.which", return_value=r"C:\winget.exe"):
            step = app_update.plan(row)
        self.assertEqual(step["run"][-3:], ["--id", "BlenderFoundation.Blender", "--exact"])
        with mock.patch("shutil.which", return_value=None):
            step = app_update.plan(row)
        self.assertEqual(step["open"], app_update.PAGES["Blender"])

    def test_opencode_upgrades_itself(self):
        with mock.patch.object(eng, "opencode_exe", return_value=r"C:\oc\opencode.exe"):
            step = app_update.plan({"name": "OpenCode", "id": "opencode"})
        self.assertEqual(step["run"][-2:], [r"C:\oc\opencode.exe", "upgrade"])

    def test_a_remote_app_says_where_it_is_updated(self):
        step = app_update.plan({"name": "ComfyUI", "id": "comfyui", "remote": True})
        self.assertIn("LLM PC", step["tip"])

    def test_a_bridge_entered_by_hand_has_no_updater(self):
        self.assertIsNone(app_update.plan({"name": "Some DAW", "id": "some-daw"}))

    def test_run_opens_or_starts_a_console(self):
        with mock.patch("os.startfile", create=True) as opened:
            app_update.run({"open": "https://example.com/"})
        opened.assert_called_once_with("https://example.com/")
        with mock.patch("subprocess.Popen") as started:
            app_update.run({"run": ["cmd", "/k", "ver"]})
        self.assertEqual(started.call_args[0][0], ["cmd", "/k", "ver"])


if __name__ == "__main__":
    unittest.main()
