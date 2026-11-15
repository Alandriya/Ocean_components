"""Tests for surgical main.py registration."""
import ast
import tempfile
import unittest
from pathlib import Path

from Forecast_1d.install_main_task import install, patch_source


CURRENT_MAIN_SKELETON = '''from pathlib import Path

def run_selected_task(task, **settings):
    """Dispatch task."""
    root = settings["files_path_prefix"]
    return root

def main():
    """Run selected task."""
    task = "sde_conditions"
    # ==========================================================================
    # SHARED SETTINGS: preserved from the current GitHub entrypoint.
    files_path_prefix = "D:/Nastya/Data/OceanFull/"
    settings = locals().copy()
    settings.pop("task", None)
    return run_selected_task(task, **settings)

if __name__ == "__main__":
    main()
'''


class MainTaskInstallerTest(unittest.TestCase):
    def test_patch_is_idempotent_and_preserves_old_task(self):
        amended = patch_source(CURRENT_MAIN_SKELETON)
        self.assertEqual(amended, patch_source(amended))
        ast.parse(amended)
        self.assertIn('task = "extract_1d_series"', amended)
        self.assertIn('Previous active selection: task = "sde_conditions"', amended)
        self.assertIn('extract_1d_overwrite = False', amended)
        self.assertIn('from Forecast_1d.extract_points import DEFAULT_SUBDIR, extract', amended)
        self.assertIn('files_path_prefix = "D:/Nastya/Data/OceanFull/"', amended)

    def test_installer_creates_backup_and_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'main.py'
            target.write_text(CURRENT_MAIN_SKELETON, encoding='utf-8')
            changed, backup = install(target)
            self.assertTrue(changed)
            self.assertEqual(backup.read_text(encoding='utf-8'), CURRENT_MAIN_SKELETON)
            changed, backup = install(target)
            self.assertFalse(changed)
            self.assertIsNone(backup)

    def test_keep_current_task(self):
        amended = patch_source(CURRENT_MAIN_SKELETON, select_task=False)
        self.assertIn('    task = "sde_conditions"', amended)
        self.assertIn('    if task == "extract_1d_series":', amended)

    def test_fails_on_unrecognized_dispatcher(self):
        unknown = CURRENT_MAIN_SKELETON.replace('    root = settings["files_path_prefix"]',
                                                '    root = settings.get("files_path_prefix")')
        with self.assertRaisesRegex(RuntimeError, 'root anchor'):
            patch_source(unknown)


if __name__ == '__main__':
    unittest.main()
