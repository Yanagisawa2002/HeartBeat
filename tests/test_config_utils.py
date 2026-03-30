import unittest
from pathlib import Path

import yaml

from src.config_utils import load_config, resolve_config_path


class TestConfigUtils(unittest.TestCase):
    def _get_temp_root(self) -> Path:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "config_utils"
        temp_root.mkdir(parents=True, exist_ok=True)
        return temp_root

    def test_resolve_config_path_finds_repo_config(self) -> None:
        resolved = resolve_config_path("configs/config.yaml")
        self.assertTrue(resolved.exists())
        self.assertEqual(resolved.name, "config.yaml")

    def test_load_config_reads_absolute_temp_file(self) -> None:
        config_path = self._get_temp_root() / "test_config.yaml"
        expected = {"device": "cpu", "seed": 7, "data": {"signal_length": 100}}
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(expected, handle)

        loaded = load_config(str(config_path))
        self.assertEqual(loaded["device"], "cpu")
        self.assertEqual(loaded["seed"], 7)
        self.assertEqual(loaded["data"]["signal_length"], 100)
