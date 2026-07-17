from __future__ import annotations

from importlib.metadata import entry_points


EXPECTED_COMMANDS = {
    "captcha-train": "captcha_vision.training.train:main",
    "captcha-evaluate": "captcha_vision.training.evaluate:main",
    "captcha-predict": "captcha_vision.cli_scripts.predict:main",
    "captcha-solve": "captcha_vision.cli_scripts.solve:main",
    "captcha-compare": "captcha_vision.cli_scripts.compare_experiments:main",
}


def test_console_entry_points_are_installed_and_importable() -> None:
    installed = {
        entry.name: entry
        for entry in entry_points(group="console_scripts")
        if entry.name in EXPECTED_COMMANDS
    }

    assert {name: entry.value for name, entry in installed.items()} == EXPECTED_COMMANDS
    for entry in installed.values():
        assert callable(entry.load())
