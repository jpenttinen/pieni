"""Credential destination regressions; no real SDK or network requests."""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import pieni
from tests.test_pieni import TempWorkspaceCase


class ProviderDestinationTests(TempWorkspaceCase):
    def test_checkout_cannot_redirect_user_provider_or_choose_custom_destination(self):
        for user_provider in (None, "openai", "https://trusted.invalid/v1"):
            with self.subTest(user_provider=user_provider):
                if user_provider:
                    self.write(self.home / ".pieni/pieni.ini",
                               f"[pieni]\nprovider = {user_provider}\nmodel = m\n")
                self.write(self.workspace / "pieni.ini",
                           "[pieni]\nprovider = https://attacker.invalid/v1\n")
                with self.assertRaisesRegex(pieni.ConfigError, "select this custom provider"):
                    pieni.load_config(cwd=self.workspace, home=self.home)

    def test_explicit_cli_provider_wins_over_hostile_local_destination(self):
        self.write(self.workspace / "pieni.ini",
                   "[pieni]\nprovider = https://attacker.invalid/v1\n")
        for provider in ("openai", "https://trusted.invalid/v1"):
            settings = pieni.load_config(cli_provider=provider,
                                         cwd=self.workspace, home=self.home)
            self.assertEqual(settings["provider"], provider)

    def test_custom_destination_in_user_settings_is_trusted(self):
        provider = "https://trusted.invalid/v1"
        self.write(self.home / ".pieni/pieni.ini", f"[pieni]\nprovider = {provider}\n")
        self.write(self.workspace / "pieni.ini", f"[pieni]\nprovider = {provider}\nmodel = m\n")
        settings = pieni.load_config(cwd=self.workspace, home=self.home)
        self.assertEqual(settings["provider"], provider)
        self.assertEqual(settings["model"], "m")

    def test_hostile_local_configuration_fails_before_client_creation(self):
        self.write(self.home / ".pieni/pieni.ini", "[pieni]\nprovider = openai\nmodel = m\n")
        self.write(self.workspace / "pieni.ini",
                   "[pieni]\nprovider = https://attacker.invalid/v1\n")
        previous = os.getcwd()
        try:
            os.chdir(self.workspace)
            with mock.patch("pieni.config_paths", return_value=[
                    self.home / ".pieni/pieni.ini", self.workspace / "pieni.ini"]), \
                    mock.patch("pieni.build_provider") as build, \
                    mock.patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic"}), \
                    mock.patch("sys.stderr"):
                self.assertEqual(pieni.main(["-r", "hello"]), 2)
                build.assert_not_called()
        finally:
            os.chdir(previous)


class CustomCredentialTests(unittest.TestCase):
    def test_custom_endpoint_uses_only_its_explicit_credential(self):
        for custom_key in (None, "custom-secret"):
            with self.subTest(custom_key=custom_key):
                environment = {"OPENAI_API_KEY": "openai-secret"}
                if custom_key:
                    environment["PIENI_CUSTOM_API_KEY"] = custom_key
                constructor = mock.Mock(return_value=SimpleNamespace(close=lambda: None))
                with mock.patch("pieni.load_sdk", return_value=constructor):
                    provider = pieni.build_provider("https://trusted.invalid/v1", "m", environment)
                    provider.close()
                self.assertEqual(constructor.call_args.kwargs, {
                    "api_key": custom_key or "not-needed", "base_url": "https://trusted.invalid/v1"})
