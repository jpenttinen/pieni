"""Regression checks for private storage and failed context replacement."""
import os
import sqlite3
import stat
import unittest
from unittest import mock

import pieni
from tests.test_pieni import TempWorkspaceCase


class ContextRollbackTests(TempWorkspaceCase):
    def test_failed_message_commit_is_not_replayed_by_the_next_write(self):
        store = pieni.Store(self.workspace / pieni.DB_PATH)
        self.addCleanup(store.close)
        session = store.start_session("fake", "m", str(self.workspace))
        store.execute("PRAGMA busy_timeout=0")
        reader = sqlite3.connect(store.path)
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT payload FROM messages").fetchall()
            with self.assertRaisesRegex(pieni.PieniError, "database is locked"):
                store.add_message(session, {"role": "user", "content": "failed write"})
            self.assertFalse(store.connection.in_transaction)
        finally:
            reader.close()
        next_message = {"role": "user", "content": "next write"}
        store.add_message(session, next_message)
        self.assertEqual(store.resume("fake", "m", str(self.workspace))[1], [next_message])

    def test_interrupted_message_commit_is_rolled_back_before_next_write(self):
        class InterruptOnce(sqlite3.Connection):
            def commit(self):
                if self.interrupt:
                    self.interrupt = False
                    raise KeyboardInterrupt
                super().commit()

        store = pieni.Store(self.workspace / pieni.DB_PATH)
        self.addCleanup(store.close)
        session = store.start_session("fake", "m", str(self.workspace))
        store.connection.close()
        store.connection = sqlite3.connect(store.path, factory=InterruptOnce)
        store.connection.row_factory = sqlite3.Row
        store.connection.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            store.add_message(session, {"role": "user", "content": "interrupted write"})
        self.assertFalse(store.connection.in_transaction)
        next_message = {"role": "user", "content": "next write"}
        store.add_message(session, next_message)
        self.assertEqual(store.resume("fake", "m", str(self.workspace))[1], [next_message])

    def test_failed_message_save_does_not_extend_the_active_context(self):
        agent, _, store, _ = self.make_agent([])
        before = list(agent.messages)
        with mock.patch.object(store, "add_message", side_effect=pieni.PieniError("disk full")):
            with self.assertRaisesRegex(pieni.PieniError, "disk full"):
                agent.remember({"role": "user", "content": "not saved"})
        self.assertEqual(agent.messages, before)

    def test_failed_initial_save_still_reports_task_usage_and_handles_interrupt(self):
        for error in (pieni.PieniError("disk full"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                agent, provider, store, output = self.make_agent([])
                with mock.patch.object(store, "add_message", side_effect=error):
                    if isinstance(error, KeyboardInterrupt):
                        self.assertFalse(agent.run_task("not saved"))
                        self.assertIn("interrupted", output)
                    else:
                        with self.assertRaisesRegex(pieni.PieniError, "disk full"):
                            agent.run_task("not saved")
                self.assertEqual(agent.messages, [])
                self.assertEqual(provider.requests, [])
                self.assertTrue(output[-1].startswith("Tokens: 0 |"))

    def test_database_initialization_failure_closes_its_connection(self):
        connection = mock.Mock()
        connection.executescript.side_effect = sqlite3.OperationalError("disk full")
        with mock.patch("pieni.sqlite3.connect", return_value=connection):
            with self.assertRaisesRegex(pieni.PieniError, "cannot open the database.*disk full"):
                pieni.Store(self.workspace / pieni.DB_PATH)
        connection.close.assert_called_once_with()

    def test_failed_commit_cannot_be_committed_by_next_message(self):
        store = pieni.Store(self.workspace / pieni.DB_PATH)
        self.addCleanup(store.close)
        session = store.start_session("fake", "m", str(self.workspace))
        original = {"role": "user", "content": "original context"}
        boundary = store.add_message(session, original)
        store.execute("PRAGMA busy_timeout=0")
        reader = sqlite3.connect(store.path)
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT payload FROM messages").fetchall()
            with self.assertRaisesRegex(pieni.PieniError, "database is locked"):
                store.replace_context(session, boundary, {"role": "user", "content": "summary"})
            self.assertFalse(store.connection.in_transaction)
        finally:
            reader.close()
        next_message = {"role": "user", "content": "next task"}
        store.add_message(session, next_message)
        self.assertEqual(store.resume("fake", "m", str(self.workspace))[1],
                         [original, next_message])


@unittest.skipUnless(os.name == "posix", "requires POSIX file permissions")
class PrivateStorageTests(TempWorkspaceCase):
    def test_new_conversation_and_sqlite_sidecars_are_private_with_public_umask(self):
        old_umask = os.umask(0o022)
        try:
            database = self.workspace / pieni.DB_PATH
            store = pieni.Store(database)
            self.addCleanup(store.close)
            self.assertEqual(stat.S_IMODE(database.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)
            store.execute("PRAGMA journal_mode=WAL")
            store.start_session("fake", "m", str(self.workspace))
            for suffix in ("-wal", "-shm"):
                self.assertEqual(stat.S_IMODE(
                    type(database)(str(database) + suffix).stat().st_mode), 0o600)
        finally:
            os.umask(old_umask)

    def test_existing_history_is_tightened_without_changing_shared_parent(self):
        database = self.workspace / pieni.DB_PATH
        store = pieni.Store(database)
        session = store.start_session("fake", "m", str(self.workspace))
        message = {"role": "user", "content": "private"}
        store.add_message(session, message)
        store.close()
        self.workspace.chmod(0o755)
        database.parent.chmod(0o755)
        database.chmod(0o644)
        reopened = pieni.Store(database)
        self.addCleanup(reopened.close)
        self.assertEqual(stat.S_IMODE(self.workspace.stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE(database.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)
        self.assertEqual(reopened.resume("fake", "m", str(self.workspace))[1], [message])

    def test_symlink_storage_never_changes_target_permissions(self):
        target = self.root / "outside"
        target.mkdir(mode=0o755)
        link = self.workspace / pieni.APP_DIR
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(pieni.PieniError, "symbolic link"):
            pieni.Store(link / "pieni.db")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)
        link.unlink()
        link.mkdir()
        target_db = target / "pieni.db"
        target_db.write_bytes(b"sentinel")
        target_db.chmod(0o644)
        (link / "pieni.db").symlink_to(target_db)
        with self.assertRaisesRegex(pieni.PieniError, "symbolic link"):
            pieni.Store(link / "pieni.db")
        self.assertEqual(target_db.read_bytes(), b"sentinel")
        self.assertEqual(stat.S_IMODE(target_db.stat().st_mode), 0o644)

    def test_existing_sidecar_symlink_is_rejected_without_changing_target(self):
        directory = self.workspace / pieni.APP_DIR
        directory.mkdir()
        target = self.root / "outside.txt"
        target.write_text("sentinel", encoding="utf-8")
        target.chmod(0o644)
        (directory / "pieni.db-wal").symlink_to(target)
        with self.assertRaisesRegex(pieni.PieniError, "sidecar"):
            pieni.Store(directory / "pieni.db")
        self.assertEqual(target.read_text(encoding="utf-8"), "sentinel")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_hardlinked_database_and_sidecars_never_change_target_permissions(self):
        directory = self.workspace / pieni.APP_DIR
        directory.mkdir()
        for suffix in ("", "-journal", "-wal", "-shm"):
            with self.subTest(suffix=suffix):
                target = self.root / ("outside" + suffix)
                target.write_bytes(b"sentinel")
                target.chmod(0o644)
                linked = directory / ("pieni.db" + suffix)
                if linked.exists():
                    linked.unlink()
                os.link(target, linked)
                with self.assertRaisesRegex(pieni.PieniError, "expected one file link, found 2") as caught:
                    pieni.Store(directory / "pieni.db")
                self.assertIn(str(linked), str(caught.exception))
                self.assertEqual(target.read_bytes(), b"sentinel")
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)
                linked.unlink()

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires named pipes")
    def test_special_database_and_sidecars_are_rejected_without_waiting_for_io(self):
        directory = self.workspace / pieni.APP_DIR
        directory.mkdir()
        for suffix in ("", "-journal", "-wal", "-shm"):
            with self.subTest(suffix=suffix):
                path = directory / ("pieni.db" + suffix)
                if path.exists():
                    path.unlink()
                os.mkfifo(path, mode=0o644)
                with self.assertRaisesRegex(pieni.PieniError, "must be regular files") as caught:
                    pieni.Store(directory / "pieni.db")
                self.assertIn(str(path), str(caught.exception))
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
                path.unlink()
