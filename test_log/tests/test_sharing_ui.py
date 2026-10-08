"""Sharing callbacks use the frozen service API and keep imports explicit."""

import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vd_test_log.models import DataPaths, EventLayout, TestDay, new_id, utc_now_iso
from vd_test_log import sharing_ui
from vd_test_log.services import TestLogServices
from vd_test_log.ui import TestLogWindow


@dataclass(frozen=True)
class Counts:
    days: int = 0
    setups: int = 0
    laps: int = 0
    event_layouts: int = 0
    attachments: int = 0


@dataclass(frozen=True)
class Conflict:
    kind: str
    record_id: str
    description: str


@dataclass(frozen=True)
class Preview:
    source_label: str
    fingerprint: str
    incoming: Counts
    added: Counts
    skipped: Counts
    conflicts: tuple[Conflict, ...]
    separate_copy: bool


@dataclass(frozen=True)
class ImportResult:
    source_label: str
    fingerprint: str
    added: Counts
    skipped: Counts
    backup_path: Path | None


class FakeEventManager:
    def __init__(self, *, allow=True):
        self.allow = allow
        self._record = SimpleNamespace(id="existing-event")
        self._is_new = False
        self.resolve_calls = 0
        self.refreshed_ids = []
        self.loaded_ids = []

    def winfo_exists(self):
        return True

    def _resolve_unsaved(self):
        self.resolve_calls += 1
        return self.allow

    def refresh_tree(self, select_id=None):
        self.refreshed_ids.append(select_id)

    def _load_event(self, event_id):
        self.loaded_ids.append(event_id)


class SharingUiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp.name)
        data_root = self.root_path / "data"
        paths = DataPaths(
            root=data_root,
            database=data_root / "test_log.sqlite3",
            attachments=data_root / "attachments",
            lock_file=data_root / ".test_log.lock",
            backup_root=data_root / "backups",
        )
        self.services = TestLogServices.open(paths)
        self.root = tk.Tk()
        self.root.withdraw()
        self.window = TestLogWindow(self.root, self.services)
        self.window.build()
        self.root.update()

    def tearDown(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.services.close()
        self.temp.cleanup()

    @staticmethod
    def make_preview(*, separate=False, conflicts=(), laps=2):
        return Preview(
            source_label="Morgan's laptop",
            fingerprint="package-fingerprint-1",
            incoming=Counts(days=1, setups=2, laps=laps, event_layouts=1, attachments=3),
            added=Counts(days=1, setups=1, laps=laps, event_layouts=1, attachments=3),
            skipped=Counts(setups=1),
            conflicts=tuple(conflicts),
            separate_copy=separate,
        )

    def find_widgets(self, parent, widget_type):
        found = []
        for child in parent.winfo_children():
            if isinstance(child, widget_type):
                found.append(child)
            found.extend(self.find_widgets(child, widget_type))
        return found

    def interact_with_import_dialog(self, callback):
        errors = []

        def interact():
            dialog = None
            try:
                dialog = next(
                    child
                    for child in self.root.winfo_children()
                    if isinstance(child, tk.Toplevel) and child.title() == "Import Log Package"
                )
                callback(dialog)
            except BaseException as error:
                errors.append(error)
                if dialog is not None and dialog.winfo_exists():
                    buttons = self.find_widgets(dialog, tk.Button) + self.find_widgets(dialog, ttk.Button)
                    cancel = next((button for button in buttons if button.cget("text") == "Cancel"), None)
                    if cancel is not None:
                        cancel.invoke()

        after_id = self.root.after(0, interact)
        try:
            self.window.sharing_ui.import_package()
        except BaseException:
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
            raise
        if errors:
            raise errors[0]

    def test_menu_is_present_and_main_minimum_geometry_is_preserved(self):
        menu = self.window.menu_bar
        labels = [menu.entrycget(index, "label") for index in range(menu.index("end") + 1)]

        self.assertIn("Sharing", labels)
        self.assertEqual(self.root.minsize(), (900, 630))

    def test_package_count_summary_uses_outing_wording(self):
        summary = sharing_ui._format_counts(Counts(setups=2))

        self.assertIn("2 outings", summary)
        self.assertNotIn("setup", summary.lower())

    def test_export_cancel_and_optional_label_do_not_write_early(self):
        export = Mock(return_value=self.root_path / "saved.zip")
        self.services.export_share_package = export

        with patch("tkinter.filedialog.asksaveasfilename", return_value="") as choose_path, patch(
            "tkinter.simpledialog.askstring"
        ) as choose_label:
            self.window.sharing_ui.export_package()
        choose_path.assert_called_once()
        choose_label.assert_not_called()
        export.assert_not_called()

        destination = self.root_path / "personal logs.zip"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(destination)), patch(
            "tkinter.simpledialog.askstring", return_value=None
        ):
            self.window.sharing_ui.export_package()
        export.assert_not_called()

        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(destination)), patch(
            "tkinter.simpledialog.askstring", return_value=""
        ):
            self.window.sharing_ui.export_package()
        export.assert_called_once_with(destination, source_label="")
        self.assertIn("Exported", self.window.status_label.cget("text"))

    def test_dirty_cancel_stops_before_package_picker(self):
        self.services.preview_share_package = Mock()
        with patch.object(self.window, "_resolve_unsaved", return_value=False), patch(
            "tkinter.filedialog.askopenfilename"
        ) as choose_package:
            self.window.sharing_ui.import_package()

        choose_package.assert_not_called()
        self.services.preview_share_package.assert_not_called()

    def test_conflict_disables_merge_and_separate_copy_uses_preview_fingerprint(self):
        package = self.root_path / "from-driver.zip"
        conflict = Conflict("lap", "same-id", "Saved lap fields differ from the package.")
        merge_preview = self.make_preview(conflicts=(conflict,))
        separate_preview = self.make_preview(separate=True, conflicts=(), laps=5)
        self.services.preview_share_package = Mock(side_effect=(merge_preview, separate_preview))
        backup = self.root_path / "backup-2026-10-06"
        self.services.import_share_package = Mock(
            return_value=ImportResult(
                source_label="Morgan's laptop",
                fingerprint=merge_preview.fingerprint,
                added=separate_preview.added,
                skipped=separate_preview.skipped,
                backup_path=backup,
            )
        )
        event_manager = FakeEventManager()
        self.window.event_manager = event_manager

        def choose_separate_and_import(dialog):
            warning = next(
                widget
                for widget in self.find_widgets(dialog, ttk.Label)
                if "whole incoming log as a separate copy" in widget.cget("text")
            )
            self.assertIn("A changed log may duplicate previously imported history.", warning.cget("text"))
            radio = next(
                widget
                for widget in self.find_widgets(dialog, ttk.Radiobutton)
                if widget.cget("text") == "Merge new records into this log"
            )
            self.assertIn("disabled", radio.state())
            self.assertIn("5 laps", " ".join(
                str(widget.cget("text"))
                for widget in self.find_widgets(dialog, ttk.Label)
            ))
            import_button = next(
                widget
                for widget in self.find_widgets(dialog, ttk.Button)
                if widget.cget("text") == "Import"
            )
            import_button.invoke()

        with patch("tkinter.filedialog.askopenfilename", return_value=str(package)), patch(
            "tkinter.messagebox.askyesno", return_value=True
        ) as confirm, patch("tkinter.messagebox.showinfo"):
            self.interact_with_import_dialog(choose_separate_and_import)

        self.services.preview_share_package.assert_any_call(package, separate_copy=False)
        self.services.preview_share_package.assert_any_call(package, separate_copy=True)
        confirm.assert_called_once()
        self.services.import_share_package.assert_called_once_with(
            package,
            separate_copy=True,
            expected_fingerprint=merge_preview.fingerprint,
        )
        self.assertIn(str(backup), self.window.status_label.cget("text"))
        self.assertEqual(event_manager.refreshed_ids, ["existing-event"])
        self.assertEqual(event_manager.loaded_ids, ["existing-event"])

    def test_event_manager_dirty_cancel_stops_before_share_dialogs(self):
        event_manager = FakeEventManager(allow=False)
        self.window.event_manager = event_manager
        with patch.object(self.window, "_resolve_unsaved", return_value=True), patch(
            "tkinter.filedialog.askopenfilename"
        ) as choose_import, patch("tkinter.filedialog.asksaveasfilename") as choose_export:
            self.window.sharing_ui.import_package()
            self.window.sharing_ui.export_package()

        self.assertEqual(event_manager.resolve_calls, 2)
        choose_import.assert_not_called()
        choose_export.assert_not_called()

    def test_canceling_separate_copy_confirmation_never_applies_import(self):
        package = self.root_path / "from-driver.zip"
        self.services.preview_share_package = Mock(
            side_effect=(self.make_preview(), self.make_preview(separate=True))
        )
        self.services.import_share_package = Mock()

        def choose_separate_and_continue_to_confirmation(dialog):
            radio = next(
                widget
                for widget in self.find_widgets(dialog, ttk.Radiobutton)
                if widget.cget("text") == "Import as a separate snapshot"
            )
            radio.invoke()
            button = next(
                widget
                for widget in self.find_widgets(dialog, ttk.Button)
                if widget.cget("text") == "Import"
            )
            button.invoke()

        with patch("tkinter.filedialog.askopenfilename", return_value=str(package)), patch(
            "tkinter.messagebox.askyesno", return_value=False
        ) as confirm:
            self.interact_with_import_dialog(choose_separate_and_continue_to_confirmation)

        confirm.assert_called_once()
        self.services.import_share_package.assert_not_called()

    def test_discarding_dirty_editors_restores_saved_form_values(self):
        day = TestDay(
            id=new_id(),
            date="2026-10-06",
            location="Test Site",
            weather=None,
            notes="Saved day note",
            created_at=utc_now_iso(),
        )
        self.services.save_day(day)
        self.window.refresh_tree(select_item=f"day:{day.id}")
        self.window.tree.selection_set(f"day:{day.id}")
        self.root.update()
        self.window.manage_events()
        manager = self.window.event_manager
        event = self.services.save_event_layout(
            EventLayout(
                id=new_id(),
                track_name="Test Site",
                layout_name="Short Loop",
                event_name=None,
                event_type=None,
                length_m=None,
                notes=None,
                archived=False,
                created_at=utc_now_iso(),
            )
        )
        manager._load_event(event.id)
        manager.fields["layout_name"].delete(0, "end")
        manager.fields["layout_name"].insert(0, "Discard this event edit")
        self.window.fields["notes"].delete("1.0", "end")
        self.window.fields["notes"].insert("1.0", "Discard this day edit")

        with patch("tkinter.messagebox.askyesnocancel", side_effect=(False, False)), patch(
            "tkinter.filedialog.asksaveasfilename", return_value=""
        ):
            self.window.sharing_ui.export_package()

        self.assertEqual(self.window.fields["notes"].get("1.0", "end-1c"), "Saved day note")
        self.assertEqual(manager.fields["layout_name"].get(), "Short Loop")
        self.assertFalse(self.window._is_dirty())
        self.assertFalse(manager._is_dirty())

    def test_import_dialog_cancel_never_applies_preview(self):
        package = self.root_path / "from-driver.zip"
        preview = self.make_preview()
        self.services.preview_share_package = Mock(return_value=preview)
        self.services.import_share_package = Mock()

        def cancel(dialog):
            button = next(
                widget
                for widget in self.find_widgets(dialog, ttk.Button)
                if widget.cget("text") == "Cancel"
            )
            button.invoke()

        with patch("tkinter.filedialog.askopenfilename", return_value=str(package)):
            self.interact_with_import_dialog(cancel)

        self.services.import_share_package.assert_not_called()

    def test_package_preview_error_is_reported_and_ui_stays_available(self):
        package = self.root_path / "broken.zip"
        self.services.preview_share_package = Mock(side_effect=ValueError("invalid package header"))
        self.services.import_share_package = Mock()
        with patch("tkinter.filedialog.askopenfilename", return_value=str(package)), patch(
            "tkinter.messagebox.showerror"
        ) as showerror:
            self.window.sharing_ui.import_package()

        self.assertIn("invalid package header", self.window.status_label.cget("text"))
        showerror.assert_called_once()
        self.services.import_share_package.assert_not_called()
        self.assertTrue(self.root.winfo_exists())

    @unittest.skipUnless(
        hasattr(TestLogServices, "export_share_package")
        and hasattr(TestLogServices, "preview_share_package")
        and hasattr(TestLogServices, "import_share_package"),
        "offline package service is being implemented in parallel",
    )
    def test_real_service_import_callback_updates_the_central_log(self):
        source_root = self.root_path / "source data"
        source_paths = DataPaths(
            root=source_root,
            database=source_root / "test_log.sqlite3",
            attachments=source_root / "attachments",
            lock_file=source_root / ".test_log.lock",
            backup_root=source_root / "backups",
        )
        source = TestLogServices.open(source_paths)
        source_root = tk.Toplevel(self.root)
        source_root.withdraw()
        try:
            day = TestDay(
                id=new_id(),
                date="2026-10-06",
                location="North Loop",
                weather=None,
                notes="Offline handoff",
                created_at=utc_now_iso(),
            )
            source.save_day(day)
            package = self.root_path / "shared logs.zip"
            source_window = TestLogWindow(source_root, source)
            source_window.build()
            with patch("tkinter.filedialog.asksaveasfilename", return_value=str(package)), patch(
                "tkinter.simpledialog.askstring", return_value="Morgan"
            ):
                source_window.sharing_ui.export_package()
            self.assertTrue(package.is_file())

            def import_package(dialog):
                import_button = next(
                    widget
                    for widget in self.find_widgets(dialog, ttk.Button)
                    if widget.cget("text") == "Import"
                )
                import_button.invoke()

            with patch("tkinter.filedialog.askopenfilename", return_value=str(package)), patch(
                "tkinter.messagebox.showinfo"
            ):
                self.interact_with_import_dialog(import_package)

            self.assertEqual(self.services.get_day(day.id), day)
            self.assertTrue(self.window.tree.exists(f"day:{day.id}"))
            self.assertIn("Morgan", self.window.status_label.cget("text"))
        finally:
            source_root.destroy()
            source.close()


if __name__ == "__main__":
    unittest.main()
