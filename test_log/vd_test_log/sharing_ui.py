"""Dialogs and callbacks for offline log package exchange."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk


def _format_counts(counts) -> str:
    units = (
        (counts.days, "day", "days"),
        (counts.setups, "setup", "setups"),
        (counts.laps, "lap", "laps"),
        (counts.event_layouts, "event/layout", "event/layouts"),
        (counts.attachments, "attachment", "attachments"),
    )
    return ", ".join(
        f"{value} {singular if value == 1 else plural}"
        for value, singular, plural in units
    )


def _has_records(counts) -> bool:
    return any(
        getattr(counts, name)
        for name in ("days", "setups", "laps", "event_layouts", "attachments")
    )


class ImportPreviewDialog(tk.Toplevel):
    """Show validated package counts and let the user choose a safe import mode."""

    def __init__(self, parent, preview, preview_for_mode):
        super().__init__(parent)
        self.preview = preview
        self.preview_for_mode = preview_for_mode
        self.merge_allowed = not bool(preview.conflicts)
        self.result = None

        self.title("Import Log Package")
        self.geometry("680x520")
        self.minsize(600, 420)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        body = ttk.Frame(self, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(3, weight=1)

        source_label = preview.source_label.strip() or "(no source label)"
        ttk.Label(
            body,
            text=f"Package from: {source_label}",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.summary_label = ttk.Label(body, justify="left", wraplength=630)
        self.summary_label.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        options = ttk.LabelFrame(body, text="Import mode", padding=(8, 5))
        options.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        options.columnconfigure(0, weight=1)
        self.mode_var = tk.StringVar(value="merge" if self.merge_allowed else "separate")
        self.merge_radio = ttk.Radiobutton(
            options,
            text="Merge new records into this log",
            variable=self.mode_var,
            value="merge",
            command=self._mode_changed,
        )
        self.merge_radio.grid(row=0, column=0, sticky="w")
        self.separate_radio = ttk.Radiobutton(
            options,
            text="Import as a separate snapshot",
            variable=self.mode_var,
            value="separate",
            command=self._mode_changed,
        )
        self.separate_radio.grid(row=1, column=0, sticky="w")
        self.separate_note = ttk.Label(
            options,
            text=(
                "Imports the whole incoming log as a separate copy. A changed log may duplicate "
                "previously imported history."
            ),
            wraplength=590,
            justify="left",
        )
        self.separate_note.grid(row=2, column=0, sticky="w", padx=(20, 0), pady=(2, 0))

        details_frame = ttk.LabelFrame(body, text="Conflicts and validation", padding=6)
        details_frame.grid(row=3, column=0, sticky="nsew")
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        self.details = tk.Text(details_frame, height=8, width=72, wrap="word", state="disabled")
        details_scroll = ttk.Scrollbar(details_frame, orient="vertical", command=self.details.yview)
        self.details.configure(yscrollcommand=details_scroll.set)
        self.details.grid(row=0, column=0, sticky="nsew")
        details_scroll.grid(row=0, column=1, sticky="ns")

        buttons = ttk.Frame(body)
        buttons.grid(row=4, column=0, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.cancel).pack(side="right")
        self.import_button = ttk.Button(buttons, text="Import", command=self.accept)
        self.import_button.pack(side="right", padx=(0, 6))

        if not self.merge_allowed:
            self.merge_radio.state(["disabled"])
            self._mode_changed()
        else:
            self._render_preview()

        self.bind("<Escape>", lambda _event: self.cancel())

    def _mode_changed(self) -> None:
        separate_copy = self.mode_var.get() == "separate"
        try:
            self.preview = self.preview_for_mode(separate_copy)
        except Exception as error:
            messagebox.showerror("Preview import", str(error), parent=self)
            self.mode_var.set("separate" if self.preview.separate_copy else "merge")
            return
        self._render_preview()

    def _render_preview(self) -> None:
        self.summary_label.configure(
            text=(
                f"Incoming: {_format_counts(self.preview.incoming)}\n"
                f"New in this mode: {_format_counts(self.preview.added)}\n"
                f"Already present: {_format_counts(self.preview.skipped)}"
            )
        )
        if self.preview.separate_copy:
            self.separate_note.grid()
        else:
            self.separate_note.grid_remove()
        lines = []
        if self.preview.conflicts:
            lines.append(
                "This mode has blocking conflicts. Nothing will be imported until a conflict-free "
                "preview is available."
            )
            lines.append("")
            for conflict in self.preview.conflicts:
                lines.append(f"{conflict.kind} {conflict.record_id}: {conflict.description}")
        else:
            lines.append("No blocking conflicts in this mode.")
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(lines))
        self.details.configure(state="disabled")
        if self.preview.conflicts:
            self.import_button.state(["disabled"])
        else:
            self.import_button.state(["!disabled"])

    def accept(self) -> None:
        if self.preview.conflicts:
            return
        self.result = self.preview
        self.destroy()

    def cancel(self) -> None:
        self.result = None
        self.destroy()

    def show(self):
        self.grab_set()
        self.wait_window()
        return self.result


class SharingUI:
    """Attach sharing menu actions to a built :class:`TestLogWindow`."""

    def __init__(self, window):
        self.window = window
        self.root = window.root

    def install_menu(self) -> tk.Menu:
        menu = tk.Menu(self.root, tearoff=False)
        sharing = tk.Menu(menu, tearoff=False)
        sharing.add_command(label="Export Log Package…", command=self.export_package)
        sharing.add_command(label="Import Log Package…", command=self.import_package)
        menu.add_cascade(label="Sharing", menu=sharing)
        self.root.configure(menu=menu)
        self.menu = menu
        self.sharing_menu = sharing
        return menu

    def _resolve_unsaved(self) -> bool:
        main_dirty = self.window._is_dirty()
        if not self.window._resolve_unsaved():
            return False
        if main_dirty and self.window._is_dirty():
            self._restore_main_editor()
        manager = self.window.event_manager
        if manager is None:
            return True
        try:
            if not manager.winfo_exists():
                return True
        except tk.TclError:
            return True
        is_dirty = getattr(manager, "_is_dirty", None)
        manager_dirty = bool(is_dirty()) if callable(is_dirty) else False
        if not manager._resolve_unsaved():
            return False
        if manager_dirty and callable(is_dirty) and is_dirty():
            record = getattr(manager, "_record", None)
            if record is not None and not getattr(manager, "_is_new", False):
                manager._load_event(record.id)
            else:
                manager._show_blank()
        return True

    def _restore_main_editor(self) -> None:
        kind = self.window._current_kind
        if kind is None:
            return
        record = self.window._record
        is_new = self.window._is_new
        context_id = self.window._context_id
        if record is not None and not is_new:
            getter = {
                "day": self.window.services.get_day,
                "setup": self.window.services.get_setup,
                "lap": self.window.services.get_lap,
            }.get(kind)
            saved = getter(record.id) if getter is not None else None
            if saved is not None:
                self.window._show_editor(kind, saved, is_new=False)
                self.window.refresh_tree(select_item=f"{kind}:{saved.id}")
                return
        self.window._show_editor(kind, None, is_new=True, context_id=context_id)
        self.window.refresh_tree()

    def export_package(self) -> None:
        if not self._resolve_unsaved():
            return
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Log Package",
            defaultextension=".zip",
            filetypes=(("Vehicle test log packages", "*.zip"), ("All files", "*.*")),
            initialfile="vehicle-test-log.zip",
        )
        if not destination:
            return
        source_label = simpledialog.askstring(
            "Package source label",
            "Optional source label (for example, the driver's name):",
            initialvalue="",
            parent=self.root,
        )
        if source_label is None:
            return
        try:
            written = self.window.services.export_share_package(
                Path(destination),
                source_label=source_label.strip(),
            )
        except Exception as error:
            message = f"Could not export log package: {error}"
            self.window._set_status(message)
            messagebox.showerror("Export Log Package", str(error), parent=self.root)
            return
        self.window._set_status(f"Exported log package: {written}")

    def import_package(self) -> None:
        if not self._resolve_unsaved():
            return
        source = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a Log Package",
            filetypes=(("Vehicle test log packages", "*.zip"), ("All files", "*.*")),
        )
        if not source:
            return
        source_path = Path(source)
        try:
            merge_preview = self.window.services.preview_share_package(
                source_path,
                separate_copy=False,
            )
        except Exception as error:
            self._report_error("Preview Import", "Could not read log package", error)
            return

        dialog = ImportPreviewDialog(
            self.root,
            merge_preview,
            lambda separate_copy: self.window.services.preview_share_package(
                source_path,
                separate_copy=separate_copy,
            ),
        )
        preview = dialog.show()
        if preview is None:
            return
        if preview.separate_copy and not messagebox.askyesno(
            "Import Separate Snapshot",
            "Imports the whole incoming log as a separate copy. A changed log may duplicate "
            "previously imported history. Continue?",
            parent=self.root,
        ):
            return

        try:
            result = self.window.services.import_share_package(
                source_path,
                separate_copy=preview.separate_copy,
                expected_fingerprint=preview.fingerprint,
            )
        except Exception as error:
            self._report_error("Import Log Package", "Could not import log package", error)
            return

        self._refresh_after_import()
        backup = str(result.backup_path) if result.backup_path is not None else "not needed"
        outcome = (
            f"Imported from {result.source_label or 'an unlabeled package'}. "
            f"Added: {_format_counts(result.added)}. "
            f"Already present: {_format_counts(result.skipped)}. Backup: {backup}."
        )
        self.window._set_status(outcome)
        messagebox.showinfo("Log Package Imported", outcome, parent=self.root)

    def _refresh_after_import(self) -> None:
        self.window.refresh_tree()
        manager = self.window.event_manager
        if manager is None:
            return
        try:
            if not manager.winfo_exists():
                return
        except tk.TclError:
            return
        record = getattr(manager, "_record", None)
        record_id = record.id if record is not None and not getattr(manager, "_is_new", False) else None
        manager.refresh_tree(select_id=record_id)
        if record_id is not None:
            manager._load_event(record_id)

    def _report_error(self, title: str, prefix: str, error: Exception) -> None:
        message = f"{prefix}: {error}"
        self.window._set_status(message)
        messagebox.showerror(title, str(error), parent=self.root)
