"""Modern CustomTkinter GUI for Fox Reader Downloader."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk

try:
    from . import downloader, github_client, hardware
except ImportError:
    import downloader, github_client, hardware

# Theme styling
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

BRAND_ORANGE = "#E65100"
BRAND_ORANGE_HOVER = "#F57C00"
CARD_BG = "#212121"
SUCCESS_GREEN = "#2E7D32"


class FoxDownloaderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Fox Reader — Downloader & Installer")
        self.geometry("740x660")
        self.minsize(680, 600)

        # Set icon if available
        self._setup_icon()

        # State
        self.hw_info = hardware.detect_system_hardware()
        self.releases: list[dict[str, Any]] = []
        self.selected_release_data: dict[str, Any] | None = None
        self.selected_variant: dict[str, Any] | None = None
        self.sha256_map: dict[str, str] = {}
        self.cancel_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.is_working = False
        self.msg_queue: queue.Queue[tuple[Any, tuple[Any, ...]]] = queue.Queue()

        # Build UI layout
        self._create_widgets()

        # Start UI event polling and background fetch
        self._poll_ui_queue()
        self.after(100, self._start_fetch_releases)

    def _poll_ui_queue(self):
        try:
            while not self.msg_queue.empty():
                fn, args = self.msg_queue.get_nowait()
                fn(*args)
        except Exception:
            pass
        self.after(40, self._poll_ui_queue)

    def dispatch_ui(self, fn: Any, *args: Any):
        """Thread-safe UI callback dispatch via queue."""
        self.msg_queue.put((fn, args))

    def _setup_icon(self):
        icon_candidates = [
            os.path.join(os.path.dirname(__file__), "favicon.ico"),
            os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static", "favicon.ico"),
        ]
        for ic in icon_candidates:
            if os.path.isfile(ic):
                try:
                    self.iconbitmap(ic)
                    break
                except Exception:
                    pass

    def _create_widgets(self):
        # Main scrollable or padded container
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=24, pady=20)

        # ── Header ──────────────────────────────────────────
        header_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 12))

        title_label = ctk.CTkLabel(
            header_frame,
            text="Fox Reader Installer",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#FFA726",
        )
        title_label.pack(side="left")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Automatic hardware detection & split-file joiner",
            font=ctk.CTkFont(size=12),
            text_color="#9E9E9E",
        )
        subtitle.pack(side="left", padx=(12, 0), pady=(4, 0))

        # ── Hardware Detection Card ─────────────────────────
        hw_card = ctk.CTkFrame(self.container, fg_color=CARD_BG, corner_radius=8)
        hw_card.pack(fill="x", pady=6, ipadx=10, ipady=8)

        hw_title = ctk.CTkLabel(
            hw_card,
            text="Detected System Hardware",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        hw_title.pack(anchor="w", padx=12, pady=(4, 2))

        self.hw_label = ctk.CTkLabel(
            hw_card,
            text=f"OS: {self.hw_info['os'].capitalize()} ({self.hw_info['arch']})  •  {self.hw_info['details']}",
            font=ctk.CTkFont(size=12),
            text_color="#B0BEC5",
            anchor="w",
            justify="left",
        )
        self.hw_label.pack(anchor="w", padx=12, pady=(0, 4))

        # ── Release & Variant Configuration Card ────────────
        config_card = ctk.CTkFrame(self.container, fg_color=CARD_BG, corner_radius=8)
        config_card.pack(fill="x", pady=6, ipadx=10, ipady=10)

        # Release row
        rel_row = ctk.CTkFrame(config_card, fg_color="transparent")
        rel_row.pack(fill="x", padx=12, pady=4)

        rel_label = ctk.CTkLabel(rel_row, text="Release Version:", font=ctk.CTkFont(size=13), width=120, anchor="w")
        rel_label.pack(side="left")

        self.release_combo = ctk.CTkComboBox(
            rel_row,
            values=["Fetching releases..."],
            command=self._on_release_selected,
            width=200,
        )
        self.release_combo.pack(side="left", padx=(0, 10))

        self.prerelease_var = ctk.BooleanVar(value=False)
        self.prerelease_chk = ctk.CTkCheckBox(
            rel_row,
            text="Include pre-releases",
            variable=self.prerelease_var,
            command=self._start_fetch_releases,
            font=ctk.CTkFont(size=12),
        )
        self.prerelease_chk.pack(side="left")

        # GGUF Preference toggle
        gguf_row = ctk.CTkFrame(config_card, fg_color="transparent")
        gguf_row.pack(fill="x", padx=12, pady=(8, 4))

        self.gguf_var = ctk.BooleanVar(value=True)
        self.gguf_chk = ctk.CTkCheckBox(
            gguf_row,
            text="Include AI Translation backend (GGUF / llama.cpp models)",
            variable=self.gguf_var,
            command=self._on_gguf_toggled,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FFB74D",
        )
        self.gguf_chk.pack(side="left")

        # Variant selection row
        var_row = ctk.CTkFrame(config_card, fg_color="transparent")
        var_row.pack(fill="x", padx=12, pady=4)

        var_label = ctk.CTkLabel(var_row, text="Selected Variant:", font=ctk.CTkFont(size=13), width=120, anchor="w")
        var_label.pack(side="left")

        self.variant_combo = ctk.CTkComboBox(
            var_row,
            values=["No variants loaded"],
            command=self._on_variant_selected,
            width=360,
        )
        self.variant_combo.pack(side="left", padx=(0, 10))

        # Recommendation badge
        self.rec_badge = ctk.CTkLabel(
            config_card,
            text="★ Analyzing recommendation...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#81C784",
        )
        self.rec_badge.pack(anchor="w", padx=12, pady=(2, 4))

        # ── Destination Folder Card ─────────────────────────
        dest_card = ctk.CTkFrame(self.container, fg_color=CARD_BG, corner_radius=8)
        dest_card.pack(fill="x", pady=6, ipadx=10, ipady=10)

        dest_title = ctk.CTkLabel(dest_card, text="Installation Location (Portable)", font=ctk.CTkFont(size=13, weight="bold"))
        dest_title.pack(anchor="w", padx=12, pady=(2, 4))

        folder_row = ctk.CTkFrame(dest_card, fg_color="transparent")
        folder_row.pack(fill="x", padx=12, pady=2)

        # Default: ./FoxReader alongside the app
        default_dir = os.path.abspath(os.path.join(os.getcwd(), "FoxReader"))
        self.dest_entry = ctk.CTkEntry(folder_row, font=ctk.CTkFont(size=12))
        self.dest_entry.insert(0, default_dir)
        self.dest_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.browse_btn = ctk.CTkButton(
            folder_row,
            text="Browse...",
            width=90,
            command=self._browse_folder,
        )
        self.browse_btn.pack(side="right")

        # Options row
        opt_row = ctk.CTkFrame(dest_card, fg_color="transparent")
        opt_row.pack(fill="x", padx=12, pady=(8, 0))

        self.cleanup_var = ctk.BooleanVar(value=True)
        self.cleanup_chk = ctk.CTkCheckBox(
            opt_row,
            text="Delete downloaded archive files after successful extraction (recommended to free space)",
            variable=self.cleanup_var,
            font=ctk.CTkFont(size=12),
        )
        self.cleanup_chk.pack(side="left")

        # ── Progress & Action Section ───────────────────────
        self.progress_frame = ctk.CTkFrame(self.container, fg_color=CARD_BG, corner_radius=8)
        self.progress_frame.pack(fill="x", pady=6, ipadx=10, ipady=10)

        self.status_label = ctk.CTkLabel(
            self.progress_frame,
            text="Ready to install.",
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=12, pady=(4, 2))

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal", mode="determinate")
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=12, pady=6)

        self.speed_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#9E9E9E",
            anchor="w",
        )
        self.speed_label.pack(fill="x", padx=12, pady=(0, 4))

        # Bottom Buttons
        btn_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(10, 0))

        self.install_btn = ctk.CTkButton(
            btn_frame,
            text="Download & Install",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=BRAND_ORANGE,
            hover_color=BRAND_ORANGE_HOVER,
            height=40,
            command=self._start_install,
        )
        self.install_btn.pack(side="right", padx=(10, 0))

        self.cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            fg_color="#424242",
            hover_color="#616161",
            height=40,
            command=self._on_cancel,
            state="disabled",
        )
        self.cancel_btn.pack(side="right")

    # ── Folder Selection ────────────────────────────────────
    def _browse_folder(self):
        chosen = filedialog.askdirectory(initialdir=self.dest_entry.get() or os.getcwd())
        if chosen:
            self.dest_entry.delete(0, "end")
            self.dest_entry.insert(0, chosen)

    # ── GitHub Releases Fetching ────────────────────────────
    def _start_fetch_releases(self):
        self.release_combo.configure(values=["Fetching..."])
        self.status_label.configure(text="Fetching available releases from GitHub...")
        include_pre = self.prerelease_var.get()
        threading.Thread(target=self._fetch_releases_worker, args=(include_pre,), daemon=True).start()

    def _fetch_releases_worker(self, include_pre: bool):
        try:
            releases = github_client.fetch_releases(include_prereleases=include_pre)
            self.dispatch_ui(self._on_releases_fetched, releases)
        except Exception as err:
            self.dispatch_ui(self._on_releases_error, str(err))

    def _on_releases_fetched(self, releases: list[dict[str, Any]]):
        self.releases = releases
        if not releases:
            self.release_combo.configure(values=["No releases found"])
            self.status_label.configure(text="No releases found on GitHub.")
            return

        tags = [r.get("tag_name", "unknown") for r in releases]
        self.release_combo.configure(values=tags)
        self.release_combo.set(tags[0])
        self._on_release_selected(tags[0])
        self.status_label.configure(text="Ready to install.")

    def _on_releases_error(self, err_msg: str):
        self.status_label.configure(text=f"Error fetching releases: {err_msg}", text_color="#E57373")
        messagebox.showerror("GitHub Connection Error", f"Could not fetch releases from GitHub:\n{err_msg}")

    def _on_release_selected(self, tag: str):
        # Find release data
        rel = next((r for r in self.releases if r.get("tag_name") == tag), None)
        if not rel:
            return

        parsed = github_client.parse_release_assets(rel)
        self.selected_release_data = parsed

        # Update variants combo
        self._populate_variants()

    def _on_gguf_toggled(self):
        self._populate_variants()

    def _populate_variants(self):
        if not self.selected_release_data:
            return

        variants = self.selected_release_data.get("variants", [])
        os_name = self.hw_info.get("os", "windows")
        # Filter for user's OS
        matching_os = [v for v in variants if v["os"] == os_name]

        if not matching_os:
            self.variant_combo.configure(values=["No builds for this OS in release"])
            self.rec_badge.configure(text="No builds matching your OS found in this release.", text_color="#E57373")
            return

        # Find best recommendation
        rec = github_client.match_recommended_variant(
            matching_os, self.hw_info, prefer_gguf=self.gguf_var.get()
        )
        self.selected_variant = rec

        # Display names for dropdown
        names = []
        for v in matching_os:
            gguf_tag = " + GGUF" if v["has_gguf"] else ""
            split_tag = f" ({len(v['parts'])} parts)" if v["is_split"] else ""
            size_gb = v["total_size"] / (1024**3)
            label = f"{v['device'].upper()}{gguf_tag}{split_tag} ~ {size_gb:.2f} GB"
            names.append(label)

        self.variant_combo.configure(values=names)

        # Set selection to recommended
        if rec:
            rec_idx = matching_os.index(rec)
            self.variant_combo.set(names[rec_idx])
            device_desc = f"{rec['device'].upper()} {'(AI Translation GGUF)' if rec['has_gguf'] else '(No GGUF)'}"
            self.rec_badge.configure(
                text=f"★ Recommended for your machine: {device_desc}",
                text_color="#81C784",
            )

    def _on_variant_selected(self, chosen_label: str):
        if not self.selected_release_data:
            return
        os_name = self.hw_info.get("os", "windows")
        matching_os = [v for v in self.selected_release_data.get("variants", []) if v["os"] == os_name]

        for idx, v in enumerate(matching_os):
            gguf_tag = " + GGUF" if v["has_gguf"] else ""
            split_tag = f" ({len(v['parts'])} parts)" if v["is_split"] else ""
            size_gb = v["total_size"] / (1024**3)
            label = f"{v['device'].upper()}{gguf_tag}{split_tag} ~ {size_gb:.2f} GB"
            if label == chosen_label:
                self.selected_variant = v
                break

    # ── Installation Workflow ───────────────────────────────
    def _start_install(self):
        if not self.selected_variant:
            messagebox.showwarning("No Variant Selected", "Please select a build variant to install.")
            return

        dest_dir = self.dest_entry.get().strip()
        if not dest_dir:
            messagebox.showwarning("Missing Location", "Please select an installation folder.")
            return

        # Pre-flight check: write permissions
        if not downloader.check_write_permission(dest_dir):
            messagebox.showerror(
                "Write Permission Denied",
                f"Cannot write to destination folder:\n{dest_dir}\n\nPlease pick another directory."
            )
            return

        self.cancel_event.clear()
        self.is_working = True
        self.install_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.release_combo.configure(state="disabled")
        self.variant_combo.configure(state="disabled")
        self.browse_btn.configure(state="disabled")

        self.worker_thread = threading.Thread(
            target=self._install_worker,
            args=(self.selected_variant, dest_dir),
            daemon=True,
        )
        self.worker_thread.start()

    def _on_cancel(self):
        if self.is_working:
            if messagebox.askyesno("Cancel Installation", "Are you sure you want to cancel the download?"):
                self.cancel_event.set()
                self.status_label.configure(text="Cancelling operation...")
                self.cancel_btn.configure(state="disabled")

    def _install_worker(self, variant: dict[str, Any], dest_dir: str):
        temp_dir = os.path.join(dest_dir, ".download_cache")
        created_files = []
        try:
            os.makedirs(temp_dir, exist_ok=True)
            created_files.append(temp_dir)

            # Step 1: Fetch SHA256SUMS.txt if available
            sha256_url = self.selected_release_data.get("sha256sums_url")
            checksums = {}
            if sha256_url:
                self._update_status("Fetching checksum manifest (SHA256SUMS.txt)...", 0.05)
                try:
                    checksums = github_client.fetch_sha256sums(sha256_url)
                except Exception:
                    pass

            # Step 2: Download parts
            parts = variant.get("parts", [])
            num_parts = len(parts)
            part_files = []

            for idx, part in enumerate(parts, 1):
                part_name = part["filename"]
                part_url = part["url"]
                part_path = os.path.join(temp_dir, part_name)
                part_files.append(part_path)
                created_files.append(part_path)

                def part_progress(downloaded, total, speed):
                    if total:
                        pct = downloaded / total
                        overall_pct = (idx - 1 + pct) / (num_parts + 2)
                    else:
                        overall_pct = 0.1
                    speed_mb = speed / (1024**2)
                    dl_mb = downloaded / (1024**2)
                    tot_mb = total / (1024**2) if total else 0
                    msg = f"Downloading part {idx}/{num_parts}: {dl_mb:.0f} MB / {tot_mb:.0f} MB"
                    if speed > 0 and total > downloaded:
                        eta_s = int((total - downloaded) / speed)
                        m, s = divmod(eta_s, 60)
                        eta_str = f"  •  ETA: {m}m {s:02d}s" if m > 0 else f"  •  ETA: {s}s"
                    else:
                        eta_str = ""
                    sp_msg = f"{speed_mb:.1f} MB/s{eta_str}"
                    self._update_status(msg, overall_pct, sp_msg)

                downloader.download_file(
                    part_url,
                    part_path,
                    expected_size=part["size"],
                    progress_callback=part_progress,
                    cancel_event=self.cancel_event,
                )

                # Verify part checksum if listed
                if part_name in checksums:
                    self._update_status(f"Verifying checksum for {part_name}...", -1)
                    if not downloader.verify_sha256(part_path, checksums[part_name]):
                        raise RuntimeError(f"Checksum mismatch on {part_name}! Download may be corrupted.")

            # Step 3: Rejoin split parts if needed
            base_archive_name = variant["base_name"]
            joined_archive_path = os.path.join(temp_dir, base_archive_name)
            created_files.append(joined_archive_path)

            if variant["is_split"]:
                self._update_status("Rejoining split parts into unified archive...", 0.75)

                def join_progress(written, total):
                    pct = 0.75 + (written / total) * 0.1
                    self._update_status("Rejoining split parts...", pct)

                downloader.rejoin_split_parts(
                    part_files,
                    joined_archive_path,
                    progress_callback=join_progress,
                    cancel_event=self.cancel_event,
                )

                # Check rejoined checksum if present
                if base_archive_name in checksums:
                    self._update_status("Verifying SHA256 of rejoined archive...", -1)
                    if not downloader.verify_sha256(joined_archive_path, checksums[base_archive_name]):
                        raise RuntimeError(f"Checksum mismatch on rejoined archive {base_archive_name}!")
            else:
                joined_archive_path = part_files[0]

            # Step 4: Extract
            self._update_status("Extracting archive into destination folder...", 0.85)

            def extract_progress(cur, total, name):
                pct = 0.85 + (cur / total) * 0.15
                self._update_status(f"Extracting: {name} ({cur}/{total})", pct)

            downloader.extract_archive(
                joined_archive_path,
                dest_dir,
                progress_callback=extract_progress,
                cancel_event=self.cancel_event,
            )

            # Step 5: Clean up temporary files if opted
            if self.cleanup_var.get():
                self._update_status("Cleaning up temporary downloaded files...", 0.99)
                downloader.cleanup_temp_files([temp_dir])

            # Finished!
            self.dispatch_ui(self._on_install_completed, dest_dir)

        except InterruptedError:
            self._update_status("Installation cancelled by user.", 0.0)
            downloader.cleanup_temp_files([temp_dir])
            self.dispatch_ui(self._reset_ui)
        except Exception as err:
            self._update_status(f"Error: {err}", 0.0)
            self.dispatch_ui(self._on_install_error, str(err))

    def _update_status(self, text: str, progress: float, speed_text: str = ""):
        def update():
            self.status_label.configure(text=text)
            if progress >= 0:
                self.progress_bar.set(progress)
            self.speed_label.configure(text=speed_text)
        self.dispatch_ui(update)

    def _on_install_completed(self, dest_dir: str):
        self.progress_bar.set(1.0)
        self.status_label.configure(
            text="Installation complete!",
            text_color="#81C784",
        )
        self.speed_label.configure(text="")
        self._reset_ui()

        # Display completion message with exact launcher path
        launcher_name = "launcher.exe" if sys.platform.startswith("win") else "launcher"
        launcher_path = os.path.join(dest_dir, launcher_name)
        if not os.path.isfile(launcher_path):
            sub_launcher = os.path.join(dest_dir, "fox-reader", launcher_name)
            if os.path.isfile(sub_launcher):
                launcher_path = sub_launcher
                dest_dir = os.path.join(dest_dir, "fox-reader")

        msg = (
            f"Fox Reader was successfully downloaded and extracted!\n\n"
            f"Installation Location:\n{dest_dir}\n\n"
            f"To start the app, simply run:\n{launcher_path}"
        )
        messagebox.showinfo("Installation Complete", msg)

    def _on_install_error(self, err_msg: str):
        self._reset_ui()
        messagebox.showerror("Installation Failed", f"An error occurred during installation:\n\n{err_msg}")

    def _reset_ui(self):
        self.is_working = False
        self.install_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.release_combo.configure(state="normal")
        self.variant_combo.configure(state="normal")
        self.browse_btn.configure(state="normal")


def main():
    app = FoxDownloaderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
