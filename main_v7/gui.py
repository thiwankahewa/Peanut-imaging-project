#!/usr/bin/env python3
"""CustomTkinter screens. Run with `python -m main_v7` or `python main_v7/gui.py`."""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
from queue import Empty
import sys
import tkinter as tk
import time
import threading

# Direct execution works from any directory, as does `python -m main_v7`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import customtkinter as ctk
from PIL import Image

from developer_access import DeveloperPasswordStore
from main_v7 import backend
from main_v7.scan_clock import system_time_synchronized


BG = "#F2F4EF"
CARD = "#FFFFFF"
INK = "#203B30"
MUTED = "#607268"
GREEN = "#236B49"
HOVER = "#185238"
PALE = "#E5EFE6"
LINE = "#DCE4DA"
TAB_BORDER = "#9CAF9F"
AMBER = "#8C581B"
FONT = "DejaVu Sans"


def label(parent, text="", size=13, bold=False, color=INK, **kwargs):
    return ctk.CTkLabel(parent, text=text, text_color=color, height=20,
                        font=(FONT, size, "bold" if bold else "normal"), **kwargs)


def button(parent, text, command, secondary=False, **kwargs):
    return ctk.CTkButton(parent, text=text, command=command, height=40,
                         corner_radius=10, font=(FONT, 13, "bold"),
                         fg_color=PALE if secondary else GREEN,
                         hover_color=LINE if secondary else HOVER,
                         text_color=INK if secondary else "white", **kwargs)


def card(parent, **kwargs):
    return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16, **kwargs)


class HoldButton(ctk.CTkButton):
    """Activate once after a continuous press; release, leaving, or hiding cancels."""

    HOLD_SECONDS = 3.0

    def __init__(self, parent, text, command, **kwargs):
        self.idle_text = f"{text}\nHold for 3 seconds"
        self.action_text = text
        self.hold_command = command
        self.hold_job = None
        self.pressed_at = None
        self.pressed = False
        super().__init__(parent, text=self.idle_text, command=None, height=48,
                         corner_radius=10, font=(FONT, 13, "bold"), **kwargs)
        self.bind("<ButtonPress-1>", self.start_hold, add="+")
        self.bind("<ButtonRelease-1>", self.cancel_hold, add="+")
        self.bind("<B1-Motion>", self.check_pointer, add="+")
        self.bind("<Leave>", self.check_pointer, add="+")
        # CTkButton.bind targets its contents; Unmap must watch the outer frame.
        tk.Frame.bind(self, "<Unmap>", self.cancel_hold, add="+")

    def start_hold(self, event=None):
        if self.cget("state") == "disabled" or self.pressed:
            return
        self.pressed = True
        self.pressed_at = time.monotonic()
        self.update_hold()

    def update_hold(self):
        self.hold_job = None
        if not self.pressed or self.pressed_at is None:
            return
        if self.cget("state") == "disabled" or not self.winfo_ismapped():
            self.cancel_hold()
            return
        remaining = self.HOLD_SECONDS - (time.monotonic() - self.pressed_at)
        if remaining <= 0:
            self.pressed_at = None
            self.configure(text=self.idle_text)
            self.hold_command()
        else:
            self.configure(text=f"{self.action_text}\nKeep holding · {remaining:.1f} s")
            self.hold_job = self.after(50, self.update_hold)

    def check_pointer(self, event):
        if not (self.winfo_rootx() <= event.x_root < self.winfo_rootx() + self.winfo_width()
                and self.winfo_rooty() <= event.y_root < self.winfo_rooty() + self.winfo_height()):
            self.cancel_hold()

    def cancel_hold(self, event=None):
        if self.hold_job is not None:
            self.after_cancel(self.hold_job)
            self.hold_job = None
        self.pressed = False
        self.pressed_at = None
        self.configure(text=self.idle_text)

    def destroy(self):
        self.cancel_hold()
        super().destroy()


class Dialog(ctk.CTkToplevel):
    """Themed confirmations and password entry, confined to the GUI thread."""

    def __init__(self, parent, title, text, confirm="OK", cancel=False, password=False, number_input=False):
        super().__init__(parent)
        self.withdraw()
        self.title(title)
        self.configure(fg_color=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.result = None
        self.entry = None
        self.keypad_buttons = {}
        width = min(490, parent.winfo_width() - 30)
        self.grid_columnconfigure(0, weight=1)
        label(self, title, 21, True, anchor="w").grid(row=0, column=0, sticky="ew", padx=24, pady=(14, 6))
        message = ctk.CTkTextbox(self, height=48 if password or number_input else 100, fg_color=BG, text_color=MUTED,
                                 font=(FONT, 13), wrap="word", border_width=0)
        message.grid(row=1, column=0, sticky="ew", padx=20)
        message.insert("1.0", text)
        message.configure(state="disabled")
        if password or number_input:
            self.entry = ctk.CTkEntry(self, height=42, show="•" if password else "", font=(FONT, 16),
                                     fg_color=CARD, border_color=LINE, text_color=INK)
            self.entry.grid(row=2, column=0, sticky="ew", padx=24, pady=(8, 0))
            self.entry.bind("<Return>", lambda event: self.finish(True))
            keypad = ctk.CTkFrame(self, fg_color="transparent")
            keypad.grid(row=3, column=0, sticky="ew", padx=21, pady=(8, 0))
            keypad.grid_columnconfigure((0, 1, 2), weight=1, uniform="keys")
            for index, key in enumerate(("1", "2", "3", "4", "5", "6", "7", "8", "9", "Clear", "0", "⌫")):
                item = button(keypad, key, lambda value=key: self.enter_key(value), True, width=70)
                item.configure(height=44, font=(FONT, 20 if key.isdigit() else 15, "bold"))
                item.grid(row=index // 3, column=index % 3, sticky="ew", padx=3, pady=3)
                self.keypad_buttons[key] = item
        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.grid(row=4, column=0, sticky="ew", padx=24, pady=(12, 14))
        controls.grid_columnconfigure((0, 1), weight=1)
        if cancel:
            button(controls, "Cancel", lambda: self.finish(False), secondary=True).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        button(controls, confirm, lambda: self.finish(True)).grid(row=0, column=1, sticky="ew")
        self.protocol("WM_DELETE_WINDOW", lambda: self.finish(False))
        self.bind("<Escape>", lambda event: self.finish(False))
        self.update_idletasks()
        height = self.winfo_reqheight()
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.deiconify()
        self.wait_visibility()
        self.grab_set()
        if self.entry:
            self.entry.focus_set()
        self.wait_window()

    def enter_key(self, key):
        if key == "Clear":
            self.entry.delete(0, "end")
        elif key == "⌫":
            if self.entry.get():
                self.entry.delete(len(self.entry.get()) - 1, "end")
        else:
            self.entry.insert("end", key)
        self.entry.focus_set()

    def finish(self, accepted):
        self.result = self.entry.get() if accepted and self.entry else (True if accepted else None)
        self.grab_release()
        self.destroy()


class ImageBrowser(ctk.CTkFrame):
    def __init__(self, parent, analysis=True):
        super().__init__(parent, fg_color="transparent")
        self.analysis = analysis
        self.scans = []
        self.files = []
        self.selected = None
        self.page = 0
        self.photo = None
        self.original = None
        self.resize_job = None
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        left = card(self, width=246)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        title = "Saved analysis" if analysis else "Captured images"
        label(left, title, 16, True, anchor="w").grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        self.items = ctk.CTkScrollableFrame(left, fg_color=CARD, corner_radius=0, width=210)
        self.items.grid(row=1, column=0, sticky="nsew", padx=6)
        self.items.grid_columnconfigure(0, weight=1)
        pager = ctk.CTkFrame(left, fg_color="transparent")
        pager.grid(row=2, column=0, sticky="ew", padx=12, pady=8)
        pager.grid_columnconfigure(1, weight=1)
        self.previous = button(pager, "‹", lambda: self.change_page(-1), True, width=36)
        self.previous.grid(row=0, column=0)
        self.page_label = label(pager, "")
        self.page_label.grid(row=0, column=1)
        self.next = button(pager, "›", lambda: self.change_page(1), True, width=36)
        self.next.grid(row=0, column=2)
        button(left, "Refresh", self.refresh, True).grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        right = card(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        self.viewport = ctk.CTkFrame(right, fg_color=CARD, corner_radius=12)
        self.viewport.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.preview = label(self.viewport, "Your saved images will appear here", color=MUTED, wraplength=340)
        self.preview.place(relx=0.5, rely=0.5, anchor="center")
        self.viewport.bind("<Configure>", self.schedule_resize)
        self.caption = label(right, "", 11, color=MUTED, justify="left", anchor="w", wraplength=440)
        self.caption.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))

    def refresh(self, preferred=None):
        try:
            self.scans = backend.list_saved_scans(self.analysis)
        except OSError as exc:
            self.caption.configure(text=f"Could not read saved images: {exc}")
            return
        target = Path(preferred) if preferred else self.selected
        self.page = next((index for index, scan in enumerate(self.scans) if target in scan.images), 0)
        self.selected = target
        self.render_list()
        if self.selected:
            self.select(self.selected)
        else:
            self.original = self.photo = None
            self.preview.configure(image=None, text="No complete analysis scans" if self.analysis else "No captured images yet")
            self.caption.configure(text="Analysis scans appear when all six output images are saved." if self.analysis else "Captured images are saved after each scan.")

    def change_page(self, delta):
        page = max(0, min(self.page + delta, len(self.scans) - 1))
        if page == self.page:
            return
        self.page = page
        self.selected = None
        self.render_list()
        if self.selected:
            self.select(self.selected)

    def render_list(self):
        for widget in self.items.winfo_children():
            widget.destroy()
        self.item_buttons = {}
        pages = len(self.scans)
        self.page = max(0, min(self.page, pages - 1))
        self.files = list(self.scans[self.page].images) if self.scans else []
        if self.selected not in self.files:
            self.selected = self.files[0] if self.files else None
        for row, path in enumerate(self.files):
            title = backend.saved_image_title(path, self.analysis)
            capture = path.parent.name if self.analysis else path.stem
            text = f"{title[:25]}\n{capture[:25]}"
            item = ctk.CTkButton(self.items, text=text, anchor="w", height=52, corner_radius=8,
                                 font=(FONT, 11), fg_color=PALE if path == self.selected else CARD,
                                 text_color=INK, hover_color=PALE, command=lambda p=path: self.select(p))
            item.grid(row=row, column=0, sticky="ew", pady=3)
            self.item_buttons[path] = item
        self.page_label.configure(text=f"Scan {self.page + 1 if pages else 0} / {pages}")
        self.previous.configure(state="normal" if self.page else "disabled")
        self.next.configure(state="normal" if self.page + 1 < pages else "disabled")

    def select(self, path):
        self.selected = path
        for item_path, item in self.item_buttons.items():
            item.configure(fg_color=PALE if item_path == path else CARD)
        try:
            with Image.open(path, formats=("PNG",)) as image:
                if image.mode in ("I;16", "I"):
                    image = image.point(lambda value: value / 256).convert("L")
                self.original = image.convert("RGB")
            details = path.name
            if self.analysis:
                summary = backend.read_saved_summary(path)
                dud = summary.get("days_left")
                if dud is None:
                    dud = summary.get("decision", {}).get("mpb_derived_dud_at_q95_maturity")
                estimate = f"{dud:.1f} days" if dud is not None else "unavailable"
                details = f"{path.parent.name}\nEstimated digging: {estimate}"
                if self.winfo_toplevel().developer_mode:
                    details += f" · {summary.get('model_name', 'Legacy output')}"
            self.caption.configure(text=details)
            self.resize_preview()
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.original = self.photo = None
            self.preview.configure(image=None, text="Preview unavailable")
            self.caption.configure(text=str(exc))

    def schedule_resize(self, event=None):
        if self.resize_job:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(100, self.resize_preview)

    def resize_preview(self):
        self.resize_job = None
        width = max(30, self.viewport.winfo_width() - 8)
        height = max(30, self.viewport.winfo_height() - 8)
        self.caption.configure(wraplength=width)
        if self.original is None:
            return
        ratio = min(width / self.original.width, height / self.original.height)
        size = (max(1, int(self.original.width * ratio)), max(1, int(self.original.height * ratio)))
        self.photo = ctk.CTkImage(light_image=self.original, dark_image=self.original, size=size)
        self.preview.configure(image=self.photo, text="")


class PeanutApp(ctk.CTk):
    def __init__(self, *, service=None, auto_connect=True):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")
        super().__init__()
        self.title("Peanut Imaging · v7")
        self.geometry("800x480")
        self.minsize(800, 480)
        self.configure(fg_color=BG)
        self.attributes("-fullscreen", True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.backend = service or backend.ImagingBackend()
        self.developer_mode = False
        self.password_store = DeveloperPasswordStore(backend.PROJECT_DIR / "config" / "developer_password.json")
        self.closing = False
        self.authorizing = False
        self.current_page = "Scan"
        self.device_state = self.backend.state()
        self.latest_result = None
        self.clock_ready = False
        self.clock_prompting = False
        self.clock_checking = False
        self.clock_deadline = time.monotonic() + 30
        self.status = tk.StringVar(value="Starting the imaging system…")
        self.selected_model = tk.StringVar(value="")
        self.collect_data = tk.BooleanVar(value=False)
        self.color = tk.StringVar(value="")
        self.cultivar = tk.StringVar(value="")
        self.result_values = {}
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._navigation()
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 8))
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        self.pages = {}
        self._home()
        self.pages["Analysis"] = ImageBrowser(self.content)
        self.pages["Gallery"] = ImageBrowser(self.content, analysis=False)
        self._settings()
        self._developer()
        for frame in self.pages.values():
            frame.grid(row=0, column=0, sticky="nsew")
        self._footer()
        self.bind("<FocusOut>", self.cancel_power_holds, add="+")
        self.refresh_models()
        self.apply_mode()
        self.show_page("Scan")
        self.apply_state()
        self.poll_job = self.after(60, self.poll_events)
        if auto_connect:
            self.after(150, self.reconnect)
        self.after(200, self.check_clock)

    def _navigation(self):
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 12))
        toolbar.grid_columnconfigure(1, weight=1)
        nav = ctk.CTkFrame(toolbar, fg_color="transparent")
        nav.grid(row=0, column=0, sticky="w")
        self.nav_buttons = {}
        tabs = (("Scan", 62), ("Analysis", 82), ("Gallery", 74), ("Settings", 82), ("Developer", 98))
        for column, (title, width) in enumerate(tabs):
            item = button(nav, title, lambda name=title: self.show_page(name), True,
                          width=width, border_width=2, border_color=TAB_BORDER)
            item.grid(row=0, column=column, padx=(0, 6))
            self.nav_buttons[title] = item
        self.connection = label(toolbar, "Connecting", 12, True, color=MUTED, width=116,
                                 corner_radius=12, fg_color=PALE)
        self.connection.configure(height=40)
        self.connection.grid(row=0, column=2, padx=(12, 8))
        self.mode_button = button(toolbar, "Developer mode", self.toggle_mode, True, width=148)
        self.mode_button.grid(row=0, column=3)

    def _home(self):
        frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.pages["Scan"] = frame
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=4, uniform="home")
        frame.grid_columnconfigure(1, weight=6, uniform="home")
        scan = card(frame)
        scan.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        scan.grid_columnconfigure(0, weight=1)
        scan.grid_rowconfigure(3, weight=1)
        label(scan, "NEW SAMPLE", 11, True, GREEN, anchor="w").grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 6))
        label(scan, "Check \npeanut maturity", 25, True, anchor="w", justify="left").grid(row=1, column=0, sticky="ew", padx=20)
        label(scan, "Place your sample in the tray,\nthen start a scan.", 13, color=MUTED,
              justify="left", anchor="w").grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 6))
        self.scan_button = button(scan, "Scan sample  →", self.start_capture)
        self.scan_button.configure(height=56, font=(FONT, 18, "bold"))
        self.scan_button.grid(row=4, column=0, sticky="ew", padx=20, pady=(10, 12))
        self.progress_text = label(scan, "Ready when you are", 11, color=MUTED, anchor="w")
        self.progress_text.grid(row=5, column=0, sticky="ew", padx=20)
        self.progress = ctk.CTkProgressBar(scan, height=6, fg_color=PALE, progress_color=GREEN)
        self.progress.set(0)
        self.progress.grid(row=6, column=0, sticky="ew", padx=20, pady=(6, 18))
        results = card(frame)
        results.grid(row=0, column=1, sticky="nsew")
        results.grid_columnconfigure(0, weight=1)
        results.grid_rowconfigure(4, weight=1)
        top = ctk.CTkFrame(results, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 2))
        top.grid_columnconfigure(0, weight=1)
        label(top, "Latest results", 16, True, anchor="w").grid(row=0, column=0, sticky="ew")
        self.result_badge = label(top, "No sample yet", 10, color=MUTED)
        self.result_badge.grid(row=0, column=1)
        hero = ctk.CTkFrame(results, fg_color="transparent")
        hero.grid(row=1, column=0, sticky="ew", padx=18)
        self.days = label(hero, "—", 46, True, GREEN)
        self.days.pack(side="left")
        label(hero, "estimated days\nuntil digging", 13, color=MUTED, justify="left").pack(side="left", padx=(12, 0))
        stats = ctk.CTkFrame(results, fg_color=BG, corner_radius=12)
        stats.grid(row=2, column=0, sticky="ew", padx=18, pady=6)
        stats.grid_columnconfigure((0, 1), weight=1, uniform="stats")
        for row, entries in enumerate((("count", "Peanuts detected"), ("maturity", "Maturity · mean / std"),
                                        ("classes", "Brown / Black"), ("combined", "Brown + Black"))):
            key, title = entries
            label(stats, title, 12, color=MUTED, anchor="w").grid(row=row, column=0, sticky="ew", padx=(12, 0), pady=6)
            value = label(stats, "—", 15, True, anchor="e")
            value.grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=6)
            self.result_values[key] = value
        self.result_note = label(results, "Scan a sample to see its maturity estimate.", 11, color=MUTED,
                                  anchor="w", justify="left", wraplength=395)
        self.result_note.grid(row=3, column=0, sticky="ew", padx=18, pady=(4, 10))
        self.result_note.bind("<Button-1>", lambda event: self.show_result_notes())
        self.see_result_button = button(results, "See the result", self.show_analysis_result)
        self.see_result_button.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 12))
        self.see_result_button.grid_remove()
        results.bind("<Configure>", lambda event: self.result_note.configure(wraplength=max(200, event.width - 36)))

    def _settings(self):
        frame = ctk.CTkScrollableFrame(self.content, fg_color=CARD, corner_radius=16)
        self.pages["Settings"] = frame
        frame.grid_columnconfigure(0, weight=1)
        self.device_details = label(frame, "Checking cameras…", 13, color=MUTED, anchor="w")
        self.device_details.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))
        actions = ctk.CTkFrame(frame, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=16)
        actions.grid_columnconfigure((0, 1), weight=1)
        self.reconnect_button = button(actions, "Reconnect cameras", self.reconnect, True)
        self.reconnect_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.calibrate_button = button(actions, "Calibrate camera", self.calibrate)
        self.calibrate_button.grid(row=0, column=1, sticky="ew")
        label(frame, "Manual light test", 15, True, anchor="w").grid(row=3, column=0, sticky="ew", padx=16, pady=(18, 6))
        lights = ctk.CTkFrame(frame, fg_color="transparent")
        lights.grid(row=4, column=0, sticky="ew", padx=16)
        lights.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.led_buttons = []
        for index in range(4):
            item = button(lights, f"LED {index + 1}", lambda i=index + 1: self.toggle_led(i), True, width=100)
            item.grid(row=0, column=index, sticky="ew", padx=(0, 8) if index < 3 else 0)
            self.led_buttons.append(item)
        label(frame, "Turn the test light off before returning to Scan or Analysis.", 11, color=MUTED,
              anchor="w").grid(row=5, column=0, sticky="ew", padx=16, pady=(6, 12))
        power = ctk.CTkFrame(frame, fg_color="transparent")
        power.grid(row=6, column=0, sticky="ew", padx=16, pady=(0, 6))
        power.grid_columnconfigure((0, 1, 2), weight=1, uniform="power")
        self.exit_button = button(power, "Exit to desktop", self.on_close, True)
        self.exit_button.configure(height=48)
        self.exit_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.power_buttons = {}
        for column, (action, title) in enumerate((("poweroff", "Shut down"), ("reboot", "Restart")), start=1):
            item = HoldButton(power, title, lambda value=action: self.on_power(value),
                              fg_color=GREEN, hover_color=HOVER, text_color="white", width=160)
            item.grid(row=0, column=column, sticky="ew", padx=(0, 8) if column == 1 else 0)
            self.power_buttons[action] = item
        label(frame, "Hold a power button for 3 seconds. Release to cancel.", 11, color=MUTED,
              anchor="w").grid(row=7, column=0, sticky="ew", padx=16, pady=(0, 12))
        self.time_button = button(frame, "Set date and time", self.enter_scan_time, True)
        self.time_button.grid(row=8, column=0, sticky="ew", padx=16, pady=(0, 12))

    def _developer(self):
        frame = ctk.CTkScrollableFrame(self.content, fg_color=CARD, corner_radius=16)
        self.pages["Developer"] = frame
        frame.grid_columnconfigure(0, weight=1)
        label(frame, "Developer controls", 22, True, anchor="w").grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 6))
        label(frame, "Maturity model", 13, True, anchor="w").grid(row=1, column=0, sticky="ew", padx=16)
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 12))
        row.grid_columnconfigure(0, weight=1)
        self.model_combo = ctk.CTkComboBox(row, variable=self.selected_model, values=[], state="readonly",
                                          height=40, border_color=LINE, button_color=GREEN, button_hover_color=HOVER,
                                          fg_color=BG, text_color=INK, dropdown_fg_color=CARD,
                                          dropdown_text_color=INK, dropdown_hover_color=PALE, font=(FONT, 13))
        self.model_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.refresh_models_button = button(row, "Refresh models", self.refresh_models, True)
        self.refresh_models_button.grid(row=0, column=1)
        self.collection_switch = ctk.CTkSwitch(frame, text="Enable data collection", variable=self.collect_data,
                                               command=self.toggle_collection, progress_color=GREEN, text_color=INK,
                                               font=(FONT, 13))
        self.collection_switch.grid(row=3, column=0, sticky="w", padx=16, pady=6)
        choices = ctk.CTkFrame(frame, fg_color="transparent")
        choices.grid(row=4, column=0, sticky="ew", padx=16, pady=(4, 10))
        choices.grid_columnconfigure((0, 1), weight=1)
        self.collection_choices = []
        for column, (title, variable, values) in enumerate((
            ("Peanut color", self.color, ["black", "brown", "orange", "yellow", "white", "mix"]),
            ("Cultivar", self.cultivar, ["1 (Georgia-09B)", "2 (Georgia-20VH0)", "3 (TifNV-HG)"]),
        )):
            label(choices, title, 11, color=MUTED, anchor="w").grid(row=0, column=column, sticky="ew", padx=(0, 8))
            choice = ctk.CTkComboBox(choices, variable=variable, values=values, height=40, state="disabled",
                                     border_color=LINE, button_color=GREEN, button_hover_color=HOVER,
                                     fg_color=BG, text_color=INK, dropdown_fg_color=CARD,
                                     dropdown_text_color=INK, dropdown_hover_color=PALE)
            choice.grid(row=1, column=column, sticky="ew", padx=(0, 8) if column == 0 else 0)
            self.collection_choices.append(choice)
        self.developer_details = ctk.CTkTextbox(frame, height=94, fg_color=BG, text_color=MUTED, wrap="word", font=(FONT, 11))
        self.developer_details.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 12))
        self.update_developer_details()

    def _footer(self):
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=22, pady=(0, 8))
        footer.grid_columnconfigure(0, weight=1)
        self.status_label = label(footer, textvariable=self.status, size=11, color=MUTED, anchor="w", wraplength=620)
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.mode_label = label(footer, "User mode", 11, color=MUTED, width=116, anchor="e")
        self.mode_label.grid(row=0, column=1, padx=(10, 0))

    def show_page(self, name, preferred=None):
        if name in ("Developer", "Gallery") and not self.developer_mode:
            return
        if self.device_state["active_led"] is not None and name != "Settings":
            return
        self.cancel_power_holds()
        self.current_page = name
        for page, frame in self.pages.items():
            if page == name:
                frame.grid()
            else:
                frame.grid_remove()
        for title, item in self.nav_buttons.items():
            item.configure(fg_color=GREEN if title == name else CARD,
                           text_color="white" if title == name else INK,
                           border_color=GREEN if title == name else TAB_BORDER,
                           hover_color=HOVER if title == name else PALE)
        if name in ("Analysis", "Gallery"):
            self.pages[name].refresh(preferred=preferred)

    def dialog(self, title, text, **kwargs):
        return Dialog(self, title, text, **kwargs).result

    def toggle_mode(self):
        if self.backend.busy or self.closing:
            return
        if self.developer_mode:
            self.developer_mode = False
        elif self.authorize("Developer access", "Enter the developer password."):
            self.developer_mode = True
        self.apply_mode()

    def authorize(self, title, prompt):
        if self.authorizing:
            return False
        self.authorizing = True
        self.cancel_power_holds()
        try:
            configured = self.password_store.is_configured()
            password = self.dialog(title if configured else "Set developer password",
                                   prompt if configured else "Create a password for Developer mode and exiting to desktop. Use digits for touchscreen entry.",
                                   confirm="Continue", cancel=True, password=True)
            if password is None:
                return False
            if not password:
                self.dialog("Password required", "The password cannot be empty.")
                return False
            if configured:
                if not self.password_store.verify(password):
                    self.dialog("Access locked", "Incorrect password. Please try again.")
                    return False
            else:
                confirmation = self.dialog("Confirm password", "Enter the new password again.",
                                           confirm="Save password", cancel=True, password=True)
                if confirmation is None:
                    return False
                if password != confirmation:
                    self.dialog("Passwords do not match", "Please try again.")
                    return False
                self.password_store.set_password(password)
            return True
        except (OSError, ValueError) as exc:
            self.dialog("Password verification unavailable", str(exc))
            return False
        finally:
            self.authorizing = False

    def apply_mode(self):
        self.mode_button.configure(text="User mode" if self.developer_mode else "Developer mode")
        mode = "Developer" if self.developer_mode else "User"
        self.mode_label.configure(text=f"{mode} mode", text_color=MUTED)
        for name in ("Gallery", "Developer"):
            if self.developer_mode:
                self.nav_buttons[name].grid()
            else:
                self.nav_buttons[name].grid_remove()
        if not self.developer_mode:
            self.collect_data.set(False)
            self.toggle_collection()
            if self.current_page in ("Gallery", "Developer"):
                self.show_page("Scan")
        if self.current_page == "Analysis":
            self.pages["Analysis"].refresh()
        self.apply_state()

    def toggle_collection(self):
        if not self.developer_mode:
            self.collect_data.set(False)
        if not self.collect_data.get():
            self.color.set("")
            self.cultivar.set("")
        state = "readonly" if self.collect_data.get() and not self.backend.busy else "disabled"
        for item in self.collection_choices:
            item.configure(state=state)

    def refresh_models(self):
        if self.backend.busy:
            return
        self.model_names = [path.name for path in self.backend.models()]
        self.model_combo.configure(values=self.model_names)
        if self.selected_model.get() not in self.model_names:
            preferred = backend.DEFAULT_MATURITY_MODEL
            self.selected_model.set(preferred if preferred in self.model_names else (self.model_names[0] if self.model_names else ""))
        self.apply_state()

    def apply_state(self):
        busy = self.backend.busy or self.closing
        self.update_result_button()
        enabled = "disabled" if busy else "normal"
        ready = self.device_state["ready"] and self.clock_ready and not busy
        self.scan_button.configure(state="normal" if ready else "disabled", text="Scan sample  →")
        self.mode_button.configure(state=enabled)
        self.reconnect_button.configure(state=enabled)
        self.time_button.configure(state=enabled)
        self.calibrate_button.configure(state="normal" if not busy and self.device_state["flir"] else "disabled")
        self.model_combo.configure(state="readonly" if not busy and getattr(self, "model_names", []) else "disabled")
        self.refresh_models_button.configure(state=enabled)
        self.collection_switch.configure(state=enabled)
        self.exit_button.configure(state="disabled" if self.closing else "normal")
        for item in self.power_buttons.values():
            allowed = not busy and platform.system() == "Linux"
            if not allowed:
                item.cancel_hold()
            item.configure(state="normal" if allowed else "disabled")
        self.toggle_collection()
        for index, item in enumerate(self.led_buttons, start=1):
            active = self.device_state["active_led"] == index
            item.configure(state=enabled, text=f"LED {index}" + (" · ON" if active else ""),
                           fg_color=GREEN if active else PALE, text_color="white" if active else INK)
        for title, item in self.nav_buttons.items():
            locked = self.closing or (self.device_state["active_led"] is not None and title != "Settings")
            item.configure(state="disabled" if locked else "normal")
        if self.device_state["active_led"] is not None and self.current_page != "Settings":
            self.show_page("Settings")
        connected = self.device_state["flir"] and self.device_state["usb"]
        badge = "Working…" if busy else ("Ready" if ready else "Needs attention")
        self.connection.configure(text=badge, text_color=GREEN if ready else MUTED)
        self.device_details.configure(text=(f"FLIR: {'connected' if self.device_state['flir'] else 'offline'}    ·    "
                                            f"USB: {'connected' if self.device_state['usb'] else 'offline'}    ·    "
                                            f"Calibration: {'ready' if self.device_state['calibrated'] else 'required'}"))
        if not busy:
            prompt = "Ready when you are" if ready else ("Calibration required" if connected else "Connect cameras in Settings")
            if self.device_state["active_led"] is not None:
                prompt = "Turn off the test light in Settings"
            elif not self.clock_ready:
                prompt = "Checking date and time…" if time.monotonic() < self.clock_deadline else "Set date and time before scanning"
            self.progress_text.configure(text=prompt)

    def reconnect(self):
        if self.backend.connect():
            self.apply_state()

    def calibrate(self):
        if self.backend.busy or self.closing:
            return
        if self.dialog("Prepare calibration board", "Align the calibration board with the peanut tray, then close the lid.",
                       confirm="Start calibration", cancel=True):
            self.progress.set(0)
            self.backend.calibrate()
            self.apply_state()

    def toggle_led(self, index):
        if self.backend.toggle_led(index):
            self.apply_state()

    def start_capture(self):
        if not self.clock_ready:
            if time.monotonic() >= self.clock_deadline:
                self.enter_scan_time()
            return
        if self.backend.busy or self.closing or not self.device_state["ready"]:
            return
        suffix = ""
        if self.developer_mode and self.collect_data.get():
            if not self.color.get() or not self.cultivar.get():
                self.dialog("Complete data collection details", "Choose both peanut color and cultivar in Developer controls before scanning.")
                return
            suffix = f"_{self.color.get()}_{self.cultivar.get()}"
        if not self.dialog("Ready to scan?", "Insert the tray with your peanut sample, then start the scan.", confirm="Start scan", cancel=True):
            return
        model = self.selected_model.get()
        request = backend.CaptureRequest(backend.MODEL_DIR / model if model else None, suffix)
        if self.backend.capture(request):
            self.display_result(None)
            self.status.set("Starting scan…")
            self.progress.set(0)
            self.apply_state()

    def display_result(self, result):
        self.latest_result = result
        self.days.configure(text="—")
        for item in self.result_values.values():
            item.configure(text="—")
        self.result_badge.configure(text="No current result")
        self.result_note.configure(text="Results appear here after a successful analysis.", text_color=MUTED)
        if result:
            days = result.get("days_left")
            self.days.configure(text=f"{days:.1f}" if days is not None else "—")
            self.result_values["count"].configure(text=str(result.get("n_peanuts", 0)))
            mean, std = result.get("mean_maturity"), result.get("std_maturity")
            self.result_values["maturity"].configure(text=f"{mean:.3f} / {std:.3f}" if mean is not None and std is not None else "—")
            brown, black, combined = (result.get(key) for key in ("brown_ratio", "black_ratio", "brown_black_ratio"))
            self.result_values["classes"].configure(text=f"{brown:.0%} / {black:.0%}" if brown is not None and black is not None else "—")
            self.result_values["combined"].configure(text=f"{combined:.1%}" if combined is not None else "—")
            warnings = result.get("warnings", [])
            self.result_badge.configure(text="Latest scan")
            note = (
                "DUD unavailable for this model." if days is None else "MPB-derived estimate at Q95 maturity.")
            if warnings:
                note += " Tap here for quality notes."
            self.result_note.configure(text=note, text_color=AMBER if warnings else MUTED,
                                       cursor="hand2" if warnings else "")
        self.update_developer_details()

        self.update_result_button()

    def update_result_button(self):
        if self.latest_result and not self.backend.busy and not self.closing:
            self.see_result_button.grid()
        else:
            self.see_result_button.grid_remove()

    def show_analysis_result(self):
        if not self.latest_result or self.backend.busy or self.closing:
            return
        directory = self.latest_result.get("result_directory")
        preferred = Path(directory) / "digital_mpb_profile.png" if directory else None
        self.show_page("Analysis", preferred=preferred)

    def show_result_notes(self):
        if self.latest_result and self.latest_result.get("warnings"):
            self.dialog("Analysis notes", "\n\n".join(self.latest_result["warnings"]))

    def update_developer_details(self):
        result = self.latest_result
        text = "No current analysis. The selected model is fixed for the duration of each scan."
        if result:
            text = f"Model: {result.get('model_name', '—')}\n" + "\n".join(result.get("warnings", []))
            if result.get("result_directory"):
                text += f"\nSaved in: {result['result_directory']}"
        self.developer_details.configure(state="normal")
        self.developer_details.delete("1.0", "end")
        self.developer_details.insert("1.0", text)
        self.developer_details.configure(state="disabled")

    def poll_events(self):
        errors = []
        try:
            while True:
                event = self.backend.events.get_nowait()
                if event.kind == "closed":
                    self.destroy()
                    return
                if event.kind == "clock_status" and not self.closing:
                    self.clock_checking = False
                    if event.data:
                        self.backend.scan_clock.use_system_time()
                        self.clock_ready = True
                        self.apply_state()
                    elif not self.clock_ready and time.monotonic() >= self.clock_deadline:
                        self.enter_scan_time()
                    delay = 60000 if self.clock_ready or time.monotonic() >= self.clock_deadline else 5000
                    self.after(delay, self.check_clock)
                elif event.kind == "state":
                    self.device_state = event.data
                    self.apply_state()
                elif event.kind == "status" and not self.closing:
                    self.status.set(event.data)
                elif event.kind == "progress":
                    self.progress.set(event.data)
                    self.progress_text.configure(text=f"{event.data:.0%} complete")
                elif event.kind == "result":
                    self.display_result(event.data)
                elif event.kind == "error" and not self.closing:
                    errors.append(event.data)
                elif event.kind == "close_failed":
                    self.closing = False
                    self.status.set("Power request failed · reconnect cameras to continue scanning")
                    self.apply_state()
                    errors.append(event.data)
                elif event.kind == "done":
                    if event.data == "capture" and self.current_page in ("Analysis", "Gallery"):
                        self.pages[self.current_page].refresh()
        except Empty:
            pass
        for error in errors:
            self.dialog(error["title"], error["message"])
        self.poll_job = self.after(60, self.poll_events)

    def check_clock(self):
        if self.closing or self.clock_checking:
            return
        self.clock_checking = True
        def check():
            self.backend.emit("clock_status", system_time_synchronized())
        threading.Thread(target=check, daemon=True).start()

    def enter_scan_time(self):
        if self.closing or self.clock_prompting:
            return
        if self.authorizing or self.backend.busy or self.grab_current() is not None:
            self.after(1000, self.enter_scan_time)
            return
        self.clock_prompting = True
        self.cancel_power_holds()
        try:
            while not self.closing:
                value = self.dialog("Set date and time",
                                    "Automatic time is unavailable. Enter YYYYMMDDHHMM (24-hour).\nExample: 202610061430 = 6 Oct 2026, 14:30.",
                                    number_input=True, cancel=True, confirm="Set time")
                if value is None:
                    self.status.set("Date and time required before scanning")
                    # Ask again later if the user needs to check the current time.
                    break
                try:
                    self.backend.scan_clock.set_manual(value)
                except ValueError as exc:
                    self.dialog("Check date and time", str(exc))
                    continue
                self.clock_ready = True
                self.status.set("Scan time set · " + self.backend.scan_clock.now().strftime("%Y-%m-%d %H:%M"))
                break
        finally:
            self.clock_prompting = False
            self.apply_state()

    def on_close(self):
        if self.closing:
            return
        if not self.authorize("Exit to desktop", "Enter the developer password to exit to the desktop."):
            return
        if self.backend.busy and not self.dialog("Close the imaging system?", "The current step will finish before the cameras are released. Analysis may take a little longer.",
                                                confirm="Close safely", cancel=True):
            return
        self.closing = True
        self.status.set("Closing safely · waiting for the current step…")
        self.apply_state()
        self.backend.request_close()

    def cancel_power_holds(self, event=None):
        for item in self.power_buttons.values():
            item.cancel_hold()

    def on_power(self, action):
        if self.closing or self.backend.busy or self.authorizing or self.current_page != "Settings":
            return
        self.cancel_power_holds()
        self.closing = True
        self.status.set("Shutting down the Pi…" if action == "poweroff" else "Restarting the Pi…")
        self.apply_state()
        self.backend.request_close(power_action=action)

    def destroy(self):
        # Tk owns the queue, including CustomTkinter's delayed sizing callbacks.
        # Cancel it before deleting widget commands to avoid callbacks after exit.
        for pending in self.tk.call("after", "info"):
            # Leave each widget to delete its own registered Tcl command.
            self.tk.call("after", "cancel", pending)
        super().destroy()


def main():
    parser = argparse.ArgumentParser(description="Peanut Imaging v7 · CustomTkinter")
    parser.parse_args()
    app = PeanutApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.backend.request_close()
        if app.backend.worker:
            app.backend.worker.join()
        app.destroy()


if __name__ == "__main__":
    main()
