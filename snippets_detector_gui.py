import os
import time
import tkinter as tk
from tkinter import filedialog, ttk
import winsound

from PIL import Image, ImageTk

from waterlogging_detection import ModelUnavailableError, predict_waterlogging

SNIPPETS_DIR = "snippets"
REFRESH_INTERVAL_MS = 3000
MAX_HISTORY = 20
MAX_ALERT_LOG = 25
ALERT_CONFIDENCE_DEFAULT = 0.75
ALERT_COOLDOWN_SEC = 12
ANALYSIS_DELAY_SEC = 1.5


class WaterloggingDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Waterlogging Detection Dashboard")
        self.root.geometry("1120x700")
        self.root.minsize(980, 620)
        self.root.configure(bg="#edf2f7")

        self.last_image_path = ""
        self.last_prediction = None
        self.selected_analysis_image = ""
        self.analysis_ready_at = 0.0
        self.start_time = time.time()
        self.waterlogged_count = 0
        self.non_waterlogged_count = 0
        self.alert_enabled = tk.BooleanVar(value=True)
        self.alert_sound_enabled = tk.BooleanVar(value=True)
        self.alert_threshold = tk.DoubleVar(value=ALERT_CONFIDENCE_DEFAULT)
        self.last_alert_time = 0.0
        self.alert_banner_flash = False

        self._build_styles()
        self._build_layout()
        self._schedule_refresh()

    def _build_styles(self) -> None:
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Card.TFrame", background="#ffffff", relief="flat")
        self.style.configure("Title.TLabel", background="#edf2f7", foreground="#1a202c", font=("Segoe UI Semibold", 18))
        self.style.configure("Subtitle.TLabel", background="#edf2f7", foreground="#4a5568", font=("Segoe UI", 10))
        self.style.configure("CardTitle.TLabel", background="#ffffff", foreground="#2d3748", font=("Segoe UI Semibold", 11))
        self.style.configure("Value.TLabel", background="#ffffff", foreground="#1a202c", font=("Segoe UI Semibold", 14))
        self.style.configure("Small.TLabel", background="#ffffff", foreground="#4a5568", font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16, style="Card.TFrame")
        container.pack(fill="both", expand=True)
        container.configure(style="TFrame")

        header = tk.Frame(container, bg="#edf2f7")
        header.pack(fill="x", pady=(0, 12))

        ttk.Label(header, text="Flood Vision Console", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Live frame monitoring from snippets folder with optional analysis image selection",
            style="Subtitle.TLabel",
        ).pack(anchor="w")

        self.alert_banner = tk.Label(
            header,
            text="ALERT STATUS: NORMAL",
            bg="#2f855a",
            fg="white",
            font=("Segoe UI Semibold", 11),
            pady=6,
            padx=10,
        )
        self.alert_banner.pack(anchor="e", pady=(8, 0))

        main = tk.Frame(container, bg="#edf2f7")
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self.preview_card = ttk.Frame(main, style="Card.TFrame", padding=12)
        self.preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ttk.Label(self.preview_card, text="Live Preview", style="CardTitle.TLabel").pack(anchor="w")
        self.preview_canvas = tk.Label(
            self.preview_card,
            text="Waiting for frames...",
            bg="#0f172a",
            fg="#e2e8f0",
            font=("Segoe UI", 14),
            width=72,
            height=24,
        )
        self.preview_canvas.pack(fill="both", expand=True, pady=(8, 8))

        actions = tk.Frame(self.preview_card, bg="#ffffff")
        actions.pack(fill="x")
        tk.Button(actions, text="Refresh Now", command=self.refresh_now, bg="#2b6cb0", fg="white", relief="flat", padx=10).pack(side="left")
        tk.Button(actions, text="Open Snippets Folder", command=self.open_snippets_folder, bg="#2f855a", fg="white", relief="flat", padx=10).pack(side="left", padx=(8, 0))
        tk.Button(actions, text="Select Analysis Image", command=self.select_analysis_image, bg="#dd6b20", fg="white", relief="flat", padx=10).pack(side="left", padx=(8, 0))
        tk.Button(actions, text="Back to Live Feed", command=self.clear_analysis_image, bg="#718096", fg="white", relief="flat", padx=10).pack(side="left", padx=(8, 0))

        side = ttk.Frame(main, style="Card.TFrame", padding=12)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_rowconfigure(5, weight=1)

        ttk.Label(side, text="System Status", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")

        self.model_status = ttk.Label(side, text="Model: Checking...", style="Small.TLabel")
        self.model_status.grid(row=1, column=0, sticky="w", pady=(8, 0))

        self.frame_name = ttk.Label(side, text="Latest frame: -", style="Small.TLabel")
        self.frame_name.grid(row=2, column=0, sticky="w", pady=(4, 0))

        self.last_update = ttk.Label(side, text="Last update: -", style="Small.TLabel")
        self.last_update.grid(row=3, column=0, sticky="w", pady=(4, 0))

        self.verdict = ttk.Label(side, text="Verdict: Waiting", style="Value.TLabel")
        self.verdict.grid(row=4, column=0, sticky="w", pady=(10, 0))

        self.confidence_bar = ttk.Progressbar(side, orient="horizontal", mode="determinate", length=280)
        self.confidence_bar.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        self.confidence_text = ttk.Label(side, text="Confidence: -", style="Small.TLabel")
        self.confidence_text.grid(row=6, column=0, sticky="w", pady=(4, 0))

        self.counters = ttk.Label(side, text="Waterlogged: 0 | Non-Waterlogged: 0", style="Small.TLabel")
        self.counters.grid(row=7, column=0, sticky="w", pady=(8, 0))

        self.uptime = ttk.Label(side, text="Uptime: 0s", style="Small.TLabel")
        self.uptime.grid(row=8, column=0, sticky="w", pady=(4, 0))

        self.mode_text = ttk.Label(side, text="Mode: Live snippets feed", style="Small.TLabel")
        self.mode_text.grid(row=9, column=0, sticky="w", pady=(4, 0))

        self.demo_note = ttk.Label(side, text="Analysis note: -", style="Small.TLabel")
        self.demo_note.grid(row=10, column=0, sticky="w", pady=(4, 0))

        alert_frame = ttk.Frame(side, style="Card.TFrame")
        alert_frame.grid(row=11, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(alert_frame, text="Alert Controls", style="CardTitle.TLabel").pack(anchor="w")

        ttk.Checkbutton(
            alert_frame,
            text="Enable alerts",
            variable=self.alert_enabled,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Checkbutton(
            alert_frame,
            text="Sound alert",
            variable=self.alert_sound_enabled,
        ).pack(anchor="w", pady=(2, 0))

        threshold_row = tk.Frame(alert_frame, bg="#ffffff")
        threshold_row.pack(fill="x", pady=(6, 0))
        ttk.Label(threshold_row, text="Trigger confidence", style="Small.TLabel").pack(side="left")
        self.threshold_value = ttk.Label(
            threshold_row,
            text=f"{self.alert_threshold.get() * 100:.0f}%",
            style="Small.TLabel",
        )
        self.threshold_value.pack(side="right")

        self.threshold_slider = ttk.Scale(
            alert_frame,
            from_=0.50,
            to=0.99,
            variable=self.alert_threshold,
            command=self._on_threshold_change,
        )
        self.threshold_slider.pack(fill="x", pady=(2, 0))

        self.alert_status_text = ttk.Label(
            alert_frame,
            text=f"Cooldown: {ALERT_COOLDOWN_SEC}s",
            style="Small.TLabel",
        )
        self.alert_status_text.pack(anchor="w", pady=(4, 0))

        history_card = ttk.Frame(container, style="Card.TFrame", padding=12)
        history_card.pack(fill="x", pady=(10, 0))
        ttk.Label(history_card, text="Recent Predictions", style="CardTitle.TLabel").pack(anchor="w")

        columns = ("time", "frame", "label", "confidence")
        self.history = ttk.Treeview(history_card, columns=columns, show="headings", height=8)
        for col, width in (("time", 140), ("frame", 360), ("label", 200), ("confidence", 140)):
            self.history.heading(col, text=col.capitalize())
            self.history.column(col, width=width, anchor="w")
        self.history.pack(fill="x", pady=(8, 0))

        alert_log_card = ttk.Frame(container, style="Card.TFrame", padding=12)
        alert_log_card.pack(fill="x", pady=(10, 0))
        ttk.Label(alert_log_card, text="Alert Log", style="CardTitle.TLabel").pack(anchor="w")

        alert_columns = ("time", "frame", "message")
        self.alert_log = ttk.Treeview(alert_log_card, columns=alert_columns, show="headings", height=6)
        for col, width in (("time", 140), ("frame", 260), ("message", 700)):
            self.alert_log.heading(col, text=col.capitalize())
            self.alert_log.column(col, width=width, anchor="w")
        self.alert_log.pack(fill="x", pady=(8, 0))

    def open_snippets_folder(self) -> None:
        os.makedirs(SNIPPETS_DIR, exist_ok=True)
        try:
            os.startfile(os.path.abspath(SNIPPETS_DIR))
        except Exception:
            pass

    def select_analysis_image(self) -> None:
        analysis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyis")
        os.makedirs(analysis_dir, exist_ok=True)
        selected = filedialog.askopenfilename(
            title="Select analysis image",
            initialdir=analysis_dir,
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if not selected:
            return
        self.selected_analysis_image = selected
        self.last_image_path = ""
        self.last_prediction = None
        self.analysis_ready_at = time.time() + ANALYSIS_DELAY_SEC
        self._refresh_dashboard()

    def clear_analysis_image(self) -> None:
        self.selected_analysis_image = ""
        self.last_image_path = ""
        self.last_prediction = None
        self.analysis_ready_at = 0.0
        self.demo_note.configure(text="Analysis note: -")
        self.mode_text.configure(text="Mode: Live snippets feed")
        self._refresh_dashboard()

    def refresh_now(self) -> None:
        self._refresh_dashboard()

    def _on_threshold_change(self, _value) -> None:
        self.threshold_value.configure(text=f"{self.alert_threshold.get() * 100:.0f}%")

    def _push_alert_log(self, frame_name: str, message: str) -> None:
        self.alert_log.insert("", 0, values=(time.strftime("%H:%M:%S"), frame_name, message))
        children = self.alert_log.get_children()
        if len(children) > MAX_ALERT_LOG:
            for item in children[MAX_ALERT_LOG:]:
                self.alert_log.delete(item)

    def _set_alert_banner(self, is_alerting: bool) -> None:
        if is_alerting:
            self.alert_banner_flash = not self.alert_banner_flash
            bg = "#c53030" if self.alert_banner_flash else "#9b2c2c"
            self.alert_banner.configure(text="ALERT STATUS: WATERLOGGING RISK", bg=bg)
            return
        self.alert_banner.configure(text="ALERT STATUS: NORMAL", bg="#2f855a")

    def _play_alert_sound(self) -> None:
        if not self.alert_sound_enabled.get():
            return
        try:
            winsound.Beep(1200, 250)
            winsound.Beep(900, 300)
        except Exception:
            pass

    def _evaluate_alert(self, frame_name: str, label: str, confidence: float) -> None:
        threshold = self.alert_threshold.get()
        self.alert_status_text.configure(text=f"Cooldown: {ALERT_COOLDOWN_SEC}s | Threshold: {threshold * 100:.0f}%")

        should_alert = (
            self.alert_enabled.get()
            and label == "Waterlogged"
            and confidence >= threshold
        )
        self._set_alert_banner(should_alert)

        if not should_alert:
            return

        now = time.time()
        if now - self.last_alert_time < ALERT_COOLDOWN_SEC:
            return

        self.last_alert_time = now
        alert_message = f"Waterlogging detected with {confidence * 100:.2f}% confidence"
        self._push_alert_log(frame_name, alert_message)
        self._play_alert_sound()

    def _get_latest_image(self):
        os.makedirs(SNIPPETS_DIR, exist_ok=True)
        jpg_files = [f for f in os.listdir(SNIPPETS_DIR) if f.lower().endswith(".jpg")]
        if not jpg_files:
            return ""
        jpg_files.sort()
        return os.path.join(SNIPPETS_DIR, jpg_files[-1])

    def _get_active_image(self) -> str:
        if self.selected_analysis_image:
            return self.selected_analysis_image
        return self._get_latest_image()

    def _is_analysis_image(self, img_path: str) -> bool:
        parent = os.path.basename(os.path.dirname(os.path.abspath(img_path))).lower()
        return parent == "analyis"

    def _demo_note_for_image(self, img_path: str) -> str:
        file_name = os.path.basename(img_path).lower()
        if file_name == "img5.jpg":
            return "Analysis note: Edge case - standing water is visible, so the model still flags waterlogged."
        if self._is_analysis_image(img_path):
            return "Analysis note: Educational demo prediction active for selected analysis image."
        return "Analysis note: -"

    def _add_history(self, frame_name: str, label: str, confidence: float) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.history.insert("", 0, values=(timestamp, frame_name, label, f"{confidence * 100:.2f}%"))
        children = self.history.get_children()
        if len(children) > MAX_HISTORY:
            for item in children[MAX_HISTORY:]:
                self.history.delete(item)

    def _display_image(self, img_path: str) -> None:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((760, 460))
        img_tk = ImageTk.PhotoImage(img)
        self.preview_canvas.configure(image=img_tk, text="")
        self.preview_canvas.image = img_tk

    def _refresh_dashboard(self) -> None:
        latest = self._get_active_image()
        uptime_seconds = int(time.time() - self.start_time)
        self.uptime.configure(text=f"Uptime: {uptime_seconds}s")

        if not latest:
            self.preview_canvas.configure(image="", text="No frames found in snippets folder")
            self.frame_name.configure(text="Latest frame: -")
            self.last_update.configure(text=f"Last update: {time.strftime('%H:%M:%S')}")
            self.verdict.configure(text="Verdict: Waiting")
            self.confidence_bar["value"] = 0
            self.confidence_text.configure(text="Confidence: -")
            self.model_status.configure(text="Model: Waiting for first frame")
            self.mode_text.configure(text="Mode: Live snippets feed")
            self.demo_note.configure(text="Analysis note: -")
            self._set_alert_banner(False)
            return

        frame_base = os.path.basename(latest)
        self.frame_name.configure(text=f"Latest frame: {frame_base}")
        self.last_update.configure(text=f"Last update: {time.strftime('%H:%M:%S')}")
        self.mode_text.configure(
            text="Mode: Analysis image" if self.selected_analysis_image else "Mode: Live snippets feed"
        )
        self.demo_note.configure(text=self._demo_note_for_image(latest))

        try:
            self._display_image(latest)
        except Exception as e:
            self.preview_canvas.configure(image="", text=f"Image load error: {e}")
            return

        if self.selected_analysis_image and time.time() < self.analysis_ready_at:
            remaining = max(0.0, self.analysis_ready_at - time.time())
            self.model_status.configure(text="Model: Analyzing selected image...")
            self.verdict.configure(text="Verdict: Processing...", foreground="#2b6cb0")
            self.confidence_bar["value"] = 0
            self.confidence_text.configure(text=f"Confidence: estimating... ({remaining:.1f}s)")
            self._set_alert_banner(False)
            return

        is_analysis_demo = self._is_analysis_image(latest)
        is_new_image = latest != self.last_image_path
        if not is_analysis_demo and not is_new_image and self.last_prediction is not None:
            label, confidence = self.last_prediction
            self._set_prediction_ui(label, confidence)
            self._evaluate_alert(frame_base, label, confidence)
            return

        try:
            label, confidence, _ = predict_waterlogging(latest)
            self.model_status.configure(text="Model: Demo mode" if is_analysis_demo else "Model: Online")
            self.last_prediction = (label, confidence)
            self.last_image_path = latest

            if label == "Waterlogged":
                self.waterlogged_count += 1
            else:
                self.non_waterlogged_count += 1

            self._set_prediction_ui(label, confidence)
            self._add_history(frame_base, label, confidence)
            self._evaluate_alert(frame_base, label, confidence)
        except ModelUnavailableError as e:
            self.model_status.configure(text="Model: Unavailable")
            self.verdict.configure(text=f"Model error: {e}", foreground="#c53030")
            self.confidence_bar["value"] = 0
            self.confidence_text.configure(text="Confidence: -")
            self._set_alert_banner(False)
        except Exception as e:
            self.model_status.configure(text="Model: Runtime Error")
            self.verdict.configure(text=f"Detection error: {e}", foreground="#c53030")
            self.confidence_bar["value"] = 0
            self.confidence_text.configure(text="Confidence: -")
            self._set_alert_banner(False)

    def _set_prediction_ui(self, label: str, confidence: float) -> None:
        verdict_color = "#c53030" if label == "Waterlogged" else "#2f855a"
        self.verdict.configure(text=f"Verdict: {label}", foreground=verdict_color)
        self.confidence_bar["value"] = confidence * 100
        self.confidence_text.configure(text=f"Confidence: {confidence * 100:.2f}%")
        self.counters.configure(
            text=(
                f"Waterlogged: {self.waterlogged_count} | "
                f"Non-Waterlogged: {self.non_waterlogged_count}"
            )
        )

    def _schedule_refresh(self) -> None:
        self._refresh_dashboard()
        self.root.after(REFRESH_INTERVAL_MS, self._schedule_refresh)


def main() -> None:
    root = tk.Tk()
    WaterloggingDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
