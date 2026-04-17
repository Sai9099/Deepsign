import cv2
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import time
import threading

from predictor import SignLanguagePredictor
from utils import speak_text_async
import ml_classifier

LETTERS = [chr(i) for i in range(65, 91)]  # A-Z

# ─── Theme Configuration ──────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Color palette
COLORS = {
    "bg_dark":       "#121214", # Very dark zinc
    "bg_card":       "#1c1c1f", # Dark zinc card
    "bg_card_alt":   "#27272a", # Lighter zinc
    "accent":        "#ec4899", # Pink 500
    "accent_bright": "#f472b6", # Pink 400
    "primary":       "#3b82f6", # Blue 500
    "primary_hover": "#2563eb", # Blue 600
    "success":       "#10b981", # Emerald 500
    "success_hover": "#059669", # Emerald 600
    "danger":        "#ef4444", # Red 500
    "danger_hover":  "#dc2626", # Red 600
    "warning":       "#f59e0b", # Amber 500
    "warning_hover": "#d97706", # Amber 600
    "text":          "#f4f4f5", # Zinc 100
    "text_dim":      "#a1a1aa", # Zinc 400
    "text_bright":   "#ffffff", # White
    "highlight":     "#60a5fa", # Blue 400
    "border":        "#2c2c30", # Zinc 700
    "glow_blue":     "#3b82f6", 
    "glow_purple":   "#a855f7", 
}


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepSign — ASL Translator")
        self.root.geometry("1280x750")
        self.root.minsize(1150, 700)
        self.root.configure(fg_color=COLORS["bg_dark"])

        # Load predictor
        self.predictor = SignLanguagePredictor()

        # Open video source
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            print("Error: Could not open webcam.")

        # Logic State
        self.stable_letter = ""
        self.stable_start_time = 0
        self.letter_added = False
        self.required_hold_time = 2.0

        # Calibration State
        self.calibrating = False
        self.cal_letter_index = 0
        self.cal_samples_per_letter = 0
        self.cal_target_samples = 15
        self.cal_collecting = False
        self.cal_last_sample_time = 0

        self._build_gui()
        self.update_video()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _build_gui(self):
        # ─── Top Title Bar ────────────────────────────────────
        title_bar = ctk.CTkFrame(self.root, fg_color=COLORS["bg_card"], corner_radius=0, height=50)
        title_bar.pack(fill="x", padx=0, pady=0)
        title_bar.pack_propagate(False)

        # App icon/name
        ctk.CTkLabel(
            title_bar, text="🤟  DeepSign",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=COLORS["primary"]
        ).pack(side="left", padx=24)

        # Model status badge
        model_text = "✅ ML Model Active" if self.predictor.use_ml else "⚡ Rule-Based Mode"
        badge_color = COLORS["success"] if self.predictor.use_ml else COLORS["warning"]
        self.model_badge = ctk.CTkLabel(
            title_bar, text=model_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=badge_color,
            fg_color=COLORS["bg_dark"],
            corner_radius=12,
            padx=12, pady=4
        )
        self.model_badge.pack(side="right", padx=20)

        # ─── Main Content ─────────────────────────────────────
        content = ctk.CTkFrame(self.root, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=(10, 15))

        # ─── Left: Video Feed Card ────────────────────────────
        left_card = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=20, border_width=1, border_color=COLORS["border"])
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(
            left_card, text="📷  Live Camera Feed",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_dim"]
        ).pack(pady=(12, 8))

        # Video canvas — use standard tkinter Canvas for reliable video rendering
        video_wrapper = ctk.CTkFrame(left_card, fg_color=COLORS["bg_dark"], corner_radius=16)
        video_wrapper.pack(padx=20, pady=(0, 20), fill="both", expand=True)

        self.video_canvas = tk.Canvas(video_wrapper, bg=COLORS["bg_dark"], highlightthickness=0)
        self.video_canvas.pack(padx=6, pady=6, fill="both", expand=True)

        # ─── Right Panel ──────────────────────────────────────
        right_panel = ctk.CTkFrame(content, fg_color="transparent", width=380)
        right_panel.pack(side="right", fill="y", padx=(10, 0))
        right_panel.pack_propagate(False)

        # ─── Prediction Card ─────────────────────────────────
        pred_card = ctk.CTkFrame(right_panel, fg_color=COLORS["bg_card"], corner_radius=20, border_width=1, border_color=COLORS["border"])
        pred_card.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            pred_card, text="Current Prediction",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_dim"]
        ).pack(pady=(14, 4))

        # Big letter display
        self.lbl_letter = ctk.CTkLabel(
            pred_card, text="—",
            font=ctk.CTkFont(family="Segoe UI", size=96, weight="bold"),
            text_color=COLORS["highlight"]
        )
        self.lbl_letter.pack(pady=(0, 4))

        # Confidence bar
        conf_frame = ctk.CTkFrame(pred_card, fg_color="transparent")
        conf_frame.pack(fill="x", padx=25, pady=(0, 4))

        self.lbl_conf = ctk.CTkLabel(
            conf_frame, text="0%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text"]
        )
        self.lbl_conf.pack(side="right")

        ctk.CTkLabel(
            conf_frame, text="Accuracy",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_dim"]
        ).pack(side="left")

        self.conf_bar = ctk.CTkProgressBar(pred_card, height=12, corner_radius=6, fg_color=COLORS["bg_dark"], progress_color=COLORS["highlight"])
        self.conf_bar.pack(fill="x", padx=30, pady=(4, 12))
        self.conf_bar.set(0)

        # Status
        self.lbl_status = ctk.CTkLabel(
            pred_card, text="Waiting for hand...",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        self.lbl_status.pack(pady=(0, 14))

        # ─── Text Output Card ────────────────────────────────
        text_card = ctk.CTkFrame(right_panel, fg_color=COLORS["bg_card"], corner_radius=20, border_width=1, border_color=COLORS["border"])
        text_card.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(
            text_card, text="✍️  Formed Text",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_dim"]
        ).pack(pady=(14, 8))

        self.text_area = ctk.CTkTextbox(
            text_card, height=80, corner_radius=10,
            font=ctk.CTkFont(size=16),
            fg_color=COLORS["bg_dark"],
            text_color=COLORS["text_bright"],
            border_width=1, border_color=COLORS["border"]
        )
        self.text_area.pack(fill="x", padx=15, pady=(0, 10))

        # Action buttons row
        btn_row = ctk.CTkFrame(text_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=(0, 14))

        ctk.CTkButton(
            btn_row, text="Space", width=100, height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            corner_radius=10, command=self.add_space
        ).pack(side="left", padx=(0, 8), expand=True)

        ctk.CTkButton(
            btn_row, text="Clear", width=100, height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"],
            corner_radius=10, command=self.clear_text
        ).pack(side="left", padx=(0, 8), expand=True)

        ctk.CTkButton(
            btn_row, text="🔊 Speak", width=100, height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            corner_radius=10, command=self.speak_text
        ).pack(side="left", expand=True)

        # ─── Calibration Card ────────────────────────────────
        cal_card = ctk.CTkFrame(right_panel, fg_color=COLORS["bg_card"], corner_radius=20, border_width=1, border_color=COLORS["border"])
        cal_card.pack(fill="x", pady=(0, 0))

        ctk.CTkLabel(
            cal_card, text="🎯  Model Training",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_dim"]
        ).pack(pady=(14, 8))

        self.cal_btn = ctk.CTkButton(
            cal_card, text="Start Calibration", height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS["highlight"], hover_color=COLORS["primary_hover"],
            text_color=COLORS["bg_dark"],
            corner_radius=12, command=self.toggle_calibration
        )
        self.cal_btn.pack(fill="x", padx=20, pady=(4, 12))

        self.cal_progress = ctk.CTkProgressBar(cal_card, height=8, corner_radius=4, fg_color=COLORS["bg_dark"], progress_color=COLORS["highlight"])
        self.cal_progress.pack(fill="x", padx=25, pady=(0, 8))
        self.cal_progress.set(0)

        self.cal_status_lbl = ctk.CTkLabel(
            cal_card, text="Train with your hand signs for best accuracy",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_dim"],
            wraplength=260
        )
        self.cal_status_lbl.pack(pady=(0, 14))

    # ─── Video Loop ───────────────────────────────────────────────

    def update_video(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)

            if self.calibrating:
                frame = self._handle_calibration_frame(frame)
            else:
                frame, predicted_letter, confidence, debug_info = self.predictor.process_frame(frame)
                self._handle_prediction(predicted_letter, confidence, debug_info)

            # Convert and display on canvas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Resize to fit the canvas
            canvas_w = self.video_canvas.winfo_width()
            canvas_h = self.video_canvas.winfo_height()
            if canvas_w > 10 and canvas_h > 10:
                img_w, img_h = img.size
                scale = min(canvas_w / img_w, canvas_h / img_h)
                new_w, new_h = int(img_w * scale), int(img_h * scale)
                if new_w > 0 and new_h > 0:
                    img = img.resize((new_w, new_h), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_w // 2, canvas_h // 2,
                image=imgtk, anchor="center"
            )
            self.video_canvas._imgtk = imgtk  # Keep reference

        # Reduced frequency to 30ms (approx 33 FPS) to prevent UI lag
        self.root.after(32, self.update_video)

    def _handle_prediction(self, predicted_letter, confidence, debug_info):
        # Format diagnostic string
        ext = debug_info.get('ext', 0)
        ext_names = debug_info.get('ext_names', 'None')
        orient = debug_info.get('orient', 'up')
        diag_text = f"Fingers: {ext_names} ({ext}) | Orient: {orient.upper()} | "
        
        if predicted_letter:
            self.lbl_letter.configure(text=predicted_letter, text_color=COLORS["highlight"])
            self.lbl_conf.configure(text=f"{int(confidence * 100)}%")
            self.conf_bar.set(confidence)

            if confidence > 0.8:
                self.conf_bar.configure(progress_color=COLORS["success"])
            elif confidence > 0.5:
                self.conf_bar.configure(progress_color=COLORS["warning"])
            else:
                self.conf_bar.configure(progress_color=COLORS["danger"])

            if predicted_letter == self.stable_letter:
                hold_duration = time.time() - self.stable_start_time
                remaining = self.required_hold_time - hold_duration
                if hold_duration >= self.required_hold_time and not self.letter_added:
                    self.add_letter(predicted_letter)
                    self.letter_added = True
                    self.lbl_status.configure(text=f"{diag_text}✅ Added '{predicted_letter}'!", text_color=COLORS["success"])
                elif remaining > 0:
                    self.lbl_status.configure(text=f"{diag_text}Hold steady... {remaining:.1f}s", text_color=COLORS["text_dim"])
            else:
                self.stable_letter = predicted_letter
                self.stable_start_time = time.time()
                self.letter_added = False
                self.lbl_status.configure(text=f"{diag_text}Hold sign to add...", text_color=COLORS["text_dim"])
        else:
            self.lbl_letter.configure(text="—", text_color=COLORS["text_dim"])
            self.lbl_conf.configure(text="0%")
            self.conf_bar.set(0)
            self.conf_bar.configure(progress_color=COLORS["highlight"])
            self.stable_letter = ""
            self.letter_added = False
            self.lbl_status.configure(text=f"{diag_text}Show a hand sign...", text_color=COLORS["text_dim"])

    # ─── Calibration Mode ─────────────────────────────────────────

    def toggle_calibration(self):
        if self.calibrating:
            self._stop_calibration()
        else:
            self._start_calibration()

    def _start_calibration(self):
        self.calibrating = True
        self.cal_letter_index = 0
        self.cal_samples_per_letter = 0
        self.cal_collecting = False
        self.cal_last_sample_time = time.time()

        self.cal_btn.configure(text="⏹  Stop Calibration", fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"])
        self.lbl_letter.configure(text=LETTERS[0], text_color=COLORS["warning"])
        self.lbl_conf.configure(text="")
        self.conf_bar.set(0)
        self.lbl_status.configure(text="Get ready! Show letter A", text_color=COLORS["warning"])
        self.cal_status_lbl.configure(text="Show each letter when prompted — hold steady!", text_color=COLORS["warning"])

    def _stop_calibration(self):
        self.calibrating = False
        self.cal_btn.configure(text="Start Calibration", fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"])
        self.lbl_letter.configure(text="—", text_color=COLORS["highlight"])
        self.lbl_status.configure(text="Show a hand sign...", text_color=COLORS["text_dim"])

        counts = ml_classifier.get_training_counts()
        total = sum(counts.values())
        covered = sum(1 for c in counts.values() if c >= 3)

        self.cal_progress.set(covered / 26.0)

        if covered == 26:
            self.cal_status_lbl.configure(text="Training model...", text_color=COLORS["glow_purple"])
            self.root.update()
            self._train_model()
        else:
            self.cal_status_lbl.configure(
                text=f"{total} samples, {covered}/26 letters. Complete all 26 to train.",
                text_color=COLORS["text_dim"]
            )

    def _handle_calibration_frame(self, frame):
        current_letter = LETTERS[self.cal_letter_index]
        landmarks = self.predictor.detect_landmarks(frame)

        if landmarks:
            self.predictor._draw_landmarks(frame, landmarks)

        h, w, _ = frame.shape

        if not self.cal_collecting:
            elapsed = time.time() - self.cal_last_sample_time
            remaining = max(0, 3 - elapsed)

            if remaining > 0:
                # Countdown overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 30), -1)
                frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
                cv2.putText(frame, f"Show '{current_letter}' — starting in {int(remaining) + 1}...",
                           (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2, cv2.LINE_AA)
                self.lbl_status.configure(text=f"Prepare to show '{current_letter}'...", text_color=COLORS["warning"])
            else:
                self.cal_collecting = True
                self.cal_last_sample_time = time.time()
                self.cal_samples_per_letter = 0

        if self.cal_collecting:
            # Recording overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (0, 40, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            # Pulsing record dot
            pulse = int((time.time() * 4) % 2)
            dot_color = (0, 0, 255) if pulse else (0, 0, 180)
            cv2.circle(frame, (30, 40), 10, dot_color, -1)
            cv2.putText(frame, f"REC  '{current_letter}'  ({self.cal_samples_per_letter}/{self.cal_target_samples})",
                       (50, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 255, 100), 2, cv2.LINE_AA)

            elapsed = time.time() - self.cal_last_sample_time
            if elapsed >= 0.15 and landmarks:
                ml_classifier.save_training_sample(current_letter, landmarks)
                self.cal_samples_per_letter += 1
                self.cal_last_sample_time = time.time()

                progress = (self.cal_letter_index + self.cal_samples_per_letter / self.cal_target_samples) / 26.0
                self.cal_progress.set(progress)
                self.lbl_status.configure(
                    text=f"Recording '{current_letter}': {self.cal_samples_per_letter}/{self.cal_target_samples}",
                    text_color=COLORS["success"]
                )

            if self.cal_samples_per_letter >= self.cal_target_samples:
                self.cal_letter_index += 1
                self.cal_collecting = False
                self.cal_last_sample_time = time.time()
                self.cal_samples_per_letter = 0

                if self.cal_letter_index >= 26:
                    self._stop_calibration()
                    return frame

                next_letter = LETTERS[self.cal_letter_index]
                self.lbl_letter.configure(text=next_letter, text_color=COLORS["warning"])
                self.lbl_status.configure(text=f"Next up: '{next_letter}'", text_color=COLORS["warning"])

        # Bottom progress overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (15, 15, 30), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        progress_text = f"Letter {self.cal_letter_index + 1}/26: '{current_letter}'"
        cv2.putText(frame, progress_text, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 220), 1, cv2.LINE_AA)

        return frame

    def _train_model(self):
        def _train():
            success, msg = ml_classifier.train_model()
            self.root.after(0, lambda: self._on_training_complete(success, msg))

        thread = threading.Thread(target=_train, daemon=True)
        thread.start()

    def _on_training_complete(self, success, msg):
        if success:
            self.predictor.reload_ml_model()
            self.model_badge.configure(text="✅ ML Model Active", text_color=COLORS["success"])
            self.cal_status_lbl.configure(text=f"Done! {msg}", text_color=COLORS["success"])
            self.cal_progress.set(1.0)
            self.cal_progress.configure(progress_color=COLORS["success"])

            # Success dialog
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("Training Complete")
            dialog.geometry("400x200")
            dialog.configure(fg_color=COLORS["bg_card"])
            dialog.transient(self.root)
            dialog.grab_set()

            ctk.CTkLabel(dialog, text="🎉", font=ctk.CTkFont(size=40)).pack(pady=(20, 5))
            ctk.CTkLabel(dialog, text="Model Trained Successfully!",
                        font=ctk.CTkFont(size=16, weight="bold"),
                        text_color=COLORS["success"]).pack()
            ctk.CTkLabel(dialog, text=msg, font=ctk.CTkFont(size=11),
                        text_color=COLORS["text_dim"], wraplength=350).pack(pady=5)
            ctk.CTkButton(dialog, text="Got it!", command=dialog.destroy,
                         fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
                         corner_radius=8).pack(pady=8)
        else:
            self.cal_status_lbl.configure(text=f"Failed: {msg}", text_color=COLORS["danger"])

    # ─── Text Actions ─────────────────────────────────────────────

    def add_letter(self, letter):
        self.text_area.insert("end", letter)
        self.text_area.see("end")

    def add_space(self):
        self.text_area.insert("end", " ")
        self.text_area.see("end")

    def clear_text(self):
        self.text_area.delete("0.0", "end")

    def speak_text(self):
        text = self.text_area.get("0.0", "end").strip()
        speak_text_async(text)

    def on_closing(self):
        self.predictor.release()
        if self.vid.isOpened():
            self.vid.release()
        self.root.destroy()


if __name__ == '__main__':
    app = ctk.CTk()
    sign_app = SignLanguageApp(app)
    app.mainloop()
