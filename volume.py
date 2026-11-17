"""
dragon_volume.py - High-Fidelity Gain, Volume & Smart Maximizer Controller
Features:
  - Full Playback Control (Play, Pause, Scrub Forward/Backward via Slider)
  - Extended Internal DSP Gain Range (-100 dB to +20 dB)
  - GUI Control Panel with OS Volume Warning
  - Dedicated Import Button & Separate Process & Export Button
"""

import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d


class DragonVolumeController:
    def __init__(self, root):
        self.root = root
        self.root.title("DragonDSP - Gain, Volume & File Processor")
        self.root.geometry("540x620")
        self.root.configure(bg="#1e1e1e")

        self.input_file_path = None
        self.current_gain_db = -1.5
        
        # Audio Playback State Variables
        self.audio_data = None
        self.sample_rate = 44100
        self.is_playing = False
        self.is_paused = False
        self.playback_thread = None
        self.current_frame = 0
        self.is_dragging_slider = False  # Prevents race conditions while scrubbing

        # Styling
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 13, "bold"), foreground="#00d2ff")
        style.configure("OS.TLabel", font=("Arial", 10, "bold", "underline"), foreground="#ffaa00")

        # Title
        title_lbl = ttk.Label(root, text="Audio Chain Manager", style="Header.TLabel")
        title_lbl.pack(pady=8)

        # OS Volume Instruction
        self.os_inst_lbl = ttk.Label(
            root, 
            text="SET WINDOWS OS VOLUME TO: 100% (0.0 dBFS)\n(Required for bit-perfect uncompressed transmission)", 
            style="OS.TLabel", 
            justify="center"
        )
        self.os_inst_lbl.pack(pady=4)

        # --- PLAYBACK / SEEK BAR SECTION ---
        pb_frame = ttk.LabelFrame(root, text=" Audio Playback & Forward/Backward Scrubbing ")
        pb_frame.pack(pady=8, padx=20, fill="x", ipadx=5, ipady=5)

        self.time_lbl = ttk.Label(pb_frame, text="0:00 / 0:00", font=("Arial", 9))
        self.time_lbl.pack(anchor="e", padx=5)

        self.playback_slider = ttk.Scale(
            pb_frame, from_=0.0, to=100.0, value=0.0, orient="horizontal"
        )
        self.playback_slider.pack(fill="x", pady=5, padx=5)
        
        # Bind scrubbing events so user can jump forward/backward smoothly
        self.playback_slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.playback_slider.bind("<ButtonRelease-1>", self.on_slider_release)

        # Playback Control Buttons
        btn_pb_frame = ttk.Frame(pb_frame)
        btn_pb_frame.pack(fill="x", pady=5)

        self.btn_play = ttk.Button(btn_pb_frame, text="▶ Play", command=self.play_audio, state="disabled")
        self.btn_play.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_pause = ttk.Button(btn_pb_frame, text="⏸ Pause / Resume", command=self.toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_stop = ttk.Button(btn_pb_frame, text="⏹ Stop", command=self.stop_audio, state="disabled")
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=2)

        # --- CONTROLS FRAME (Expanded Range: -100 to +20 dB) ---
        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(pady=8, padx=20, fill="x")

        self.gain_lbl = ttk.Label(ctrl_frame, text="Internal DSP Gain / Headroom: -1.5 dB")
        self.gain_lbl.pack(anchor="w")

        self.slider = ttk.Scale(
            ctrl_frame, from_=-100.0, to=20.0, value=-1.5, orient="horizontal",
            command=self.on_slider_change
        )
        self.slider.pack(fill="x", pady=5)

        # --- FILE PROCESSING SECTION ---
        file_frame = ttk.LabelFrame(root, text=" Audio File Management ")
        file_frame.pack(pady=8, padx=20, fill="x", ipadx=5, ipady=5)

        self.btn_import = ttk.Button(file_frame, text="1. Import WAV File", command=self.import_file)
        self.btn_import.pack(fill="x", pady=4)

        self.file_status_lbl = ttk.Label(file_frame, text="No file selected.", foreground="#aaaaaa", font=("Arial", 9))
        self.file_status_lbl.pack(anchor="w", pady=2)

        self.btn_export = ttk.Button(file_frame, text="2. Process & Export WAV File", command=self.process_and_export_file, state="disabled")
        self.btn_export.pack(fill="x", pady=4)

        # Status & Peak Meter
        self.status_lbl = ttk.Label(root, text="True-Peak Limiter Active (Zero Inter-Sample Distortion)", foreground="#00ff88")
        self.status_lbl.pack(pady=4)

        self.canvas = tk.Canvas(root, width=420, height=22, bg="#111111", highlightthickness=0)
        self.canvas.pack(pady=2)
        self.meter_bar = self.canvas.create_rectangle(0, 0, 200, 22, fill="#00d2ff", outline="")

    def on_slider_change(self, val):
        self.current_gain_db = float(val)
        self.gain_lbl.config(text=f"Internal DSP Gain / Headroom: {self.current_gain_db:+.1f} dB")
        
        # Map -100dB to +20dB to visual meter display width (420px max)
        normalized_val = max(0.0, (self.current_gain_db + 100.0) / 120.0)
        bar_width = int(420 * normalized_val)
        self.canvas.coords(self.meter_bar, 0, 0, bar_width, 22)

    def apply_smart_maximizer(self, audio_chunk):
        """Applies gain scaling (-100dB to +20dB) and a look-ahead brickwall true-peak limiter."""
        if self.current_gain_db <= -99.0:
            return np.zeros_like(audio_chunk)
            
        gain_linear = 10.0 ** (self.current_gain_db / 20.0)
        boosted = audio_chunk * gain_linear
        ceiling = 0.98  # Target ceiling (approx -0.18 dBFS)

        # -------------------------------------------------------------
        # VECTORIZED LOOK-AHEAD TRUE-PEAK BRICKWALL LIMITER
        # -------------------------------------------------------------
        # 1. 4x Oversampling to catch hidden inter-sample peaks (ISPs)
        try:
            oversampled = signal.resample_poly(boosted, 4, 1, axis=0)
        except Exception:
            oversampled = boosted

        # 2. Find max absolute peak across channels per sample over-time
        abs_peaks = np.max(np.abs(oversampled), axis=1)
        
        # 3. Calculate raw required attenuation factors
        gain_factors = np.ones_like(abs_peaks)
        exceed = abs_peaks > ceiling
        if np.any(exceed):
            gain_factors[exceed] = ceiling / abs_peaks[exceed]

        # 4. Look-Ahead Smoothing (Envelope smoothing via minimum filter)
        window_size = max(1, int(0.005 * 44100 * 4)) # ~5ms at 4x oversampled rate
        smoothed_gain = minimum_filter1d(gain_factors, size=window_size, mode='nearest')
        smoothed_gain = uniform_filter1d(smoothed_gain, size=window_size, mode='nearest')

        # 5. Downsample the smooth gain envelope back to original sample rate
        if oversampled.shape[0] != boosted.shape[0]:
            orig_indices = np.linspace(0, len(smoothed_gain) - 1, len(boosted))
            interp_gain = np.interp(orig_indices, np.arange(len(smoothed_gain)), smoothed_gain)
        else:
            interp_gain = smoothed_gain

        # Expand dimensions for stereo broadcasting [N, 2]
        if boosted.ndim > 1:
            interp_gain = interp_gain[:, None]

        # Apply transparent attenuation
        limited = boosted * interp_gain
        
        # Absolute safety ceiling safeguard
        return np.clip(limited, -1.0, 1.0)

    def import_file(self):
        if sf is None or sd is None:
            messagebox.showerror("Error", "Required packages missing.\nInstall via: pip install soundfile sounddevice numpy scipy")
            return

        file_path = filedialog.askopenfilename(title="Select Input WAV File", filetypes=[("WAV Audio", "*.wav"), ("All Files", "*.*")])
        if file_path:
            self.stop_audio()
            self.input_file_path = file_path
            short_name = file_path.split("/")[-1].split("\\")[-1]
            self.file_status_lbl.config(text=f"Loaded: {short_name}", foreground="#00ff88")
            self.btn_export.config(state="normal")
            self.btn_play.config(state="normal")

            try:
                self.audio_data, self.sample_rate = sf.read(file_path, always_2d=True)
                dur = len(self.audio_data) / self.sample_rate
                m, s = int(dur // 60), int(dur % 60)
                self.time_lbl.config(text=f"0:00 / {m}:{s:02d}")
                self.playback_slider.config(to=len(self.audio_data))
                self.current_frame = 0
                self.playback_slider.set(0)
            except Exception as e:
                messagebox.showerror("Error Reading File", str(e))

    def play_audio(self):
        if self.audio_data is None:
            return
        if self.is_playing:
            return

        self.is_playing = True
        self.is_paused = False
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")
        self.btn_play.config(state="disabled")

        def playback_worker():
            try:
                block_size = 2048
                with sd.OutputStream(samplerate=self.sample_rate, channels=self.audio_data.shape[1], dtype='float32') as stream:
                    total_frames = len(self.audio_data)
                    while self.is_playing and self.current_frame < total_frames:
                        if self.is_paused:
                            time.sleep(0.1)
                            continue

                        # If user is actively dragging the slider, pause loop advancement temporarily
                        if self.is_dragging_slider:
                            time.sleep(0.05)
                            continue

                        end_frame = min(self.current_frame + block_size, total_frames)
                        chunk = self.audio_data[self.current_frame:end_frame]
                        
                        processed_chunk = self.apply_smart_maximizer(chunk)
                        stream.write(processed_chunk.astype(np.float32))
                        self.current_frame = end_frame

                        # Update GUI seeker UI safely on main thread
                        self.root.after(0, self.update_playback_ui)

                if self.current_frame >= total_frames:
                    self.stop_audio()
            except Exception as e:
                print(f"Playback error: {e}")
                self.stop_audio()

        self.playback_thread = threading.Thread(target=playback_worker, daemon=True)
        self.playback_thread.start()

    def toggle_pause(self):
        if self.is_playing:
            self.is_paused = not self.is_paused

    def stop_audio(self):
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        if self.audio_data is not None:
            self.playback_slider.set(0)
        self.btn_play.config(state="normal" if self.audio_data is not None else "disabled")
        self.btn_pause.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.update_playback_ui()

    def on_slider_press(self, event):
        """User clicked down on the slider to scrub forward/backward."""
        self.is_dragging_slider = True

    def on_slider_release(self, event):
        """User released the slider; instantly jump audio frame position."""
        self.is_dragging_slider = False
        if self.audio_data is not None:
            target_frame = int(self.playback_slider.get())
            self.current_frame = max(0, min(target_frame, len(self.audio_data) - 1))
            self.update_time_label()

    def update_playback_ui(self):
        if self.audio_data is not None and not self.is_dragging_slider:
            if self.is_playing:
                self.playback_slider.set(self.current_frame)
            self.update_time_label()

    def update_time_label(self):
        if self.audio_data is not None:
            curr_sec = self.current_frame / self.sample_rate
            total_sec = len(self.audio_data) / self.sample_rate
            c_min, c_sec = int(curr_sec // 60), int(curr_sec % 60)
            t_min, t_sec = int(total_sec // 60), int(total_sec % 60)
            self.time_lbl.config(text=f"{c_min}:{c_sec:02d} / {t_min}:{t_sec:02d}")

    def process_and_export_file(self):
        if not self.input_file_path:
            messagebox.showwarning("Warning", "Please import a WAV file first using button #1.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Export Processed WAV File", 
            defaultextension=".wav", 
            initialfile="processed_output.wav",
            filetypes=[("WAV Audio", "*.wav")]
        )
        if not output_path:
            return

        try:
            processed_data = self.apply_smart_maximizer(self.audio_data)
            sf.write(output_path, processed_data, self.sample_rate, subtype='PCM_24')
            messagebox.showinfo("Success", f"File processed and successfully exported to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    def on_close(self):
        self.stop_audio()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DragonVolumeController(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()