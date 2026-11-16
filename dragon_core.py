"""
dragon_core.py - offline "horn-loaded flagship" renderer (inspired by the ESD Acoustic Super Dragon).

What it does
  1. Zero-phase 5-way crossover (perfect reconstruction) with per-band trim, transient shaping and
     even-harmonic saturation.
  2. Zero-phase voicing EQ (sub-sonic HPF, low shelf, box cut, horn-presence lift, air shelf).
  3. Sub-harmonic synth (ideal full-range system) OR psychoacoustic bass (small speakers like the KEF Q1 Meta).
  4. Downward expander, loudness match, look-ahead limiter, TPDF dither.
  5. Optional ROOM SIMULATION: image-source model of a shoebox room, constant-directivity horn patterns,
     frequency-dependent wall absorption, and a listener you can put ANYWHERE (binaural or stereo-mic render).

Everything here is deterministic offline processing; no real-time path.

Honest notes
  * ESD publishes 18 Hz - 52 kHz (+/-3 dB), 112 dB sensitivity, a five-band active crossover, field-coil drivers,
    beryllium/titanium diaphragms and carbon-fibre horns. It does NOT publish crossover points, horn coverage
    angles or driver positions: every such number in HORNS / Settings is an ASSUMPTION you can edit.
  * This is a plausible signature model, not a measurement of the real speaker.

oh yeah kids and remember, its ok to be broke, we cannot afford a ESD acoustics super dragon, thats fine yall, no need to freak out theres PLENTY of other speakers! (KEF Q1 meta's and even CHEAPER AUDIOPHILE BUDGET ONES!!!)

"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, fields

import numpy as np
from scipy import fft as sfft
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

C_SOUND = 343.0
OCT_CENTERS = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
OCT_EDGES = OCT_CENTERS[:-1] * math.sqrt(2.0)
AIR_DB_PER_M = np.array([0.0002, 0.0007, 0.002, 0.0034, 0.0057, 0.0115, 0.035, 0.12])
HEAD_RADIUS = 0.0875

# Assumed driver / horn layout (NOT published by ESD). H/V = -6 dB coverage angles in degrees,
# f_cd = frequency below which pattern control is lost (horn mouth size), z = acoustic-centre height (m).
# Official ESD Acoustic Super Dragon Horn & Driver Layout
HORNS = [
    dict(name="E20 Sub-bass carbon horn (S20 10\" Titanium Driver)", H=360.0, V=360.0, f_cd=17.0, z=0.45),
    dict(name="E100 Low-range carbon horn (S100 8\" Titanium Driver)", H=140.0, V=110.0, f_cd=100.0, z=1.00),
    dict(name="E500 Midrange carbon horn (S500 4\" Beryllium Driver)", H=100.0, V=70.0, f_cd=250.0, z=1.50),
    dict(name="High range S2000 1.75\" Beryllium Driver", H=80.0, V=50.0, f_cd=2000.0, z=1.90),
    dict(name="Tweeter S8000 1\" Beryllium Driver Array", H=70.0, V=40.0, f_cd=8000.0, z=2.15),
]

# Representative octave-band absorption coefficients (63 Hz ... 8 kHz). Approximate textbook-style values.
_A = lambda *v: np.array(v, dtype=float)
ABSORPTION_PRESETS = {
    "Living room (typical)": dict(
        front=_A(.12, .10, .07, .05, .04, .04, .05, .06),
        back=_A(.12, .10, .07, .05, .04, .04, .05, .06),
        side=_A(.12, .10, .07, .05, .04, .04, .05, .06),
        floor=_A(.08, .09, .20, .26, .27, .37, .34, .30),
        ceiling=_A(.12, .10, .07, .05, .04, .04, .05, .06)),
    "Bare / hard room": dict(
        front=_A(.01, .01, .02, .02, .02, .03, .04, .05),
        back=_A(.01, .01, .02, .02, .02, .03, .04, .05),
        side=_A(.01, .01, .02, .02, .02, .03, .04, .05),
        floor=_A(.01, .01, .02, .02, .03, .03, .03, .04),
        ceiling=_A(.01, .01, .02, .02, .02, .03, .04, .05)),
    "Treated listening room": dict(
        front=_A(.20, .30, .45, .55, .60, .60, .60, .55),
        back=_A(.30, .55, .80, .90, .90, .85, .80, .75),
        side=_A(.20, .35, .55, .70, .75, .70, .65, .60),
        floor=_A(.10, .15, .30, .40, .45, .50, .50, .45),
        ceiling=_A(.25, .40, .60, .75, .80, .75, .70, .65)),
    "Dead (anechoic-like)": dict(
        front=_A(.85, .90, .95, .97, .97, .97, .97, .97),
        back=_A(.85, .90, .95, .97, .97, .97, .97, .97),
        side=_A(.85, .90, .95, .97, .97, .97, .97, .97),
        floor=_A(.85, .90, .95, .97, .97, .97, .97, .97),
        ceiling=_A(.85, .90, .95, .97, .97, .97, .97, .97)),
    "ESD Reference Showroom": dict(
        front=_A(.28, .42, .62, .72, .75, .72, .68, .65),
        back=_A(.38, .65, .85, .92, .92, .88, .85, .80),
        side=_A(.22, .38, .58, .70, .74, .72, .68, .62),
        floor=_A(.12, .18, .35, .52, .60, .68, .72, .75),
        ceiling=_A(.30, .48, .68, .80, .82, .78, .74, .70)),
    "Nearfield Studio Desk": dict(
        front=_A(.30, .45, .65, .75, .78, .75, .70, .65),
        back=_A(.35, .55, .75, .85, .85, .80, .75, .70),
        side=_A(.25, .40, .60, .72, .75, .72, .68, .62),
        floor=_A(.10, .15, .30, .45, .55, .62, .65, .65),
        ceiling=_A(.28, .45, .65, .78, .80, .76, .72, .68)),
    "Cozy Bedroom (Carpet + Bed)": dict(
        front=_A(.16, .22, .32, .42, .48, .52, .56, .60),
        back=_A(.28, .40, .55, .68, .72, .75, .75, .70),
        side=_A(.15, .20, .30, .40, .45, .50, .55, .55),
        floor=_A(.10, .16, .32, .52, .65, .75, .80, .82),
        ceiling=_A(.12, .14, .18, .22, .26, .30, .35, .40)),
}

OUTPUT_MODES = ["speakers", "headphones_crossfeed", "room_binaural", "room_stereo"]
PROFILES = ["dragon_ideal", "kef_q1_meta"]


# ---------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Settings:
    # ----- profile / output
    profile: str = "dragon_ideal"          # "dragon_ideal" | "kef_q1_meta"
    output_mode: str = "room_binaural"     # see OUTPUT_MODES
    loudness_match: bool = True
    ceiling_db: float = -1.0
    export_bits: int = 24                  # 16 / 24 / 32 (32 = float)

    # ----- 5-way crossover
    xover_hz: list = field(default_factory=lambda: [70.0, 350.0, 2000.0, 8000.0])
    band_trim_db: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    # ----- voicing EQ (zero-phase)
    hp_hz: float = 20.0
    hp_order: int = 4                      # 4 = 24 dB/oct
    sub_shelf_db: float = 3.0
    sub_shelf_hz: float = 55.0
    box_cut_db: float = -2.0
    box_cut_lo: float = 180.0
    box_cut_hi: float = 250.0
    presence_db: float = 1.5
    presence_lo: float = 1500.0
    presence_hi: float = 4000.0
    air_db: float = 2.0
    air_hz: float = 12000.0

    # ----- dynamics
    attack_db: float = 3.0                 # transient shaper: leading-edge boost
    sustain_db: float = -1.0               # transient shaper: sustain / decay change
    band_transient_weight: list = field(default_factory=lambda: [0.2, 0.6, 1.0, 1.0, 0.7])
    expander_thr_db: float = -60.0
    expander_ratio: float = 1.4
    expander_range_db: float = 12.0

    # ----- saturation
    h2_percent: float = 0.3                # 2nd harmonic at -12 dBFS, mids and highs only

    # ----- bass
    bass_mode: str = "subharmonic"         # "off" | "subharmonic" | "psycho"
    bass_amount: float = 0.35

    # ----- speakers / headphones (no room) modes
    ambience_db: float = -22.0             # micro early-reflections (<15 ms); -60 = off
    crossfeed_db: float = -5.0

    # ----- room simulation
    room_preset: str = "Treated listening room"
    room_w: float = 8.0                    # x: across the front wall
    room_l: float = 10.0                   # y: front wall (0) to back wall
    room_h: float = 3.6
    spk_spacing: float = 3.0
    spk_front_dist: float = 1.8            # front wall to horn acoustic centres
    aim_y: float = 5.5                     # horns are aimed at (room_w/2, aim_y)
    horn_scale: float = 1.0                # <1 narrower patterns, >1 wider
    use_omni: bool = False                 # replace horns with omni point sources (comparison)
    reflection_order: int = 6
    room_seed: int = 7
    listener_x: float = 4.0
    listener_y: float = 5.5
    listener_z: float = 1.15
    auto_face: bool = True
    yaw_deg: float = 0.0                   # 0 = facing the front wall, + = turning right

    def aim_xy(self):
        return (self.room_w / 2.0, self.aim_y)


def apply_profile(s: Settings, profile: str) -> Settings:
    """Return s tuned for a playback target. Only touches the voicing/bass fields."""
    s.profile = profile
    if profile == "kef_q1_meta":
        # KEF Q1 Meta: 51 Hz-20 kHz (+/-3 dB), 47 Hz (-6 dB), ported. Do not push deep bass into the ported
        # driver (unloaded below tuning); recreate the deep-bass impression with harmonics instead.
        s.hp_hz = 40.0
        s.hp_order = 4
        s.sub_shelf_hz = 75.0
        s.sub_shelf_db = 1.5
        s.bass_mode = "psycho"
        s.bass_amount = 0.5
        s.air_db = 1.5
        s.xover_hz = [70.0, 350.0, 2000.0, 8000.0]
    else:
        s.hp_hz = 20.0
        s.hp_order = 4
        s.sub_shelf_hz = 55.0
        s.sub_shelf_db = 3.0
        s.bass_mode = "subharmonic"
        s.bass_amount = 0.35
        s.air_db = 2.0
    return s


def settings_to_json(s: Settings) -> str:
    return json.dumps({f.name: getattr(s, f.name) for f in fields(s)}, indent=2)


def settings_from_json(text: str) -> Settings:
    data = json.loads(text)
    valid = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------------------------------------------
def load_audio(path: str):
    """Return (float64 array [n, ch], sample_rate)."""
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        data, sr = sf.read(path, always_2d=True, dtype="float64")
        return data, int(sr)

    from scipy.io import wavfile
    if not path.lower().endswith(".wav"):
        raise RuntimeError("Only .wav can be read without the 'soundfile' package (pip install soundfile).")
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)
    if data.ndim == 1:
        data = data[:, None]
    return data, int(sr)


def save_audio(path: str, data: np.ndarray, sr: int, bits: int = 24, dither: bool = True) -> None:
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(1)
    if bits in (16, 24) and dither:
        lsb = 1.0 / (2 ** (bits - 1))
        data = data + (rng.random(data.shape) - rng.random(data.shape)) * lsb  # TPDF, +/-1 LSB
    data = np.clip(data, -1.0, 1.0)
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bits, "PCM_24")
        sf.write(path, data, sr, subtype=subtype)
        return
    from scipy.io import wavfile
    if not path.lower().endswith(".wav"):
        raise RuntimeError("Only .wav can be written without the 'soundfile' package (pip install soundfile).")
    if bits == 16:
        wavfile.write(path, sr, (data * 32767.0).astype(np.int16))
    else:
        wavfile.write(path, sr, data.astype(np.float32))


# ---------------------------------------------------------------------------------------------------------------
# Small DSP helpers
# ---------------------------------------------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def one_pole(x, tau_s, sr):
    a = math.exp(-1.0 / (max(tau_s, 1e-6) * sr))
    return signal.lfilter([1.0 - a], [1.0, -a], x, axis=0)


def smooth_zero(x, tau_s, sr):
    """Zero-phase one-pole smoothing (forward + backward)."""
    y = one_pole(x, tau_s, sr)
    return one_pole(y[::-1], tau_s, sr)[::-1]


def db_to_lin(db):
    return 10.0 ** (np.asarray(db) / 20.0)


def lin_to_db(x, floor=1e-9):
    return 20.0 * np.log10(np.maximum(x, floor))


def lp_mag(f, fc, n):
    """Amplitude-complementary low-pass magnitude. exponent 2n -> 12n dB/oct."""
    return 1.0 / (1.0 + (np.maximum(f, 0.0) / fc) ** (2 * n))


def split_masks(f, edges, n):
    """Return len(edges)+1 magnitude masks that sum to exactly 1 (perfect reconstruction, zero phase)."""
    masks = []
    carry = np.ones_like(f)
    for fc in edges:
        lp = lp_mag(f, fc, n)
        masks.append(carry * lp)
        carry = carry * (1.0 - lp)
    masks.append(carry)
    return masks


def hp_mag(f, fc, order):
    x = (fc / np.maximum(f, 1e-6)) ** (2 * order)
    return 1.0 / np.sqrt(1.0 + x)


def lp_butter_mag(f, fc, order):
    x = (np.maximum(f, 0.0) / fc) ** (2 * order)
    return 1.0 / np.sqrt(1.0 + x)


class ZeroPhaseDomain:
    """FFT a signal once, then produce many zero-phase filtered versions without circular wrap problems."""

    def __init__(self, x, sr, pad_s=0.6):
        self.x = np.asarray(x, dtype=np.float64)
        self.sr = sr
        self.n = self.x.shape[0]
        self.pad = int(pad_s * sr)
        self.nfft = sfft.next_fast_len(self.n + 2 * self.pad)
        self.f = sfft.rfftfreq(self.nfft, 1.0 / sr)
        # Long files: single precision halves memory (still >100 dB of dynamic range for audio work).
        self.dt = np.float32 if self.x.size > 6_000_000 else np.float64
        xp = np.zeros((self.nfft,) + self.x.shape[1:], dtype=self.dt)
        xp[self.pad:self.pad + self.n] = self.x
        self.X = sfft.rfft(xp, axis=0)
        del xp

    def apply(self, mag):
        shape = (-1,) + (1,) * (self.X.ndim - 1)
        y = sfft.irfft(self.X * mag.astype(self.dt).reshape(shape), n=self.nfft, axis=0)
        return y[self.pad:self.pad + self.n].astype(np.float64)


def zero_phase_filter(x, sr, mag_fn):
    d = ZeroPhaseDomain(x, sr)
    return d.apply(mag_fn(d.f))


# ---------------------------------------------------------------------------------------------------------------
# Dynamics, saturation, bass
# ---------------------------------------------------------------------------------------------------------------
def transient_shaper(x, sr, attack_db, sustain_db, fast_ms=1.0, slow_ms=40.0):
    """Differential-envelope transient shaper, stereo linked, with look-ahead so the leading edge is caught."""
    if abs(attack_db) < 1e-3 and abs(sustain_db) < 1e-3:
        return x
    env = np.max(np.abs(x), axis=1)
    fast = one_pole(env, fast_ms * 1e-3, sr)
    slow = one_pole(env, slow_ms * 1e-3, sr)
    eps = 1e-6
    d = 20.0 * np.log10((fast + eps) / (slow + eps))
    level_db = 20.0 * np.log10(slow + eps)
    gate = np.clip((level_db + 60.0) / 15.0, 0.0, 1.0)  # leave very quiet material alone
    gdb = (attack_db * np.clip(d / 6.0, 0.0, 1.0) + sustain_db * np.clip(-d / 6.0, 0.0, 1.0)) * gate
    gdb = smooth_zero(gdb, 0.5e-3, sr)
    shift = int(0.001 * sr)
    if shift > 0:
        gdb = np.concatenate([gdb[shift:], np.full(shift, gdb[-1])])
    return x * (10.0 ** (gdb / 20.0))[:, None]


def downward_expander(x, sr, thr_db, ratio, range_db):
    if ratio <= 1.0 or range_db <= 0.0:
        return x
    power = one_pole(np.max(x * x, axis=1), 30e-3, sr)       # stereo-linked running mean square
    lvl = 10.0 * np.log10(power + 1e-18)                       # dB RMS
    gdb = np.where(lvl < thr_db, (ratio - 1.0) * (lvl - thr_db), 0.0)
    gdb = np.maximum(gdb, -range_db)
    gdb = smooth_zero(gdb, 20e-3, sr)
    look = int(0.008 * sr)  # look-ahead so note onsets are not expanded
    if look > 0:
        gdb = np.concatenate([gdb[look:], np.full(look, gdb[-1])])
    return x * (10.0 ** (gdb / 20.0))[:, None]


def horn_throat_compression(x, sr, amount=0.2):
    """Simulates nonlinear air-pressure compression and soft-clipping inside physical horn throats."""
    if amount <= 0.0:
        return x
    # Soft asymmetric saturation modeling horn throat air compression
    up = signal.resample_poly(x, 2, 1, axis=0)
    compressed = np.tanh(up * (1.0 + amount * 0.5)) - (amount * 0.15) * (up ** 3)
    h = signal.resample_poly(compressed, 1, 2, axis=0)
    h = h[: x.shape[0]]
    # Apply primarily to mids and highs where horn loading is active
    sos = signal.butter(2, 300.0, "hp", fs=sr, output="sos")
    h = signal.sosfiltfilt(sos, h, axis=0)
    return x * (1.0 - amount * 0.25) + h * (amount * 0.25)


def add_second_harmonic(x, sr, percent, ref_amp=0.25):
    """y = x + c*x^2 on a 4x oversampled copy; the x^2 term is high-passed so only 2nd harmonics (no bass thump)
    are added. 'percent' = 2nd harmonic amplitude relative to the fundamental at -12 dBFS (A=0.25)."""
    if percent <= 0.0:
        return x
    c = 2.0 * (percent / 100.0) / ref_amp
    up = signal.resample_poly(x, 4, 1, axis=0)
    h = signal.resample_poly(c * up * up, 1, 4, axis=0)
    h = h[: x.shape[0]]
    sos = signal.butter(2, 120.0, "hp", fs=sr, output="sos")
    h = signal.sosfiltfilt(sos, h, axis=0)
    return x + h


def subharmonic_synth(x, sr, amount):
    """dbx-120A-style octave-down generator: flip-flop divider on the 45-110 Hz band, enveloped, low-passed."""
    if amount <= 0.0:
        return np.zeros_like(x)
    mono = x.mean(axis=1)
    band = zero_phase_filter(mono, sr, lambda f: hp_mag(f, 45.0, 2) * lp_butter_mag(f, 110.0, 2))
    s = np.where(band >= 0.0, 1.0, -1.0)
    rise = (s[1:] > 0) & (s[:-1] < 0)
    toggles = np.zeros_like(s)
    toggles[1:] = rise
    state = (np.cumsum(toggles) % 2) * 2.0 - 1.0
    env = one_pole(np.abs(band), 15e-3, sr)
    sub = state * env
    sub = zero_phase_filter(sub, sr, lambda f: hp_mag(f, 22.0, 2) * lp_butter_mag(f, 60.0, 4))
    rms_b = math.sqrt(np.mean(band ** 2) + 1e-18)
    rms_s = math.sqrt(np.mean(sub ** 2) + 1e-18)
    sub *= amount * rms_b / rms_s
    return np.repeat(sub[:, None], x.shape[1], axis=1)


def psycho_bass(x, sr, amount):
    """Missing-fundamental bass for small speakers: harmonics (2f, 3f) of the 32-90 Hz band, placed at 70-260 Hz."""
    if amount <= 0.0:
        return np.zeros_like(x)
    mono = x.mean(axis=1)
    band = zero_phase_filter(mono, sr, lambda f: hp_mag(f, 32.0, 2) * lp_butter_mag(f, 90.0, 2))
    peak = max(np.percentile(np.abs(band), 99.5), 1e-6)
    u = band / peak
    harm = 0.5 * np.tanh(2.2 * u) + 1.0 * (np.abs(u) - np.mean(np.abs(u)))
    harm = zero_phase_filter(harm, sr, lambda f: hp_mag(f, 75.0, 4) * lp_butter_mag(f, 260.0, 2))
    rms_b = math.sqrt(np.mean(band ** 2) + 1e-18)
    rms_h = math.sqrt(np.mean(harm ** 2) + 1e-18)
    harm *= amount * rms_b / rms_h
    return np.repeat(harm[:, None], x.shape[1], axis=1)


def micro_ambience(x, sr, level_db):
    """Diffused ultra-short (<18 ms) early reflections, partly cross-fed, 5.5 kHz low-passed."""
    if level_db <= -59.0:
        return x
    taps_ms = [2.9, 4.3, 5.8, 7.1, 8.6, 10.2, 11.7, 13.1, 14.8, 16.2, 17.5, 18.9]
    drop_db = [-1.5, -2.0, -3.0, -3.5, -4.5, -5.5, -6.0, -7.0, -8.0, -9.0, -10.0, -11.5]
    wet = np.zeros_like(x)
    for k, (ms, dd) in enumerate(zip(taps_ms, drop_db)):
        d = int(ms * 1e-3 * sr)
        g = (10.0 ** ((level_db + dd) / 20.0)) * 0.65
        src = x[:, ::-1] if (k % 2) else x
        pol = -1.0 if (k % 3 == 1) else 1.0
        wet[d:] += pol * g * src[: x.shape[0] - d]
    b, a = signal.butter(1, 5500.0, "low", fs=sr)
    wet = signal.lfilter(b, a, wet, axis=0)
    return x + wet


def crossfeed(x, sr, level_db):
    d = max(1, int(0.00028 * sr))
    b, a = signal.butter(1, 700.0, "low", fs=sr)
    g = 10.0 ** (level_db / 20.0)
    cross = signal.lfilter(b, a, x[:, ::-1], axis=0)
    cross = np.concatenate([np.zeros((d, 2)), cross[: x.shape[0] - d]])
    return (x + g * cross) / (1.0 + g * 0.5)


def k_weighted_ms(x, sr):
    """Approximate BS.1770 K-weighted mean square (ungated), summed over channels."""
    def shelf():
        G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
        A = 10 ** (G / 40.0)
        w0 = 2 * math.pi * fc / sr
        al = math.sin(w0) / (2 * Q)
        cw = math.cos(w0)
        sA = 2 * math.sqrt(A) * al
        b0 = A * ((A + 1) + (A - 1) * cw + sA)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - sA)
        a0 = (A + 1) - (A - 1) * cw + sA
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - sA
        return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])

    def hp():
        fc, Q = 38.13547087613982, 0.5003270373253953
        w0 = 2 * math.pi * fc / sr
        al = math.sin(w0) / (2 * Q)
        cw = math.cos(w0)
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
        a = np.array([1 + al, -2 * cw, 1 - al])
        return b / a[0], a / a[0]

    b1, a1 = shelf()
    b2, a2 = hp()
    y = signal.lfilter(b1, a1, x, axis=0)
    y = signal.lfilter(b2, a2, y, axis=0)
    return float(np.sum(np.mean(y ** 2, axis=0)))


def lookahead_limiter(x, sr, ceiling_db=-1.0, lookahead_ms=10.0):
    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = np.max(np.abs(x), axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    la = max(1, int(lookahead_ms * 1e-3 * sr))
    g = minimum_filter1d(need, size=2 * la + 1, mode="nearest")
    g = uniform_filter1d(g, size=2 * la + 1, mode="nearest")  # average of mins never exceeds 'need'
    y = x * g[:, None]
    return np.clip(y, -ceiling, ceiling)


# ---------------------------------------------------------------------------------------------------------------
# Voicing EQ
# ---------------------------------------------------------------------------------------------------------------
def _plateau_db(f, lo, hi, gain_db, edge_oct=0.5):
    lf = np.log(np.maximum(f, 1e-3))
    s = edge_oct * math.log(2.0) / 2.0
    return gain_db * _sigmoid((lf - math.log(lo)) / s) * _sigmoid((math.log(hi) - lf) / s)


def voicing_magnitude(f, s: Settings):
    lf = np.log(np.maximum(f, 1e-3))
    g = np.zeros_like(f)
    g += s.sub_shelf_db * (1.0 - _sigmoid((lf - math.log(max(s.sub_shelf_hz, 1.0))) / 0.30))
    g += _plateau_db(f, s.box_cut_lo, s.box_cut_hi, s.box_cut_db, 0.6)
    g += _plateau_db(f, s.presence_lo, s.presence_hi, s.presence_db, 0.9)
    g += s.air_db * _sigmoid((lf - math.log(max(s.air_hz, 1.0))) / 0.35)
    mag = 10.0 ** (g / 20.0)
    if s.hp_hz > 0:
        mag = mag * hp_mag(f, s.hp_hz, s.hp_order)
    return mag


# ---------------------------------------------------------------------------------------------------------------
# Room simulation
# ---------------------------------------------------------------------------------------------------------------
def horn_gain(dirs, axis, H, V, f_cd, fc, back_db=30.0):
    """Constant-directivity horn pattern, linear gain (1.0 on axis). dirs: (N,3) unit vectors source->receiver.
    -6 dB at half the coverage angle, widening toward omni below the horn's control frequency f_cd."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    sv = np.cross(up, a)
    if np.linalg.norm(sv) < 1e-6:
        sv = np.array([1.0, 0.0, 0.0])
    sv /= np.linalg.norm(sv)
    vv = np.cross(a, sv)
    dx = dirs @ a
    dh = dirs @ sv
    dv = dirs @ vv
    th = np.degrees(np.arctan2(dh, dx))
    tv = np.degrees(np.arctan2(dv, dx))
    w = max(1.0, f_cd / fc) ** 1.2
    He = max(H * w, 1.0)
    Ve = max(V * w, 1.0)
    att = 6.0 * ((th / (He / 2.0)) ** 2 + (tv / (Ve / 2.0)) ** 2)
    att = np.minimum(att, back_db / w)
    return 10.0 ** (-att / 20.0)


def _sphere_points(n=3000):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


_SPHERE = _sphere_points()


def horn_mean_g2(H, V, f_cd, fc):
    """Mean squared pattern gain over the sphere (= 1/Q). Controls how much energy goes into the reverb."""
    g = horn_gain(_SPHERE, np.array([1.0, 0.0, 0.0]), H, V, f_cd, fc)
    return float(np.mean(g ** 2))


def driver_for_band(b, xover_hz):
    return int(np.sum(np.array(xover_hz) < OCT_CENTERS[b]))


@dataclass
class Scene:
    src: list           # src[ch][driver] -> (3,)
    axis: list          # axis[ch][driver] -> (3,)
    listener: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    yaw_deg: float


def build_scene(s: Settings) -> Scene:
    cx = s.room_w / 2.0
    # Outer placement for massive E20 sub-bass horns, inner towers for mid/high/tweeter stacks
    xs_sub = [cx - s.spk_spacing / 2.0, cx + s.spk_spacing / 2.0]
    xs_tower = [cx - s.spk_spacing / 4.0, cx + s.spk_spacing / 4.0]
    
    aim = np.array([s.aim_xy()[0], s.aim_xy()[1], s.listener_z])
    src, axis = [], []
    for ch in range(2):
        srcs, axes = [], []
        for i, h in enumerate(HORNS):
            # Assign specific physical heights/lateral offsets matching the stack configuration
            if i == 0:  # E20 Sub-bass horn
                p = np.array([xs_sub[ch], s.spk_front_dist, h["z"]])
            else:       # Central towers: E100, E500, S2000, S8000 stack
                p = np.array([xs_tower[ch], s.spk_front_dist, h["z"]])
                
            srcs.append(p)
            axes.append(aim - p)
        src.append(srcs)
        axis.append(axes)
        
    lis = np.array([s.listener_x, s.listener_y, s.listener_z])
    if s.auto_face:
        tgt = np.array([cx, s.spk_front_dist])
        d = tgt - lis[:2]
        yaw = math.degrees(math.atan2(d[0], -d[1])) if np.linalg.norm(d) > 1e-6 else 0.0
    else:
        yaw = s.yaw_deg
    yr = math.radians(yaw)
    forward = np.array([math.sin(yr), -math.cos(yr), 0.0])
    right = np.array([math.cos(yr), math.sin(yr), 0.0])
    return Scene(src, axis, lis, forward, right, yaw)


# 3. voicing EQ (zero-phase)
    tick("Voicing EQ", 0.58)
    out = zero_phase_filter(out, sr, lambda f: voicing_magnitude(f, s))
    out = out + synth


# 3.5 Horn throat air compression (Super Dragon acoustic signature)
    if s.profile == "dragon_ideal":
        out = horn_throat_compression(out, sr, amount=0.25)


def _surface_areas(s):
    W, L, H = s.room_w, s.room_l, s.room_h
    return dict(front=W * H, back=W * H, left=L * H, right=L * H, floor=W * L, ceiling=W * L)


def _absorption(s):
    p = ABSORPTION_PRESETS.get(s.room_preset, ABSORPTION_PRESETS["Treated listening room"])
    return dict(front=p["front"], back=p["back"], left=p["side"], right=p["side"],
                floor=p["floor"], ceiling=p["ceiling"])


def room_acoustics(s):
    """Per octave band: total absorption (m^2 sabins), RT60 and room constant R."""
    ab, area = _absorption(s), _surface_areas(s)
    A = sum(ab[k] * area[k] for k in area)
    S = sum(area.values())
    V = s.room_w * s.room_l * s.room_h
    T60 = np.clip(0.161 * V / np.maximum(A, 1e-6), 0.12, 3.5)
    abar = np.clip(A / S, 1e-3, 0.98)
    R = S * abar / (1.0 - abar)
    return dict(A=A, S=S, V=V, T60=T60, R=R)


_KERNELS: dict = {}


def head_shadow_kernel(theta_deg, sr, taps=48):
    """Brown-Duda spherical-head shadow filter + pinna directional cues as a short FIR."""
    key = (int(round(theta_deg / 5.0)), sr)
    k = _KERNELS.get(key)
    if k is not None:
        return k
    th = key[0] * 5.0
    alpha = 1.05 + 0.95 * math.cos(math.radians(th * 180.0 / 150.0))
    w0 = C_SOUND / HEAD_RADIUS
    bz, az = signal.bilinear([alpha / (2 * w0), 1.0], [1.0 / (2 * w0), 1.0], fs=sr)
    imp = np.zeros(taps)
    imp[0] = 1.0
    k = signal.lfilter(bz, az, imp)
    # Pinna (outer-ear) cues: subtle concha resonance at ~4.5 kHz and frontal notch at ~7.8 kHz
    if sr >= 22050:
        frontness = max(0.0, math.cos(math.radians(th)))
        b_concha, a_concha = signal.iirpeak(4500.0, Q=3.0, fs=sr)
        b_notch, a_notch = signal.iirpeak(7800.0, Q=4.5, fs=sr)
        k = k + (0.18 * frontness) * signal.lfilter(b_concha, a_concha, k)
        k = k - (0.32 * frontness) * signal.lfilter(b_notch, a_notch, k)
    _KERNELS[key] = k
    return k


def _image_set(s: Settings, order: int):
    r = np.arange(-order, order + 1)
    mx, my, mz = np.meshgrid(r, r, r, indexing="ij")
    keep = (np.abs(mx) + np.abs(my) + np.abs(mz)) <= order
    return mx[keep], my[keep], mz[keep]


def _img_coord(m, size, pos):
    return np.where((m & 1) == 0, m * size + pos, (m + 1) * size - pos)


def _hits(m):
    """(hits on the low wall, hits on the high wall) for image index m."""
    k = np.abs(m)
    hi_pos = (k + 1) // 2
    lo_pos = k // 2
    lo = np.where(m > 0, lo_pos, hi_pos)
    hi = np.where(m > 0, hi_pos, lo_pos)
    return np.where(m == 0, 0, lo), np.where(m == 0, 0, hi)


def _early_ir(s, scene, ch, b, sr, n_ir, pre, t_mix, mode):
    """Direct sound + image-source reflections (up to t_mix) for one speaker channel and one octave band.
    Returns array (2 ears, n_ir)."""
    ab = _absorption(s)
    beta = {k: math.sqrt(max(1.0 - float(v[b]), 0.0)) for k, v in ab.items()}
    di = driver_for_band(b, s.xover_hz)
    horn = HORNS[di]
    src = scene.src[ch][di]
    axis = scene.axis[ch][di]
    fc = float(OCT_CENTERS[b])
    scale = 1.0 if not s.use_omni else None

    mx, my, mz = _image_set(s, int(s.reflection_order))
    ix = _img_coord(mx, s.room_w, src[0])
    iy = _img_coord(my, s.room_l, src[1])
    iz = _img_coord(mz, s.room_h, src[2])
    lo_x, hi_x = _hits(mx)
    lo_y, hi_y = _hits(my)
    lo_z, hi_z = _hits(mz)
    refl = (beta["left"] ** lo_x) * (beta["right"] ** hi_x) * (beta["front"] ** lo_y) * (beta["back"] ** hi_y) \
        * (beta["floor"] ** lo_z) * (beta["ceiling"] ** hi_z)

    img = np.stack([ix, iy, iz], axis=1)
    flip = np.stack([np.where(mx & 1, -1.0, 1.0), np.where(my & 1, -1.0, 1.0), np.where(mz & 1, -1.0, 1.0)], axis=1)

    ears = np.zeros((2, n_ir))
    lis = scene.listener

    if mode == "room_stereo":
        # Virtual ORTF-style pair: 17 cm apart, cardioids toed out +/-55 degrees.
        rt, fw = scene.right, scene.forward
        mic_pos = [lis - 0.085 * rt, lis + 0.085 * rt]
        mic_ax = [math.cos(math.radians(55)) * fw - math.sin(math.radians(55)) * rt,
                  math.cos(math.radians(55)) * fw + math.sin(math.radians(55)) * rt]
        for e in range(2):
            vec = img - mic_pos[e]
            r = np.maximum(np.linalg.norm(vec, axis=1), 0.2)
            dirs_to_mic = -vec / r[:, None]
            g_src = _pattern(s, dirs_to_mic, axis * flip, horn, fc)
            cardioid = 0.5 + 0.5 * ((vec / r[:, None]) @ mic_ax[e])
            amp = g_src * cardioid * refl / r * 10.0 ** (-AIR_DB_PER_M[b] * r / 20.0)
            t = r / C_SOUND
            bounce_order = np.abs(mx) + np.abs(my) + np.abs(mz)
            rng_diff = np.random.default_rng(s.room_seed + ch * 17 + b)
            jitter = np.where(bounce_order > 0, rng_diff.uniform(-0.0018, 0.0018, size=t.shape), 0.0)
            t = np.maximum(t + jitter, 0.0)
            ok = (t < t_mix) & (amp > 1e-5)
            for ai, ti in zip(amp[ok], t[ok]):
                pos = pre + ti * sr
                d0 = int(pos)
                fr = pos - d0
                if d0 + 2 < n_ir:
                    ears[e, d0] += ai * (1.0 - fr)
                    ears[e, d0 + 1] += ai * fr
        return ears

    # Binaural
    vec = img - lis
    r = np.maximum(np.linalg.norm(vec, axis=1), 0.2)
    to_head = vec / r[:, None]                 # unit vector listener -> image source
    dirs_src_to_rec = -to_head
    g_src = _pattern(s, dirs_src_to_rec, axis * flip, horn, fc)
    amp = g_src * refl / r * 10.0 ** (-AIR_DB_PER_M[b] * r / 20.0)
    t = r / C_SOUND
    bounce_order = np.abs(mx) + np.abs(my) + np.abs(mz)
    rng_diff = np.random.default_rng(s.room_seed + ch * 17 + b)
    jitter = np.where(bounce_order > 0, rng_diff.uniform(-0.0018, 0.0018, size=t.shape), 0.0)
    t = np.maximum(t + jitter, 0.0)

    lateral = np.arcsin(np.clip(to_head @ scene.right, -1.0, 1.0))
    itd = (HEAD_RADIUS / C_SOUND) * (np.abs(lateral) + np.sin(np.abs(lateral))) * np.sign(lateral)
    phi = math.radians(10.0)
    ear_axis = [-math.cos(phi) * scene.right - math.sin(phi) * scene.forward,   # left ear
                math.cos(phi) * scene.right - math.sin(phi) * scene.forward]    # right ear
    ok = (t < t_mix) & (amp > 1e-5)
    idx = np.nonzero(ok)[0]
    K = 48
    for e in range(2):
        theta = np.degrees(np.arccos(np.clip(to_head @ ear_axis[e], -1.0, 1.0)))
        sign = -1.0 if e == 1 else 1.0          # right ear (e=1) leads for sources on the right
        te = t + sign * itd * 0.5
        for i in idx:
            pos = pre + te[i] * sr
            d0 = int(pos)
            fr = pos - d0
            if d0 + K + 2 >= n_ir or d0 < 0:
                continue
            kern = head_shadow_kernel(theta[i], sr, K)
            a = amp[i]
            ears[e, d0:d0 + K] += a * (1.0 - fr) * kern
            ears[e, d0 + 1:d0 + 1 + K] += a * fr * kern
    return ears


def _pattern(s, dirs, axes_flip, horn, fc):
    """axes_flip: (N,3) per-image mirrored axis (axis * flip). Pattern evaluated per image."""
    if s.use_omni:
        return np.ones(dirs.shape[0])
    out = np.empty(dirs.shape[0])
    # Group by unique axis (at most 8 parity combinations) for speed.
    keys = np.round(axes_flip, 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    for u in range(uniq.shape[0]):
        sel = inv == u
        out[sel] = horn_gain(dirs[sel], uniq[u], horn["H"] * s.horn_scale, horn["V"] * s.horn_scale,
                             horn["f_cd"], fc)
    return out


def _late_tail(s, ch, b, sr, n_tail, rng, acoustics, mode):
    di = driver_for_band(b, s.xover_hz)
    horn = HORNS[di]
    fc = float(OCT_CENTERS[b])
    if s.use_omni:
        g2 = 1.0
    else:
        g2 = horn_mean_g2(horn["H"] * s.horn_scale, horn["V"] * s.horn_scale, horn["f_cd"], fc)
    T60 = float(acoustics["T60"][b])
    R = float(acoustics["R"][b])
    p_rev2 = 16.0 * math.pi * g2 / R
    t_mix = 0.08
    energy = p_rev2 * math.exp(-13.8155 * t_mix / T60)
    t = np.arange(n_tail) / sr
    env = np.exp(-6.9078 * t / T60) * np.clip(t / 0.02, 0.0, 1.0)
    rho_tab = np.array([0.95, 0.80, 0.50, 0.25, 0.10, 0.05, 0.0, 0.0])
    rho = rho_tab[b] * (0.6 if mode == "room_stereo" else 1.0)
    n0 = rng.standard_normal(n_tail)
    out = np.zeros((2, n_tail))
    for e in range(2):
        ne = rng.standard_normal(n_tail)
        sig = (math.sqrt(rho) * n0 + math.sqrt(1.0 - rho) * ne) * env
        sig *= math.sqrt(energy / (np.sum(sig ** 2) + 1e-18))
        out[e] = sig
    return out


def octave_masks(f):
    return split_masks(f, OCT_EDGES, 1)


def room_irs(s: Settings, sr: int, mode: str = None):
    """Build the 2x2 (speaker channel -> ear) impulse-response matrix. Returns (irs[ch][ear][n], pre, direct_min)."""
    mode = mode or s.output_mode
    scene = build_scene(s)
    ac = room_acoustics(s)
    pre = int(0.05 * sr)
    t_mix = 0.08
    tail_s = min(float(np.max(ac["T60"])) * 1.05, 2.5)
    n_ir = pre + int((t_mix + tail_s) * sr) + int(0.05 * sr)
    nfft = sfft.next_fast_len(n_ir)
    f = sfft.rfftfreq(nfft, 1.0 / sr)
    masks = octave_masks(f)
    rng = np.random.default_rng(s.room_seed)
    H = np.zeros((2, 2, nfft // 2 + 1), dtype=complex)
    tail_start = pre + int(t_mix * sr)
    n_tail = n_ir - tail_start
    for ch in range(2):
        for b in range(len(OCT_CENTERS)):
            early = _early_ir(s, scene, ch, b, sr, n_ir, pre, t_mix, mode)
            tail = _late_tail(s, ch, b, sr, n_tail, rng, ac, mode)
            early[:, tail_start:] += tail
            for e in range(2):
                H[ch, e] += masks[b] * sfft.rfft(early[e], n=nfft)
    irs = sfft.irfft(H, n=nfft, axis=-1)[..., :n_ir]
    direct_min = min(float(np.linalg.norm(scene.src[ch][di] - scene.listener)) / C_SOUND
                     for ch in range(2) for di in range(len(HORNS)))
    return irs, pre, direct_min


def render_room(x, sr, s: Settings):
    irs, pre, direct_min = room_irs(s, sr)
    n = x.shape[0]
    start = pre + int(direct_min * sr)
    out = np.zeros((n, 2))
    for ear in range(2):
        for ch in range(2):
            y = signal.oaconvolve(x[:, ch], irs[ch, ear], mode="full")
            seg = y[start:start + n]
            out[: seg.shape[0], ear] += seg
    return out


def listener_report(s: Settings) -> dict:
    """Plain-language numbers for the GUI: distance and off-axis angle to each horn and the pattern loss there."""
    scene = build_scene(s)
    rep = {"yaw_deg": scene.yaw_deg, "speakers": []}
    for ch, name in enumerate(("Left", "Right")):
        di = 3  # high horn
        p = scene.src[ch][di]
        v = scene.listener - p
        r = float(np.linalg.norm(v))
        a = scene.axis[ch][di]
        cosang = float(np.dot(v / r, a / np.linalg.norm(a)))
        off = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        losses = {}
        for fc, di2 in ((500.0, 2), (4000.0, 3), (10000.0, 4)):
            horn = HORNS[di2]
            if s.use_omni:
                losses[int(fc)] = 0.0
            else:
                g = horn_gain((v / r)[None, :], scene.axis[ch][di2], horn["H"] * s.horn_scale,
                              horn["V"] * s.horn_scale, horn["f_cd"], fc)[0]
                losses[int(fc)] = float(20 * math.log10(max(g, 1e-6)))
        rep["speakers"].append(dict(name=name, distance_m=r, off_axis_deg=off, loss_db=losses))
    ac = room_acoustics(s)
    rep["rt60"] = {int(c): float(t) for c, t in zip(OCT_CENTERS, ac["T60"])}
    return rep


# ---------------------------------------------------------------------------------------------------------------
# Full chain
# ---------------------------------------------------------------------------------------------------------------
def process(audio: np.ndarray, sr: int, s: Settings, progress=None) -> np.ndarray:
    def tick(msg, frac):
        if progress:
            progress(msg, frac)

    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    x = x - np.mean(x, axis=0, keepdims=True)
    ref_ms = k_weighted_ms(x, sr)

    # 1. zero-phase 5-way split, per-band shaping
    tick("Splitting into 5 bands", 0.05)
    dom = ZeroPhaseDomain(x, sr)
    masks = split_masks(dom.f, s.xover_hz, 2)
    out = np.zeros_like(x)
    for k, m in enumerate(masks):
        band = dom.apply(m)
        w = s.band_transient_weight[k] if k < len(s.band_transient_weight) else 1.0
        band = transient_shaper(band, sr, s.attack_db * w, s.sustain_db * w)
        if k >= 2:
            band = add_second_harmonic(band, sr, s.h2_percent)
        trim = s.band_trim_db[k] if k < len(s.band_trim_db) else 0.0
        out += band * (10.0 ** (trim / 20.0))
        tick(f"Band {k + 1}/5 shaped", 0.05 + 0.08 * (k + 1))
    del dom

    # 2. bass synthesis (computed from the shaped signal, added after the HPF)
    tick("Bass synthesis", 0.5)
    if s.bass_mode == "subharmonic":
        synth = subharmonic_synth(out, sr, s.bass_amount)
    elif s.bass_mode == "psycho":
        synth = psycho_bass(out, sr, s.bass_amount)
    else:
        synth = np.zeros_like(out)

    # 3. voicing EQ (zero-phase)
    tick("Voicing EQ", 0.58)
    out = zero_phase_filter(out, sr, lambda f: voicing_magnitude(f, s))
    out = out + synth

    # 4. downward expander
    tick("Downward expander", 0.65)
    out = downward_expander(out, sr, s.expander_thr_db, s.expander_ratio, s.expander_range_db)

    # 5. output stage
    mode = s.output_mode
    if mode in ("room_binaural", "room_stereo"):
        tick("Simulating room, horns and listener position", 0.72)
        out = render_room(out, sr, s)
    elif mode == "headphones_crossfeed":
        tick("Crossfeed", 0.8)
        out = crossfeed(micro_ambience(out, sr, s.ambience_db), sr, s.crossfeed_db)
    else:
        tick("Micro-ambience", 0.8)
        out = micro_ambience(out, sr, s.ambience_db)

    # 6. loudness match + limiter
    tick("Loudness match and limiter", 0.9)
    if s.loudness_match:
        ms = k_weighted_ms(out, sr)
        if ms > 1e-14:
            out = out * math.sqrt(ref_ms / ms)
    out = lookahead_limiter(out, sr, s.ceiling_db)
    tick("Done", 1.0)
    return out


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Horn-loaded flagship signature renderer (offline).")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--profile", choices=PROFILES, default="dragon_ideal")
    ap.add_argument("--mode", choices=OUTPUT_MODES, default=None)
    ap.add_argument("--preset", choices=list(ABSORPTION_PRESETS), default=None)
    ap.add_argument("--listener", nargs=2, type=float, metavar=("X", "Y"), help="listener position in metres")
    ap.add_argument("--settings", help="JSON settings file (overrides the profile)")
    ap.add_argument("--bits", type=int, choices=(16, 24, 32), default=24)
    args = ap.parse_args(argv)

    if args.settings:
        with open(args.settings, "r", encoding="utf-8") as fh:
            s = settings_from_json(fh.read())
    else:
        s = apply_profile(Settings(), args.profile)
    if args.mode:
        s.output_mode = args.mode
    if args.preset:
        s.room_preset = args.preset
    if args.listener:
        s.listener_x, s.listener_y = args.listener
    s.export_bits = args.bits

    data, sr = load_audio(args.input)

    def prog(msg, frac):
        print(f"[{int(frac * 100):3d}%] {msg}")

    out = process(data, sr, s, prog)
    save_audio(args.output, out, sr, bits=s.export_bits)
    print("Wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
