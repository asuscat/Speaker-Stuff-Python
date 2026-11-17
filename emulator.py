"""
offline "horn-loaded flagship" renderer & physical model
(inspired by the ESD Acoustic Super Dragon).

Combined script integrating high-precision DSP routing, active crossover, zero-phase EQ,
and spec-driven physical modeling of horns, field-coil drivers, and room acoustics,
featuring the new 'speaker_cabinet_reverb' mode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, fields

import numpy as np
from scipy import fft as sfft
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d, map_coordinates

C_SOUND = 343.0
C = 343.0
RHO0 = 1.204
P0 = 101325.0
BETA = 1.2            # nonlinearity coefficient of air, (gamma + 1) / 2
NU_AIR = 1.5e-5       # kinematic viscosity (m^2/s)
PRANDTL = 0.71
GAMMA = 1.4

OCT_CENTERS = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
OCT_EDGES = OCT_CENTERS[:-1] * math.sqrt(2.0)
AIR_DB_PER_M = np.array([0.0002, 0.0007, 0.002, 0.0034, 0.0057, 0.0115, 0.035, 0.12])
HEAD_RADIUS = 0.0875

# ---------------------------------------------------------------------------------------------------------------
# Materials & Diaphragms for Physical Modeling
# ---------------------------------------------------------------------------------------------------------------
MATERIALS = {
    "carbon_fiber": dict(E=110.0, rho=1600.0, eta=0.030, thick=6.0, rough=1.0, nu=0.30),
    "aluminum":     dict(E=70.0,  rho=2700.0, eta=0.002, thick=4.0, rough=1.0, nu=0.33),
    "fiberglass":   dict(E=25.0,  rho=1900.0, eta=0.015, thick=6.0, rough=1.1, nu=0.30),
    "birch_plywood": dict(E=10.0, rho=700.0,  eta=0.020, thick=18.0, rough=1.8, nu=0.30),
    "mdf":          dict(E=3.5,   rho=750.0,  eta=0.050, thick=18.0, rough=1.6, nu=0.30),
    "abs_plastic":  dict(E=2.3,   rho=1050.0, eta=0.040, thick=5.0, rough=1.2, nu=0.35),
}

DIAPHRAGMS = {
    "titanium":  dict(E=116.0, rho=4500.0, Q=15.0, peak_db=2.0),
    "beryllium": dict(E=287.0, rho=1850.0, Q=12.0, peak_db=1.0),
}
_KAPPA_BK = 38e3 * math.pi * 0.0254 / math.sqrt(287e9 / 1850.0)


@dataclass
class HornSpec:
    name: str
    range_hz: tuple
    throat_mm: float
    mouth_mm: float
    length_mm: float
    height_mm: float = 0.0
    material: str = "carbon_fiber"
    assumed: bool = False


@dataclass
class DriverSpec:
    name: str
    field_A: float
    flux_T: float
    total_flux_Wb: float
    sens_db: float
    power_w: float
    diaphragm_in: float
    diaphragm_mat: str
    z_ohm: float
    weight_kg: float
    height_mm: float
    diameter_mm: float
    xmax_mm: float
    bl_asym: float = 0.02
    note: str = ""


@dataclass
class Channel:
    horn: HornSpec
    driver: DriverSpec
    z: float


CHANNELS = [
    Channel(
        HornSpec("E20 sub-bass horn", (17, 140), 160, 1290, 2850, 2050),
        DriverSpec("S20 10in titanium (field coil)", 2.5, 2.4, 9.90e-3, 110.0, 150.0, 10.0, "titanium",
                   16.0, 140.0, 430.0, 300.0, xmax_mm=8.0),
        z=0.45),
    Channel(
        HornSpec("E100 low-range horn", (100, 800), 127, 550, 830),
        DriverSpec("S100 8in titanium (field coil)", 2.25, 2.4, 6.40e-3, 114.0, 80.0, 8.0, "titanium",
                   16.0, 105.0, 390.0, 270.0, xmax_mm=5.0),
        z=1.00),
    Channel(
        HornSpec("E500 mid-range horn", (250, 8000), 60, 430, 385),
        DriverSpec("S500 4in beryllium (field coil)", 1.8, 2.4, 2.88e-3, 120.0, 30.0, 4.0, "beryllium",
                   16.0, 80.0, 340.0, 245.0, xmax_mm=2.0,
                   note="spec: 105 dB bare / 120 dB horn loaded"),
        z=1.50),
    Channel(
        HornSpec("S2000 high-range horn (ASSUMED geometry)", (2000, 20000), 31, 300, 150, assumed=True),
        DriverSpec("S2000 1.75in beryllium (field coil)", 1.6, 2.4, 0.89e-3, 112.0, 20.0, 1.75, "beryllium",
                   8.0, 60.0, 330.0, 215.0, xmax_mm=1.0),
        z=1.90),
    Channel(
        HornSpec("S8000 tweeter horn (ASSUMED geometry)", (8000, 52000), 18, 150, 80, assumed=True),
        DriverSpec("S8000 1in beryllium (field coil)", 1.5, 2.4, 0.39e-3, 110.0, 5.0, 1.0, "beryllium",
                   8.0, 45.0, 290.0, 190.0, xmax_mm=0.6),
        z=2.15),
]

ABSORPTION_PRESETS = {
    "Living room (typical)": dict(
        front=np.array([.12, .10, .07, .05, .04, .04, .05, .06]),
        back=np.array([.12, .10, .07, .05, .04, .04, .05, .06]),
        side=np.array([.12, .10, .07, .05, .04, .04, .05, .06]),
        floor=np.array([.08, .09, .20, .26, .27, .37, .34, .30]),
        ceiling=np.array([.12, .10, .07, .05, .04, .04, .05, .06])),
    "Bare / hard room": dict(
        front=np.array([.01, .01, .02, .02, .02, .03, .04, .05]),
        back=np.array([.01, .01, .02, .02, .02, .03, .04, .05]),
        side=np.array([.01, .01, .02, .02, .02, .03, .04, .05]),
        floor=np.array([.01, .01, .02, .02, .03, .03, .03, .04]),
        ceiling=np.array([.01, .01, .02, .02, .02, .03, .04, .05])),
    "Treated listening room": dict(
        front=np.array([.20, .30, .45, .55, .60, .60, .60, .55]),
        back=np.array([.30, .55, .80, .90, .90, .85, .80, .75]),
        side=np.array([.20, .35, .55, .70, .75, .70, .65, .60]),
        floor=np.array([.10, .15, .30, .40, .45, .50, .50, .45]),
        ceiling=np.array([.25, .40, .60, .75, .80, .75, .70, .65])),
    "Dead (anechoic-like)": dict(
        front=np.array([.85, .90, .95, .97, .97, .97, .97, .97]),
        back=np.array([.85, .90, .95, .97, .97, .97, .97, .97]),
        side=np.array([.85, .90, .95, .97, .97, .97, .97, .97]),
        floor=np.array([.85, .90, .95, .97, .97, .97, .97, .97]),
        ceiling=np.array([.85, .90, .95, .97, .97, .97, .97, .97])),
    "ESD Reference Showroom": dict(
        front=np.array([.28, .42, .62, .72, .75, .72, .68, .65]),
        back=np.array([.38, .65, .85, .92, .92, .88, .85, .80]),
        side=np.array([.22, .38, .58, .70, .74, .72, .68, .62]),
        floor=np.array([.12, .18, .35, .52, .60, .68, .72, .75]),
        ceiling=np.array([.30, .48, .68, .80, .82, .78, .74, .70])),
    "Nearfield Studio Desk": dict(
        front=np.array([.30, .45, .65, .75, .78, .75, .70, .65]),
        back=np.array([.35, .55, .75, .85, .85, .80, .75, .70]),
        side=np.array([.25, .40, .60, .72, .75, .72, .68, .62]),
        floor=np.array([.10, .15, .30, .45, .55, .62, .65, .65]),
        ceiling=np.array([.28, .45, .65, .78, .80, .76, .72, .68])),
    "Cozy Bedroom (Carpet + Bed)": dict(
        front=np.array([.16, .22, .32, .42, .48, .52, .56, .60]),
        back=np.array([.28, .40, .55, .68, .72, .75, .75, .70]),
        side=np.array([.15, .20, .30, .40, .45, .50, .55, .55]),
        floor=np.array([.10, .16, .32, .52, .65, .75, .80, .82]),
        ceiling=np.array([.12, .14, .18, .22, .26, .30, .35, .40])),
}

OUTPUT_MODES = ["speakers", "headphones_crossfeed", "room_binaural", "room_stereo", "speaker_cabinet_reverb"]
PROFILES = ["dragon_ideal", "kef_q1_meta"]


# ---------------------------------------------------------------------------------------------------------------
# Settings Dataclass
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Settings:
    profile: str = "dragon_ideal"
    output_mode: str = "room_binaural"
    loudness_match: bool = True
    ceiling_db: float = -1.0
    export_bits: int = 24

    # 5-way crossover
    xover_hz: list = field(default_factory=lambda: [70.0, 350.0, 2000.0, 8000.0])
    band_trim_db: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    # Voicing EQ (zero-phase)
    hp_hz: float = 20.0
    hp_order: int = 4
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

    # Dynamics
    attack_db: float = 0.0
    sustain_db: float = 0.0
    band_transient_weight: list = field(default_factory=lambda: [0.2, 0.6, 1.0, 1.0, 0.7])
    expander_thr_db: float = -60.0
    expander_ratio: float = 1.4
    expander_range_db: float = 0.0

    # Saturation
    h2_percent: float = 0.3

    # Bass
    bass_mode: str = "subharmonic"
    bass_amount: float = 0.35

    # Speakers / Headphones (no room)
    ambience_db: float = -22.0
    crossfeed_db: float = -5.0

    # Room simulation & Physical additions
    room_preset: str = "Treated listening room"
    room_w: float = 8.0
    room_l: float = 10.0
    room_h: float = 3.6
    spk_spacing: float = 3.0
    spk_front_dist: float = 1.8
    aim_y: float = 5.5
    horn_scale: float = 1.0
    use_omni: bool = False
    reflection_order: int = 6
    room_seed: int = 7
    listener_x: float = 4.0
    listener_y: float = 5.5
    listener_z: float = 1.15
    auto_face: bool = True
    yaw_deg: float = 0.0

    # Physical layer options
    physics_enable: bool = True
    horn_material: str = "carbon_fiber"
    drive_db: float = 0.0
    field_current_pct: list = field(default_factory=lambda: [100.0, 100.0, 100.0, 100.0, 100.0])

    def aim_xy(self):
        return (self.room_w / 2.0, self.aim_y)


def apply_profile(s: Settings, profile: str) -> Settings:
    s.profile = profile
    if profile == "kef_q1_meta":
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
# Audio I/O Helpers
# ---------------------------------------------------------------------------------------------------------------
def load_audio(path: str):
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
        data = data + (rng.random(data.shape) - rng.random(data.shape)) * lsb
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
# DSP Math & Filtering Helpers
# ---------------------------------------------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def one_pole(x, tau_s, sr):
    a = math.exp(-1.0 / (max(tau_s, 1e-6) * sr))
    return signal.lfilter([1.0 - a], [1.0, -a], x, axis=0)


def smooth_zero(x, tau_s, sr):
    y = one_pole(x, tau_s, sr)
    return one_pole(y[::-1], tau_s, sr)[::-1]


def lp_mag(f, fc, n):
    return 1.0 / (1.0 + (np.maximum(f, 0.0) / fc) ** (2 * n))


def split_masks(f, edges, n):
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
    def __init__(self, x, sr, pad_s=0.6):
        self.x = np.asarray(x, dtype=np.float64)
        self.sr = sr
        self.n = self.x.shape[0]
        self.pad = int(pad_s * sr)
        self.nfft = sfft.next_fast_len(self.n + 2 * self.pad)
        self.f = sfft.rfftfreq(self.nfft, 1.0 / sr)
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
# Physical Modeling & Horn Geometry Helpers
# ---------------------------------------------------------------------------------------------------------------
def throat_area(h: HornSpec) -> float:
    return math.pi * (h.throat_mm * 1e-3 / 2.0) ** 2


def mouth_area(h: HornSpec) -> float:
    return math.pi * (h.mouth_mm * 1e-3 / 2.0) ** 2


def flare_constant(h: HornSpec) -> float:
    return math.log(mouth_area(h) / throat_area(h)) / (h.length_mm * 1e-3)


def exp_cutoff_hz(h: HornSpec) -> float:
    return C * flare_constant(h) / (4.0 * math.pi)


def flare_half_angle_deg(h: HornSpec) -> float:
    return math.degrees(math.atan((h.mouth_mm - h.throat_mm) / (2.0 * h.length_mm)))


def virtual_apex_behind_mouth_m(h: HornSpec) -> float:
    ha = math.radians(flare_half_angle_deg(h))
    return (h.mouth_mm * 1e-3 / 2.0) / max(math.tan(ha), 1e-6)


def effective_diaphragm_dia_mm(d: DriverSpec) -> float:
    return 0.85 * d.diaphragm_in * 25.4


def compression_ratio(ch: Channel) -> float:
    return (effective_diaphragm_dia_mm(ch.driver) / ch.horn.throat_mm) ** 2


def d_eff_m(h: HornSpec) -> float:
    m = flare_constant(h)
    L = h.length_mm * 1e-3
    return (2.0 / m) * (1.0 - math.exp(-m * L / 2.0))


def diaphragm_breakup_hz(d: DriverSpec) -> float:
    mat = DIAPHRAGMS[d.diaphragm_mat]
    cm = math.sqrt(mat["E"] * 1e9 / mat["rho"])
    return _KAPPA_BK * cm / (math.pi * d.diaphragm_in * 0.0254)


def _smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def coverage_deg(ch_index: int, fc: float, scale: float = 1.0) -> float:
    h = CHANNELS[ch_index].horn
    lam = C / fc
    x = 0.705 * lam / (h.mouth_mm * 1e-3)
    piston = math.degrees(math.asin(min(x, 1.0)))
    half = max(flare_half_angle_deg(h), piston)
    half *= 1.0 + 1.2 * _smoothstep(0.8, 1.4, x)
    return min(2.0 * half * scale, 360.0)


def horn_gain(dirs, axis, H, V, f_cd, fc, back_db=30.0):
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
    g = horn_gain(_SPHERE, np.array([1.0, 0.0, 0.0]), H, V, f_cd, fc)
    return float(np.mean(g ** 2))


def directivity_q(ch_index: int, fc: float, scale: float = 1.0) -> float:
    cov = coverage_deg(ch_index, fc, scale)
    return 1.0 / max(horn_mean_g2(cov, cov, 1.0, fc), 1e-4)


def band_edges(xover_hz, k):
    lo = 20.0 if k == 0 else float(xover_hz[k - 1])
    hi = 20000.0 if k >= len(xover_hz) else float(xover_hz[k])
    return lo, hi


def band_mid_hz(xover_hz, k):
    lo, hi = band_edges(xover_hz, k)
    return math.sqrt(lo * hi)


def throat_peak_pa_at_full_scale(ch_index, s):
    ch = CHANNELS[ch_index]
    p_fs = ch.driver.power_w * 10.0 ** (s.drive_db / 10.0)
    fmid = band_mid_hz(s.xover_hz, ch_index)
    q = directivity_q(ch_index, fmid, s.horn_scale)
    i1 = 10.0 ** ((ch.driver.sens_db - 120.0) / 10.0)
    eta = min(0.85, 4.0 * math.pi * i1 / q)
    w_ac = eta * p_fs
    p_rms = math.sqrt(w_ac / throat_area(ch.horn) * RHO0 * C)
    return math.sqrt(2.0) * p_rms, eta, q, p_fs


def mouth_ripple_db(f, h: HornSpec, r_throat=0.6):
    f = np.maximum(f, 1e-3)
    k = 2.0 * math.pi * f / C
    a = h.mouth_mm * 1e-3 / 2.0
    rm = np.exp(-0.55 * (k * a) ** 2)
    r = np.clip(rm * r_throat, 0.0, 0.95)
    leff = h.length_mm * 1e-3 + 0.6 * a
    den = np.abs(1.0 - r * np.exp(-2j * k * leff))
    mag = np.sqrt(np.maximum(1.0 - r * r, 1e-9)) / np.maximum(den, 1e-6)
    return 20.0 * np.log10(mag)


def viscothermal_db(f, h: HornSpec, material: str):
    mat = MATERIALS.get(material, MATERIALS["carbon_fiber"])
    m = flare_constant(h)
    L = h.length_mm * 1e-3
    rt = h.throat_mm * 1e-3 / 2.0
    path = (2.0 / (m * rt)) * (1.0 - math.exp(-m * L / 2.0))
    w = 2.0 * math.pi * np.maximum(f, 1.0)
    alpha_np = np.sqrt(w * NU_AIR / 2.0) * (1.0 + (GAMMA - 1.0) / math.sqrt(PRANDTL)) / C * path * mat["rough"]
    return -8.686 * alpha_np


def wall_coloration_db(f, h: HornSpec, material: str):
    mat = MATERIALS.get(material, MATERIALS["carbon_fiber"])
    c_l = math.sqrt(mat["E"] * 1e9 / (mat["rho"] * (1.0 - mat["nu"] ** 2)))
    rho_s = mat["rho"] * mat["thick"] * 1e-3
    q = 1.0 / mat["eta"]
    m = flare_constant(h)
    rt = h.throat_mm * 1e-3 / 2.0
    L = h.length_mm * 1e-3
    mag = np.ones_like(np.asarray(f, dtype=float))
    for frac in (0.25, 0.6, 1.0):
        a = rt * math.exp(m * frac * L / 2.0)
        fr = c_l / (2.0 * math.pi * a)
        amp = min(0.35, 0.1 * q * RHO0 * C / (2.0 * math.pi * fr * rho_s))
        lor = 1.0 / (1.0 + (q * (f / fr - fr / np.maximum(f, 1e-3))) ** 2)
        mag = mag + amp * lor
    return 20.0 * np.log10(mag)


def diaphragm_breakup_db(f, d: DriverSpec):
    mat = DIAPHRAGMS[d.diaphragm_mat]
    fb = diaphragm_breakup_hz(d)
    out = np.zeros_like(np.asarray(f, dtype=float))
    for mult, scale in ((1.0, 1.0), (1.9, 0.5)):
        fr = fb * mult
        lor = 1.0 / (1.0 + (mat["Q"] * (f / fr - fr / np.maximum(f, 1e-3))) ** 2)
        out += mat["peak_db"] * scale * lor
    return out


def field_coil_gain_db(pct: float) -> float:
    r = max(pct, 1.0) / 100.0
    ik = 0.6
    return 20.0 * math.log10(math.tanh(r / ik) / math.tanh(1.0 / ik))


def linear_response(f, k, s):
    ch = CHANNELS[k]
    material = s.horn_material
    db = mouth_ripple_db(f, ch.horn)
    db = np.clip(db, -2.5, 2.5)
    db = db + viscothermal_db(f, ch.horn, material)
    db = db + wall_coloration_db(f, ch.horn, material)
    db = db + diaphragm_breakup_db(f, ch.driver)
    pct = s.field_current_pct[k] if k < len(s.field_current_pct) else 100.0
    db = db + field_coil_gain_db(pct)
    return 10.0 ** (db / 20.0)


# ---------------------------------------------------------------------------------------------------------------
# Time-Domain Dynamics & Nonlinearity
# ---------------------------------------------------------------------------------------------------------------
def transient_shaper(x, sr, attack_db, sustain_db, fast_ms=1.0, slow_ms=40.0):
    if abs(attack_db) < 1e-3 and abs(sustain_db) < 1e-3:
        return x
    env = np.max(np.abs(x), axis=1)
    fast = one_pole(env, fast_ms * 1e-3, sr)
    slow = one_pole(env, slow_ms * 1e-3, sr)
    eps = 1e-6
    d = 20.0 * np.log10((fast + eps) / (slow + eps))
    level_db = 20.0 * np.log10(slow + eps)
    gate = np.clip((level_db + 60.0) / 15.0, 0.0, 1.0)
    gdb = (attack_db * np.clip(d / 6.0, 0.0, 1.0) + sustain_db * np.clip(-d / 6.0, 0.0, 1.0)) * gate
    gdb = smooth_zero(gdb, 0.5e-3, sr)
    shift = int(0.001 * sr)
    if shift > 0:
        gdb = np.concatenate([gdb[shift:], np.full(shift, gdb[-1])])
    return x * (10.0 ** (gdb / 20.0))[:, None]


def downward_expander(x, sr, thr_db, ratio, range_db):
    if ratio <= 1.0 or range_db <= 0.0:
        return x
    power = one_pole(np.max(x * x, axis=1), 30e-3, sr)
    lvl = 10.0 * np.log10(power + 1e-18)
    gdb = np.where(lvl < thr_db, (ratio - 1.0) * (lvl - thr_db), 0.0)
    gdb = np.maximum(gdb, -range_db)
    gdb = smooth_zero(gdb, 20e-3, sr)
    look = int(0.008 * sr)
    if look > 0:
        gdb = np.concatenate([gdb[look:], np.full(look, gdb[-1])])
    return x * (10.0 ** (gdb / 20.0))[:, None]


def add_second_harmonic(x, sr, percent, ref_amp=0.25):
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


def thermal_compression(band, sr, k, s):
    p_ratio = 10.0 ** (s.drive_db / 10.0)
    if p_ratio < 1e-4:
        return band
    power = np.mean(band * band, axis=1)
    a = math.exp(-1.0 / (8.0 * sr))
    avg = signal.lfilter([1.0 - a], [1.0, -a], power)
    d_t = 150.0 * p_ratio * avg
    gain = 1.0 / (1.0 + 0.0039 * d_t)
    return band * gain[:, None]


def air_time_warp_s(k, s):
    ch = CHANNELS[k]
    p_pk, _eta, _q, _p = throat_peak_pa_at_full_scale(k, s)
    return BETA * p_pk * d_eff_m(ch.horn) / (RHO0 * C ** 3)


def air_nonlinearity(band, sr, k, s):
    gam = air_time_warp_s(k, s) * sr
    if abs(gam) < 1e-4:
        return band
    n = band.shape[0]
    idx = np.arange(n, dtype=np.float64)
    out = np.empty_like(band)
    for ch in range(band.shape[1]):
        x = band[:, ch]
        coords = np.clip(idx + gam * x, 0, n - 1)
        out[:, ch] = map_coordinates(x, [coords], order=3, mode="nearest")
    return out


def motor_nonlinearity(band, sr, k, s):
    ch = CHANNELS[k]
    p_pk, _eta, _q, _p = throat_peak_pa_at_full_scale(k, s)
    s_t = throat_area(ch.horn)
    s_d = math.pi * (effective_diaphragm_dia_mm(ch.driver) * 1e-3 / 2.0) ** 2
    v_d = (s_t / s_d) * p_pk / (RHO0 * C)
    fc = float(ch.horn.range_hz[0])
    lo, _hi = band_edges(s.xover_hz, k)
    f_ref = max(lo, 15.0)
    hexc = math.sqrt(1.0 + (fc / f_ref) ** 4)
    xi_est = v_d / (2.0 * math.pi * f_ref) * hexc
    x_bl = 2.0 * ch.driver.xmax_mm * 1e-3
    a2 = 0.35
    if a2 * (xi_est / x_bl) ** 2 < 5e-5 and ch.driver.bl_asym * xi_est / x_bl < 5e-5:
        return band
    n = band.shape[0]
    nfft = sfft.next_fast_len(n)
    f = sfft.rfftfreq(nfft, 1.0 / sr)
    mag = v_d / (2.0 * math.pi * np.maximum(f, 5.0)) * np.sqrt(1.0 + (fc / np.maximum(f, 5.0)) ** 4)
    out = np.empty_like(band)
    for chn in range(band.shape[1]):
        spec = sfft.rfft(band[:, chn], n=nfft)
        xi = sfft.irfft(spec * mag, n=nfft)[:n]
        u = xi / x_bl
        g = 1.0 - ch.driver.bl_asym * u - a2 * u * u
        out[:, chn] = band[:, chn] * np.clip(g, 0.2, 1.2)
    return out


def nonlinear_stage(band, sr, k, s):
    if not s.physics_enable:
        return band
    band = thermal_compression(band, sr, k, s)
    band = motor_nonlinearity(band, sr, k, s)
    band = air_nonlinearity(band, sr, k, s)
    return band


def micro_ambience(x, sr, level_db):
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
    g = uniform_filter1d(g, size=2 * la + 1, mode="nearest")
    y = x * g[:, None]
    return np.clip(y, -ceiling, ceiling)


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
# Scene & Room Acoustics Setup
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Scene:
    src: list
    axis: list
    listener: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    yaw_deg: float


def build_scene(s: Settings) -> Scene:
    cx = s.room_w / 2.0
    xs_sub = [cx - s.spk_spacing / 2.0, cx + s.spk_spacing / 2.0]
    xs_tower = [cx - s.spk_spacing / 4.0, cx + s.spk_spacing / 4.0]
    
    aim = np.array([s.aim_xy()[0], s.aim_xy()[1], s.listener_z])
    src, axis = [], []
    for ch in range(2):
        srcs, axes = [], []
        for i, ch_obj in enumerate(CHANNELS):
            h = ch_obj.horn
            if i == 0:
                p = np.array([xs_sub[ch], s.spk_front_dist, h.height_mm * 1e-3 if h.height_mm > 0 else ch_obj.z])
            else:
                p = np.array([xs_tower[ch], s.spk_front_dist, ch_obj.z])
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


def _surface_areas(s):
    W, L, H = s.room_w, s.room_l, s.room_h
    return dict(front=W * H, back=W * H, left=L * H, right=L * H, floor=W * L, ceiling=W * L)


def _absorption(s):
    p = ABSORPTION_PRESETS.get(s.room_preset, ABSORPTION_PRESETS["Treated listening room"])
    return dict(front=p["front"], back=p["back"], left=p["side"], right=p["side"],
                floor=p["floor"], ceiling=p["ceiling"])


def room_acoustics(s):
    ab, area = _absorption(s), _surface_areas(s)
    # Sum absorption * area across surfaces for each frequency band, then take the mean across bands for an overall T60 estimate
    total_absorption_per_band = sum(ab[k] * area[k] for k in area)
    total_area = sum(area.values())
    
    # Compute T60 per band or take a representative broadband mean for room acoustics
    V = s.room_w * s.room_l * s.room_h
    
    # Ensure T60 is an array corresponding to OCT_CENTERS
    T60_bands = []
    R_bands = []
    for b_idx in range(len(OCT_CENTERS)):
        a_band = max(float(total_absorption_per_band[b_idx]), 1e-6)
        t60_b = float(np.clip(0.161 * V / a_band, 0.12, 3.5))
        T60_bands.append(t60_b)
        
        abar_b = np.clip(a_band / total_area, 1e-3, 0.98)
        R_bands.append(total_area * abar_b / (1.0 - abar_b))
        
    return dict(
        A=total_absorption_per_band, 
        S=total_area, 
        V=V, 
        T60=np.array(T60_bands), 
        R=np.array(R_bands), 
        ab=ab, 
        area=area
    )


_KERNELS: dict = {}


def head_shadow_kernel(theta_deg, sr, taps=48):
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
    k = np.abs(m)
    hi_pos = (k + 1) // 2
    lo_pos = k // 2
    lo = np.where(m > 0, lo_pos, hi_pos)
    hi = np.where(m > 0, hi_pos, lo_pos)
    return np.where(m == 0, 0, lo), np.where(m == 0, 0, hi)


def _pattern(s, dirs, axes_flip, horn, fc, ch_idx):
    if s.use_omni:
        return np.ones(dirs.shape[0])
    out = np.empty(dirs.shape[0])
    keys = np.round(axes_flip, 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    for u in range(uniq.shape[0]):
        sel = inv == u
        cov = coverage_deg(ch_idx, fc, s.horn_scale)
        out[sel] = horn_gain(dirs[sel], uniq[u], cov, cov, 1.0, fc)
    return out


def driver_for_band(b, xover_hz):
    return int(np.sum(np.array(xover_hz) < OCT_CENTERS[b]))


def _early_ir(s, scene, ch, b, sr, n_ir, pre, t_mix, mode):
    ab = _absorption(s)
    beta = {k: math.sqrt(max(1.0 - float(v[b]), 0.0)) for k, v in ab.items()}
    di = driver_for_band(b, s.xover_hz)
    horn_spec = CHANNELS[di].horn
    src = scene.src[ch][di]
    axis = scene.axis[ch][di]
    fc = float(OCT_CENTERS[b])

    # If speaker_cabinet_reverb mode is active, shorten spatial distances to mimic close-mic/cabinet reflections
    distance_scale = 0.35 if mode == "speaker_cabinet_reverb" else 1.0

    mx, my, mz = _image_set(s, int(s.reflection_order))
    ix = _img_coord(mx, s.room_w * distance_scale, src[0] * distance_scale)
    iy = _img_coord(my, s.room_l * distance_scale, src[1] * distance_scale)
    iz = _img_coord(mz, s.room_h * distance_scale, src[2] * distance_scale)
    lo_x, hi_x = _hits(mx)
    lo_y, hi_y = _hits(my)
    lo_z, hi_z = _hits(mz)
    
    # Extra horn wall diffusion & cavity resonance damping if in cabinet mode
    cavity_boost = 1.35 if mode == "speaker_cabinet_reverb" else 1.0
    refl = (beta["left"] ** lo_x) * (beta["right"] ** hi_x) * (beta["front"] ** lo_y) * (beta["back"] ** hi_y) \
        * (beta["floor"] ** lo_z) * (beta["ceiling"] ** hi_z) * cavity_boost

    img = np.stack([ix, iy, iz], axis=1)
    flip = np.stack([np.where(mx & 1, -1.0, 1.0), np.where(my & 1, -1.0, 1.0), np.where(mz & 1, -1.0, 1.0)], axis=1)

    ears = np.zeros((2, n_ir))
    lis = scene.listener

    if mode in ("room_stereo", "speaker_cabinet_reverb"):
        rt, fw = scene.right, scene.forward
        mic_pos = [lis - 0.085 * rt, lis + 0.085 * rt]
        mic_ax = [math.cos(math.radians(55)) * fw - math.sin(math.radians(55)) * rt,
                  math.cos(math.radians(55)) * fw + math.sin(math.radians(55)) * rt]
        for e in range(2):
            vec = img - mic_pos[e]
            r = np.maximum(np.linalg.norm(vec, axis=1), 0.2)
            dirs_to_mic = -vec / r[:, None]
            g_src = _pattern(s, dirs_to_mic, axis * flip, horn_spec, fc, di)
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

    vec = img - lis
    r = np.maximum(np.linalg.norm(vec, axis=1), 0.2)
    to_head = vec / r[:, None]
    dirs_src_to_rec = -to_head
    g_src = _pattern(s, dirs_src_to_rec, axis * flip, horn_spec, fc, di)
    amp = g_src * refl / r * 10.0 ** (-AIR_DB_PER_M[b] * r / 20.0)
    t = r / C_SOUND
    bounce_order = np.abs(mx) + np.abs(my) + np.abs(mz)
    rng_diff = np.random.default_rng(s.room_seed + ch * 17 + b)
    jitter = np.where(bounce_order > 0, rng_diff.uniform(-0.0018, 0.0018, size=t.shape), 0.0)
    t = np.maximum(t + jitter, 0.0)

    lateral = np.arcsin(np.clip(to_head @ scene.right, -1.0, 1.0))
    itd = (HEAD_RADIUS / C_SOUND) * (np.abs(lateral) + np.sin(np.abs(lateral))) * np.sign(lateral)
    phi = math.radians(10.0)
    ear_axis = [-math.cos(phi) * scene.right - math.sin(phi) * scene.forward,
                math.cos(phi) * scene.right - math.sin(phi) * scene.forward]
    ok = (t < t_mix) & (amp > 1e-5)
    idx = np.nonzero(ok)[0]
    K = 48
    for e in range(2):
        theta = np.degrees(np.arccos(np.clip(to_head @ ear_axis[e], -1.0, 1.0)))
        sign = -1.0 if e == 1 else 1.0
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


def _late_tail(s, ch, b, sr, n_tail, rng, acoustics, mode):
    di = driver_for_band(b, s.xover_hz)
    horn_spec = CHANNELS[di].horn
    fc = float(OCT_CENTERS[b])
    if s.use_omni:
        g2 = 1.0
    else:
        cov = coverage_deg(di, fc, s.horn_scale)
        g2 = horn_mean_g2(cov, cov, 1.0, fc)
        
    t60_arr = np.atleast_1d(acoustics["T60"])
    T60 = float(t60_arr[b] if b < len(t60_arr) else t60_arr[0])
    
    if mode == "speaker_cabinet_reverb":
        T60 *= 0.25  # Tighter, denser cabinet/horn internal reflection tail
        
    R = float(acoustics["R"]) if np.isscalar(acoustics["R"]) else float(acoustics["R"][b])
    p_rev2 = 16.0 * math.pi * g2 / max(R, 1e-3)
    t_mix = 0.08
    energy = p_rev2 * math.exp(-13.8155 * t_mix / max(T60, 1e-3))
    t = np.arange(n_tail) / sr
    env = np.exp(-6.9078 * t / max(T60, 1e-3)) * np.clip(t / 0.02, 0.0, 1.0)
    rho_tab = np.array([0.95, 0.80, 0.50, 0.25, 0.10, 0.05, 0.0, 0.0])
    rho = rho_tab[b] * (0.6 if mode in ("room_stereo", "speaker_cabinet_reverb") else 1.0)
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
    mode = mode or s.output_mode
    scene = build_scene(s)
    ac = room_acoustics(s)
    pre = int(0.05 * sr)
    t_mix = 0.08
    tail_s = min(float(np.max(ac["T60"])) * 1.05, 2.5)
    if mode == "speaker_cabinet_reverb":
        tail_s = 0.65  # Shorten overall tail window for intimate close-cabinet acoustic reflections
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
                     for ch in range(2) for di in range(len(CHANNELS)))
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
            
    # If speaker_cabinet_reverb is active, blend in subtle cabinet throat flutter echoes and horn diffraction resonance
    if s.output_mode == "speaker_cabinet_reverb":
        b_res, a_res = signal.iirpeak(1250.0, Q=2.5, fs=sr)
        cabinet_resonance = signal.lfilter(b_res, a_res, out, axis=0) * 0.15
        out = out + cabinet_resonance
        
    return out


# ---------------------------------------------------------------------------------------------------------------
# Complete Processing Chain
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

    # 1. Zero-phase 5-way split, per-band physical shaping & transient processing
    tick("Splitting into 5 bands & applying physical model", 0.05)
    dom = ZeroPhaseDomain(x, sr)
    masks = split_masks(dom.f, s.xover_hz, 2)
    out = np.zeros_like(x)
    for k, m in enumerate(masks):
        band = dom.apply(m)
        w = s.band_transient_weight[k] if k < len(s.band_transient_weight) else 1.0
        band = transient_shaper(band, sr, s.attack_db * w, s.sustain_db * w)
        
        # Apply physical nonlinearities & linear deviations per band
        lin_mag = linear_response(dom.f, k, s)
        band = zero_phase_filter(band, sr, lambda f: lin_mag)
        band = nonlinear_stage(band, sr, k, s)

        if k >= 2:
            band = add_second_harmonic(band, sr, s.h2_percent)
        trim = s.band_trim_db[k] if k < len(s.band_trim_db) else 0.0
        out += band * (10.0 ** (trim / 20.0))
        tick(f"Band {k + 1}/5 processed", 0.05 + 0.08 * (k + 1))
    del dom

    # 2. Bass synthesis
    tick("Bass synthesis", 0.5)
    if s.bass_mode == "subharmonic":
        synth = subharmonic_synth(out, sr, s.bass_amount)
    elif s.bass_mode == "psycho":
        synth = psycho_bass(out, sr, s.bass_amount)
    else:
        synth = np.zeros_like(out)

    # 3. Voicing EQ (zero-phase)
    tick("Voicing EQ", 0.58)
    out = zero_phase_filter(out, sr, lambda f: voicing_magnitude(f, s))
    out = out + synth

    # 4. Downward expander
    tick("Downward expander", 0.65)
    out = downward_expander(out, sr, s.expander_thr_db, s.expander_ratio, s.expander_range_db)

    # 5. Output stage / Room simulation
    mode = s.output_mode
    if mode in ("room_binaural", "room_stereo", "speaker_cabinet_reverb"):
        tick("Simulating multi-horn cabinet structure & room acoustics", 0.72)
        out = render_room(out, sr, s)
    elif mode == "headphones_crossfeed":
        tick("Crossfeed", 0.8)
        out = crossfeed(micro_ambience(out, sr, s.ambience_db), sr, s.crossfeed_db)
    else:
        tick("Micro-ambience", 0.8)
        out = micro_ambience(out, sr, s.ambience_db)

    # 6. Loudness match + limiter
    tick("Loudness match and limiter", 0.9)
    if s.loudness_match:
        ms = k_weighted_ms(out, sr)
        if ms > 1e-14:
            out = out * math.sqrt(ref_ms / ms)
    out = lookahead_limiter(out, sr, s.ceiling_db)
    tick("Done", 1.0)
    return out


# ---------------------------------------------------------------------------------------------------------------
# CLI & Entrypoint
# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Horn-loaded flagship signature renderer & physical model (offline).")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--profile", choices=PROFILES, default="dragon_ideal")
    ap.add_argument("--mode", choices=OUTPUT_MODES, default=None)
    ap.add_argument("--preset", choices=list(ABSORPTION_PRESETS), default=None)
    ap.add_argument("--listener", nargs=2, type=float, metavar=("X", "Y"), help="listener position in metres")
    ap.add_argument("--settings", help="JSON settings file (overrides the profile)")
    ap.add_argument("--bits", type=int, choices=(16, 24, 32), default=24)
    ap.add_argument("--no-physics", action="store_true", help="bypass physical nonlinearities and deviations")
    ap.add_argument("--material", choices=list(MATERIALS), default="carbon_fiber", help="horn shell material")
    ap.add_argument("--drive-db", type=float, default=0.0, help="drive level offset in dB for physical stress")
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
    s.physics_enable = not args.no_physics
    s.horn_material = args.material
    s.drive_db = args.drive_db

    data, sr = load_audio(args.input)

    def prog(msg, frac):
        print(f"[{int(frac * 100):3d}%] {msg}")

    out = process(data, sr, s, prog)
    save_audio(args.output, out, sr, bits=s.export_bits)
    print("Wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())