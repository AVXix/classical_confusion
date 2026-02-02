"""MIDI feature extraction utilities.

This module extracts a fixed-length feature vector from a MIDI file using
`pretty_midi`, designed to be concatenated with audio embeddings.

Features included (fixed dim):
- Tempo curve (sampled) + tempo statistics
- Time signature summary
- Key/mode estimation (Krumhansl-Schmuckler)
- Pitch class histogram + pitch range statistics
- Chord histogram + chord change rate (simple triad detector)
- Note duration statistics
- Velocity statistics

The output is intentionally lightweight and robust to missing metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MidiFeatureConfig:
    tempo_samples: int = 32
    chord_vocab: tuple[str, ...] = tuple(
        [f"{pc}{q}" for q in ("maj", "min") for pc in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")]
        + ["other"]
    )


_PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _safe_stats(x: np.ndarray) -> np.ndarray:
    """Return [mean, std, median, min, max, iqr]."""
    if x.size == 0:
        return np.zeros((6,), dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((6,), dtype=np.float32)
    q25, q75 = np.percentile(x, [25, 75])
    return np.array(
        [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.median(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(q75 - q25),
        ],
        dtype=np.float32,
    )


def _normalize_hist(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    s = float(np.sum(x))
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return (x / s).astype(np.float32)


def _estimate_key_mode_from_pc(pc_hist: np.ndarray) -> tuple[int, int, float]:
    """Estimate key and mode from a 12-d pitch class histogram.

    Returns:
        key_pc: 0..11 (C..B)
        mode: 0=major, 1=minor
        strength: correlation score for best match
    """
    # Krumhansl-Schmuckler key profiles (normalized)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

    pc = _normalize_hist(pc_hist)
    if float(np.sum(pc)) <= 0:
        return 0, 0, 0.0

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a - np.mean(a)
        b = b - np.mean(b)
        da = float(np.linalg.norm(a))
        db = float(np.linalg.norm(b))
        if da == 0.0 or db == 0.0:
            return 0.0
        return float(np.dot(a, b) / (da * db))

    best_score = -1e9
    best_pc = 0
    best_mode = 0
    for shift in range(12):
        maj_s = corr(np.roll(major, shift), pc)
        if maj_s > best_score:
            best_score = maj_s
            best_pc = shift
            best_mode = 0
        min_s = corr(np.roll(minor, shift), pc)
        if min_s > best_score:
            best_score = min_s
            best_pc = shift
            best_mode = 1

    return int(best_pc), int(best_mode), float(best_score)


def _triad_chord_id(pc_set: set[int]) -> int:
    """Map a pitch-class set to a chord id (24 triads + other).

    Output ids: 0..23 correspond to [Cmaj..Bmaj, Cmin..Bmin] interleaved by mode.
    24 is "other".
    """
    if not pc_set:
        return 24

    pcs = set(int(p) % 12 for p in pc_set)

    # major triad: root, root+4, root+7
    for root in range(12):
        if {root, (root + 4) % 12, (root + 7) % 12}.issubset(pcs):
            return root  # 0..11

    # minor triad: root, root+3, root+7
    for root in range(12):
        if {root, (root + 3) % 12, (root + 7) % 12}.issubset(pcs):
            return 12 + root  # 12..23

    return 24


def extract_midi_feature_vector(midi_path: str | Path, *, cfg: MidiFeatureConfig | None = None) -> np.ndarray:
    """Extract a fixed-length MIDI feature vector from a .midi/.mid file."""
    cfg = cfg or MidiFeatureConfig()
    midi_path = Path(midi_path)

    import pretty_midi  # local import to keep base install lighter

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        # Corrupt/unparseable MIDI -> return all zeros.
        return np.zeros((126,), dtype=np.float32)

    # Gather all notes across instruments
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)

    # If there are no notes, return zeros (but keep dim stable)
    if not notes:
        return np.zeros((126,), dtype=np.float32)

    # --- Tempo curve ---
    try:
        t_times, tempi = pm.get_tempo_changes()
        tempi = np.asarray(tempi, dtype=np.float32)
        # Sample tempo across [0, end]
        end_t = float(pm.get_end_time())
        if not np.isfinite(end_t) or end_t <= 0:
            end_t = float(max((n.end for n in notes), default=0.0))
        if end_t <= 0:
            tempo_curve = np.zeros((cfg.tempo_samples,), dtype=np.float32)
        else:
            grid = np.linspace(0.0, end_t, num=int(cfg.tempo_samples), dtype=np.float32)
            # piecewise-constant tempo between change times
            idx = np.searchsorted(t_times.astype(np.float32), grid, side="right") - 1
            idx = np.clip(idx, 0, max(0, len(tempi) - 1))
            tempo_curve = tempi[idx].astype(np.float32)
        tempo_stats = _safe_stats(tempi)[:4]  # mean,std,median,min (we add max separately below)
        tempo_min = float(np.min(tempi)) if tempi.size else 0.0
        tempo_max = float(np.max(tempi)) if tempi.size else 0.0
        tempo_extra = np.array([tempo_min, tempo_max, float(len(tempi))], dtype=np.float32)
    except Exception:
        tempo_curve = np.zeros((cfg.tempo_samples,), dtype=np.float32)
        tempo_stats = np.zeros((4,), dtype=np.float32)
        tempo_extra = np.zeros((3,), dtype=np.float32)

    tempo_vec = np.concatenate([tempo_curve, tempo_stats, tempo_extra], axis=0)  # 32 + 4 + 3 = 39
    # We'll later slice to 37 by dropping median/min duplication? Keep stable below.

    # --- Time signature ---
    ts_changes = getattr(pm, "time_signature_changes", []) or []
    # first time signature
    if ts_changes:
        ts0 = ts_changes[0]
        ts_num = float(getattr(ts0, "numerator", 0) or 0)
        ts_den = float(getattr(ts0, "denominator", 0) or 0)
    else:
        ts_num, ts_den = 0.0, 0.0

    common = [(2, 4), (3, 4), (4, 4), (6, 8), (12, 8), (5, 4)]
    ts_hist = np.zeros((len(common),), dtype=np.float32)
    for tsc in ts_changes:
        pair = (int(getattr(tsc, "numerator", 0) or 0), int(getattr(tsc, "denominator", 0) or 0))
        if pair in common:
            ts_hist[common.index(pair)] += 1.0
    ts_vec = np.concatenate(
        [
            np.array([ts_num, ts_den, float(len(ts_changes))], dtype=np.float32),
            _normalize_hist(ts_hist),
        ],
        axis=0,
    )  # 3 + 6 = 9

    # --- Pitch histogram + pitch stats ---
    pitches = np.array([n.pitch for n in notes], dtype=np.float32)
    pc = (pitches.astype(np.int32) % 12).astype(np.int32)
    pc_hist = np.bincount(pc, minlength=12).astype(np.float32)
    pc_hist_n = _normalize_hist(pc_hist)

    pitch_stats = np.array(
        [
            float(np.min(pitches)) if pitches.size else 0.0,
            float(np.max(pitches)) if pitches.size else 0.0,
            float(np.mean(pitches)) if pitches.size else 0.0,
            float(np.std(pitches)) if pitches.size else 0.0,
        ],
        dtype=np.float32,
    )
    pitch_vec = np.concatenate([pc_hist_n, pitch_stats], axis=0)  # 12 + 4 = 16

    # --- Key / mode ---
    key_pc, mode, strength = _estimate_key_mode_from_pc(pc_hist)
    key_mode_id = key_pc + (12 if mode == 1 else 0)
    key_onehot = np.zeros((24,), dtype=np.float32)
    key_onehot[key_mode_id] = 1.0
    key_vec = np.concatenate([key_onehot, np.array([float(strength)], dtype=np.float32)], axis=0)  # 25

    # --- Note durations ---
    durations = np.array([max(0.0, float(n.end) - float(n.start)) for n in notes], dtype=np.float32)
    dur_stats = _safe_stats(durations)
    dur_count = np.array([float(len(durations))], dtype=np.float32)
    dur_vec = np.concatenate([dur_stats[:6], dur_count], axis=0)  # 6 + 1 = 7

    # --- Velocity ---
    velocities = np.array([float(getattr(n, "velocity", 0) or 0) for n in notes], dtype=np.float32)
    vel_stats = _safe_stats(velocities)
    vel_vec = vel_stats  # 6

    # --- Chord progression (beat-wise) ---
    try:
        beat_times = pm.get_beats()
        beat_times = np.asarray(beat_times, dtype=np.float32)
        if beat_times.size < 2:
            # fallback: fixed windows
            end_t = float(pm.get_end_time())
            beat_times = np.linspace(0.0, end_t, num=33, dtype=np.float32)

        # Pre-build active pitch-classes per beat using note overlaps
        chord_ids: list[int] = []
        for b0, b1 in zip(beat_times[:-1], beat_times[1:]):
            pcs: set[int] = set()
            for n in notes:
                if float(n.end) <= float(b0) or float(n.start) >= float(b1):
                    continue
                pcs.add(int(n.pitch) % 12)
            chord_ids.append(_triad_chord_id(pcs))

        chord_ids_arr = np.asarray(chord_ids, dtype=np.int32)
        chord_hist = np.bincount(chord_ids_arr, minlength=25).astype(np.float32)
        chord_hist_n = _normalize_hist(chord_hist)

        # chord change rate
        if chord_ids_arr.size >= 2:
            changes = float(np.sum(chord_ids_arr[1:] != chord_ids_arr[:-1]))
            change_rate = changes / float(chord_ids_arr.size - 1)
        else:
            change_rate = 0.0
        chord_vec = np.concatenate([chord_hist_n, np.array([float(change_rate)], dtype=np.float32)], axis=0)  # 26
    except Exception:
        chord_vec = np.zeros((26,), dtype=np.float32)

    # --- Final assembly ---
    # Tempo vector: we want 37 dims total for tempo: 32 samples + [mean,std,min,max] + [n_changes]
    # Our tempo_vec currently has 39 dims (curve + 4 + 3). Reduce to stable 37.
    tempo_mean = tempo_stats[0] if tempo_stats.size else 0.0
    tempo_std = tempo_stats[1] if tempo_stats.size else 0.0
    tempo_min = tempo_extra[0] if tempo_extra.size else 0.0
    tempo_max = tempo_extra[1] if tempo_extra.size else 0.0
    n_changes = tempo_extra[2] if tempo_extra.size else 0.0
    tempo_vec_37 = np.concatenate(
        [tempo_curve.astype(np.float32), np.array([tempo_mean, tempo_std, tempo_min, tempo_max, n_changes], dtype=np.float32)],
        axis=0,
    )

    # Total dim = 37 + 9 + 25 + 16 + 26 + 7 + 6 = 126
    feat = np.concatenate(
        [
            tempo_vec_37,
            ts_vec,
            key_vec,
            pitch_vec,
            chord_vec,
            dur_vec,
            vel_vec,
        ],
        axis=0,
    ).astype(np.float32)

    if feat.shape != (126,):
        # Safety net: keep downstream stable.
        out = np.zeros((126,), dtype=np.float32)
        out[: min(out.size, feat.size)] = feat[: min(out.size, feat.size)]
        return out

    return feat
