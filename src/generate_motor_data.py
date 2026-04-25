"""
generate_motor_data.py
Generates 1200 synthetic motor current waveforms (400 per class, 128 time steps).
Classes: healthy (0), bearing_wear (1), winding_fault (2).
Run once to produce motor_current_data.npz.
"""
import numpy as np


def generate_motor_data(n_per_class=400, seq_len=128, seed=42):
    """
    Three fault classes with subtly different waveforms:
      healthy      — clean sinusoid + amplitude variation + noise
      bearing_wear — sinusoid + high-frequency ripple (subtle)
      winding_fault— slightly asymmetric sinusoid (rectified component)
    Anomalies are intentionally small relative to noise.
    """
    rng      = np.random.default_rng(seed)
    n_cycles = 4
    t        = np.linspace(0, 2 * np.pi * n_cycles, seq_len)
    sequences, labels = [], []
    class_names = ["healthy", "bearing_wear", "winding_fault"]

    for _ in range(n_per_class):
        A         = rng.uniform(0.8, 1.2)
        load_env  = 1.0 + rng.uniform(-0.15, 0.15) * np.sin(t * rng.uniform(0.05, 0.2))
        phase     = rng.uniform(0, 2 * np.pi)
        noise     = rng.normal(0, rng.uniform(0.05, 0.08), seq_len)
        current   = A * load_env * np.sin(t + phase) + noise
        sequences.append(current.astype(np.float32)); labels.append(0)

    for _ in range(n_per_class):
        A            = rng.uniform(0.8, 1.2)
        load_env     = 1.0 + rng.uniform(-0.15, 0.15) * np.sin(t * rng.uniform(0.05, 0.2))
        phase        = rng.uniform(0, 2 * np.pi)
        noise        = rng.normal(0, rng.uniform(0.05, 0.08), seq_len)
        ripple_freq  = rng.uniform(15, 25)
        ripple_amp   = rng.uniform(0.10, 0.20)
        ripple       = ripple_amp * np.sin(ripple_freq * t + rng.uniform(0, 2 * np.pi))
        current      = A * load_env * np.sin(t + phase) + ripple + noise
        sequences.append(current.astype(np.float32)); labels.append(1)

    for _ in range(n_per_class):
        A         = rng.uniform(0.8, 1.2)
        load_env  = 1.0 + rng.uniform(-0.15, 0.15) * np.sin(t * rng.uniform(0.05, 0.2))
        phase     = rng.uniform(0, 2 * np.pi)
        noise     = rng.normal(0, rng.uniform(0.05, 0.08), seq_len)
        asymmetry = rng.uniform(0.15, 0.30)
        base      = np.sin(t + phase)
        current   = A * load_env * base + asymmetry * np.maximum(0, base) + noise
        sequences.append(current.astype(np.float32)); labels.append(2)

    sequences = np.array(sequences)
    labels    = np.array(labels, dtype=np.int64)
    idx       = rng.permutation(len(sequences))
    return sequences[idx], labels[idx], class_names


if __name__ == "__main__":
    seqs, labels, class_names = generate_motor_data(n_per_class=400, seed=42)
    print(f"Sequences: {seqs.shape}")
    print(f"Labels:    {labels.shape},  classes: {class_names}")
    print(f"Class distribution: {[int((labels==i).sum()) for i in range(3)]}")
    np.savez("motor_current_data.npz",
             sequences=seqs, labels=labels, class_names=class_names)
    print("Saved motor_current_data.npz")
