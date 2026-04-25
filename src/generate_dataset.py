"""
generate_dataset.py — Warehouse picking-time dataset generator

Produces picking_time_data.npz with 400 examples and 5 features.
True relationship contains nonlinear effects that a linear model cannot capture:
  - Congestion × distance interaction
  - Battery threshold step at 20%
  - Mild quadratic term in distance

Run once to create the .npz file, then load it in picking_time_nn.py.
"""

import numpy as np


def generate_picking_time_dataset(n=400, seed=42):
    """
    Generate synthetic warehouse picking-time data with nonlinear structure.

    Features
    --------
    distance    Uniform[2, 30]   metres to item
    load        Uniform[1, 50]   item weight (kg)
    congestion  Poisson(1.5)     number of robots in aisle
    battery     Beta(5, 2)       battery level in [0, 1], skewed high
    aisle_width Uniform[1.5, 3]  aisle width (metres)

    True model (unknown to the learner)
    ------------------------------------
    y = 5 + 0.8·d + 0.15·l + 0.4·c·d + 12·1[b < 0.2] − 2·w + 0.01·d² + ε
        ε ~ N(0, 4)

    Nonlinear terms:
      0.4·c·d     — congestion costs more when the robot is far away
      12·1[b<0.2] — sharp 12-second penalty when battery is low
      0.01·d²     — mild quadratic: distant items are disproportionately slow
    """
    rng = np.random.default_rng(seed)

    distance    = rng.uniform(2,   30,  n)
    load        = rng.uniform(1,   50,  n)
    congestion  = rng.poisson(1.5,      n).astype(float)
    battery     = rng.beta(5,   2,      n)          # [0,1], skewed toward 1
    aisle_width = rng.uniform(1.5, 3.0, n)

    y = (
        5.0
        + 0.8  * distance
        + 0.15 * load
        + 0.4  * congestion * distance          # NONLINEAR: interaction
        + 12.0 * (battery < 0.2).astype(float)  # NONLINEAR: threshold
        - 2.0  * aisle_width
        + 0.01 * distance ** 2                  # NONLINEAR: quadratic
    )
    y = y + rng.normal(0, 2.0, n)              # observation noise σ=2

    X            = np.column_stack([distance, load, congestion, battery, aisle_width])
    feature_names = ["distance", "load", "congestion", "battery", "aisle_width"]

    return X, y, feature_names


if __name__ == "__main__":
    X, y, feature_names = generate_picking_time_dataset(n=400, seed=42)

    print(f"Dataset shape : X={X.shape},  y={y.shape}")
    print(f"Features      : {feature_names}")
    print(f"Target range  : [{y.min():.1f}, {y.max():.1f}],  mean={y.mean():.1f} sec")

    np.savez("picking_time_data.npz", X=X, y=y, feature_names=feature_names)
    print("Saved → picking_time_data.npz")
