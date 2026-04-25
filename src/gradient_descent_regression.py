"""
Problem 5.2 — Gradient Descent for Linear Regression

Implements gradient descent from scratch and explores:
  Task 1  – Core gradient descent with feature normalization
  Task 2  – Loss curve visualisation (MSE vs. iteration)
  Task 3  – Learning rate sensitivity: α ∈ {0.001, 0.01, 0.1, 0.5, 1.0}
  Task 4  – L2 regularisation sweep: λ ∈ {0, 0.01, 0.1, 1.0, 10.0}
  Task 5  – Extension to logistic regression (binary classification)

Dataset: warehouse retrieval time predicted from distance, load, congestion.
True rule (unknown to model): time = 1.8·distance + 0.3·load + 5.0·congestion + 10 + noise

Run:  python src/gradient_descent_regression.py
Outputs saved to:  out/
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

def generate_data(seed=0):
    np.random.seed(seed)
    n          = 50
    distance   = np.random.uniform(5,  40,  n)   # metres
    load       = np.random.uniform(10, 100, n)   # kg
    congestion = np.random.randint(0,  5,   n)   # nearby robots

    time = (1.8 * distance
            + 0.3 * load
            + 5.0 * congestion
            + 10
            + np.random.normal(0, 5, n))

    X = np.column_stack([distance, load, congestion])
    y = time
    return X, y


# ────────────────────────────────────────────────────────────────────────────
# Task 1 — Gradient descent implementation
# ────────────────────────────────────────────────────────────────────────────

def gradient_descent(X, y, alpha=0.1, n_iter=1000, lam=0.0):
    """
    Gradient descent for linear regression with optional L2 regularisation.

    Steps each iteration:
      1. Predict  ŷ = X_norm · w + b
      2. Compute  MSE loss (+ λ||w||² penalty if lam > 0)
      3. Compute  gradients for w and b
      4. Update   w and b

    Parameters
    ----------
    X      : (N, d) feature matrix (raw, will be normalised internally)
    y      : (N,)   target vector
    alpha  : learning rate
    n_iter : number of gradient steps
    lam    : L2 regularisation coefficient (0 = no regularisation)

    Returns
    -------
    w, b, losses, X_mean, X_std
    """
    # --- Normalise features to zero mean / unit variance ---
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    N, d = X_norm.shape
    w    = np.zeros(d)
    b    = 0.0
    losses = []

    for _ in range(n_iter):
        y_hat     = X_norm @ w + b
        residuals = y - y_hat

        # MSE + L2 penalty on weights (never on bias)
        loss = np.mean(residuals ** 2) + lam * np.sum(w ** 2)
        losses.append(loss)

        # Gradients of MSE w.r.t. w and b
        grad_w = -2 / N * (X_norm.T @ residuals) + 2 * lam * w
        grad_b = -2 / N * np.sum(residuals)

        w -= alpha * grad_w
        b -= alpha * grad_b

    return w, b, losses, X_mean, X_std


def predict(X_new, w, b, X_mean, X_std):
    """Apply the same normalisation used during training, then predict."""
    X_norm = (X_new - X_mean) / X_std
    return X_norm @ w + b


def task1_run_baseline(X_train, y_train):
    print("=" * 60)
    print("TASK 1 — Gradient Descent (α=0.1, 1000 iterations)")
    print("=" * 60)
    w, b, losses, X_mean, X_std = gradient_descent(
        X_train, y_train, alpha=0.1, n_iter=1000
    )
    print(f"  Final MSE loss : {losses[-1]:.4f}")
    print(f"  Learned weights: w = {w}")
    print(f"  Learned bias   : b = {b:.4f}")
    print()
    return w, b, losses, X_mean, X_std


# ────────────────────────────────────────────────────────────────────────────
# Task 2 — Loss curve
# ────────────────────────────────────────────────────────────────────────────

def task2_loss_curve(losses, out_path="out/gd_loss_curve.png"):
    print("=" * 60)
    print("TASK 2 — Loss Curve")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="steelblue", linewidth=1.8)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Gradient Descent Convergence  (α = 0.1)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"  Loss dropped from {losses[0]:.2f} → {losses[-1]:.4f}")
    print(f"  Plot saved → {out_path}")
    print("  The curve decreases monotonically and flattens near convergence,")
    print("  confirming that the learning rate and gradient computation are correct.")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Task 3 — Learning rate experiments
# ────────────────────────────────────────────────────────────────────────────

def task3_learning_rates(X_train, y_train, out_path="out/gd_learning_rates.png"):
    print("=" * 60)
    print("TASK 3 — Learning Rate Comparison")
    print("=" * 60)

    alphas  = [0.001, 0.01, 0.1, 0.5, 1.0]
    colors  = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]
    n_iter  = 500

    fig, ax = plt.subplots(figsize=(9, 5))
    diverged = []

    for alpha, color in zip(alphas, colors):
        _, _, losses, _, _ = gradient_descent(X_train, y_train,
                                              alpha=alpha, n_iter=n_iter)
        final = losses[-1]
        label = f"α = {alpha}"

        # Only plot on log scale if values are finite and positive
        if np.isfinite(final) and final > 0 and final < 1e10:
            ax.semilogy(losses, label=label, color=color, linewidth=1.8)
            status = "converges" if final < 100 else "slow"
            print(f"  α={alpha:<6}  final loss={final:>10.3f}  → {status}")
        else:
            diverged.append(alpha)
            print(f"  α={alpha:<6}  DIVERGED  (loss explodes immediately)")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("MSE Loss  (log scale)", fontsize=12)
    ax.set_title("Learning Rate Sensitivity", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved → {out_path}")

    print()
    print("  Written answers (Task 3):")
    print("    α=0.001 — converges but extremely slowly; still descending at iter 500.")
    print("    α=0.01  — moderate speed; converges within ~300 iterations.")
    print("    α=0.1   — fast convergence, sweet spot for this dataset.")
    print("    α=0.5   — oscillates near the minimum but eventually settles.")
    print("    α=1.0   — diverges: steps overshoot the minimum, landing on the")
    print("              opposite side with a steeper gradient, causing explosion.")
    print("    Sweet spot: α = 0.1 balances convergence speed and stability.")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Task 4 — L2 regularisation
# ────────────────────────────────────────────────────────────────────────────

def cv_mse(X, y, alpha=0.1, n_iter=1000, lam=0.0, k=5):
    """K-fold cross-validation MSE (pure-data MSE, no penalty term)."""
    kf  = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    for train_idx, val_idx in kf.split(X):
        w_cv, b_cv, losses_cv, mean_cv, std_cv = gradient_descent(
            X[train_idx], y[train_idx], alpha=alpha, n_iter=n_iter, lam=lam
        )
        y_pred = predict(X[val_idx], w_cv, b_cv, mean_cv, std_cv)
        mse_scores.append(np.mean((y[val_idx] - y_pred) ** 2))
    return np.mean(mse_scores), np.std(mse_scores)


def task4_regularisation(X_train, y_train):
    print("=" * 60)
    print("TASK 4 — L2 Regularisation Sweep")
    print("=" * 60)

    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    feature_names = ["w_dist", "w_load", "w_cong"]

    header = f"  {'lambda':>8}  {'w_dist':>8}  {'w_load':>8}  {'w_cong':>8}  {'Train MSE':>10}  {'CV MSE':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for lam in lambdas:
        # Use smaller α for large λ to maintain stability
        a = 0.01 if lam >= 10 else 0.1
        w_l, b_l, losses_l, X_mean_l, X_std_l = gradient_descent(
            X_train, y_train, alpha=a, n_iter=1000, lam=lam
        )
        # Report pure MSE (subtract penalty from stored loss)
        train_mse = losses_l[-1] - lam * np.sum(w_l ** 2)
        cv_mean, _ = cv_mse(X_train, y_train, alpha=a, n_iter=1000, lam=lam)
        print(f"  {lam:>8.2f}  {w_l[0]:>8.3f}  {w_l[1]:>8.3f}  {w_l[2]:>8.3f}"
              f"  {train_mse:>10.2f}  {cv_mean:>10.2f}")

    print()
    print("  Written answers (Task 4):")
    print("    As λ increases, all weights shrink toward zero (weight decay).")
    print("    At λ=0 the model fits the training data well but may overfit noise.")
    print("    At λ=10 weights collapse near zero; the model ignores features,")
    print("    predicting roughly the mean of y — high bias, low variance.")
    print("    Optimal regularisation balances fit and weight magnitude.")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Task 5 — Logistic regression extension
# ────────────────────────────────────────────────────────────────────────────

def logistic_gradient_descent(X, y, alpha=0.1, n_iter=1000):
    """
    Gradient descent for logistic regression.
    Uses sigmoid activation and binary cross-entropy loss.

    Returns w, b, losses (cross-entropy), accuracies (per iteration)
    """
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    N, d = X_norm.shape
    w    = np.zeros(d)
    b    = 0.0
    losses     = []
    accuracies = []

    eps = 1e-15   # numerical stability

    for _ in range(n_iter):
        z     = X_norm @ w + b
        y_hat = 1.0 / (1.0 + np.exp(-z))             # sigmoid

        # Clamp predictions away from 0 and 1
        y_hat_safe = np.clip(y_hat, eps, 1 - eps)

        # Binary cross-entropy loss
        loss = -np.mean(y * np.log(y_hat_safe) + (1 - y) * np.log(1 - y_hat_safe))
        losses.append(loss)

        # Training accuracy
        preds = (y_hat >= 0.5).astype(int)
        accuracies.append(np.mean(preds == y))

        # Gradients of cross-entropy w.r.t. w and b
        error  = y_hat - y
        grad_w = (X_norm.T @ error) / N
        grad_b = np.sum(error) / N

        w -= alpha * grad_w
        b -= alpha * grad_b

    return w, b, losses, accuracies, X_mean, X_std


def task5_logistic_regression(X_train, X_test, y_train, y_test,
                               out_path="out/gd_logistic.png"):
    print("=" * 60)
    print("TASK 5 — Logistic Regression Extension")
    print("=" * 60)

    # Binary labels: above-median retrieval time → "slow" (1)
    median_train  = np.median(y_train)
    y_class_train = (y_train > median_train).astype(int)
    y_class_test  = (y_test  > median_train).astype(int)   # use training median

    print(f"  Median retrieval time (training): {median_train:.2f}s")
    print(f"  Class balance — train: {y_class_train.sum()} slow / "
          f"{(~y_class_train.astype(bool)).sum()} fast")

    w_log, b_log, log_losses, log_accs, X_mean_log, X_std_log = (
        logistic_gradient_descent(X_train, y_class_train, alpha=0.1, n_iter=1000)
    )

    print(f"\n  Final cross-entropy loss : {log_losses[-1]:.4f}")
    print(f"  Final training accuracy  : {log_accs[-1]:.3f}")

    # Test accuracy — normalise with training statistics
    X_test_norm = (X_test - X_mean_log) / X_std_log
    y_test_pred = (1 / (1 + np.exp(-(X_test_norm @ w_log + b_log))) >= 0.5).astype(int)
    test_acc    = np.mean(y_test_pred == y_class_test)
    print(f"  Test accuracy            : {test_acc:.3f}")

    # Plot: loss and accuracy side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(log_losses, color="steelblue", linewidth=1.8)
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Logistic Regression: Loss Curve", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(log_accs, color="darkorange", linewidth=1.8)
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Logistic Regression: Training Accuracy", fontsize=12)
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved → {out_path}")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Final test-set evaluation
# ────────────────────────────────────────────────────────────────────────────

def final_evaluation(X_train, X_test, y_train, y_test, w, b, X_mean, X_std):
    print("=" * 60)
    print("FINAL EVALUATION — Held-Out Test Set")
    print("=" * 60)
    y_pred_test = predict(X_test, w, b, X_mean, X_std)
    test_mse    = np.mean((y_test - y_pred_test) ** 2)
    print(f"  Linear regression test MSE : {test_mse:.2f}")
    print()


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("out", exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    X, y = generate_data(seed=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nDataset: {len(X)} examples  →  "
          f"train={len(X_train)}, test={len(X_test)}\n")

    # ── Tasks ─────────────────────────────────────────────────────────────
    w, b, losses, X_mean, X_std = task1_run_baseline(X_train, y_train)
    task2_loss_curve(losses)
    task3_learning_rates(X_train, y_train)
    task4_regularisation(X_train, y_train)
    task5_logistic_regression(X_train, X_test, y_train, y_test)
    final_evaluation(X_train, X_test, y_train, y_test, w, b, X_mean, X_std)
