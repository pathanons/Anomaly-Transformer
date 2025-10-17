import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import pandas as pd

# ---------- core utilities (vectorized) ----------

def _segment_sse(sum_y, sum_y2, n):
    """
    Return SSE of a constant-mean fit for a segment, works with scalars or arrays.
    sum_y, sum_y2, n can be broadcastable arrays.
    """
    sum_y  = np.asarray(sum_y, dtype=float)
    sum_y2 = np.asarray(sum_y2, dtype=float)
    n      = np.asarray(n, dtype=float)

    # Allocate output (handles both scalar and array cases)
    sse = np.zeros_like(np.broadcast_to(sum_y2, np.broadcast(sum_y2, sum_y, n).shape), dtype=float)

    valid = n > 0
    if np.any(valid):
        mean = np.zeros_like(sse, dtype=float)
        mean[valid] = sum_y[valid] / n[valid]
        sse[valid] = sum_y2[valid] - 2 * mean[valid] * sum_y[valid] + n[valid] * (mean[valid] ** 2)
    return sse


def _best_split_gain(y: np.ndarray, min_seg_len: int) -> Tuple[int, float]:
    """
    Return (k, gain) where k is the best split index (relative to y),
    and gain = SSE_before - (SSE_left + SSE_right). If no valid split,
    returns (-1, 0.0).
    """
    n = y.size
    if n < 2 * min_seg_len + 1:
        return -1, 0.0

    c1 = np.cumsum(y)
    c2 = np.cumsum(y**2)

    sum_all, sum2_all = c1[-1], c2[-1]
    sse_all = _segment_sse(sum_all, sum2_all, n)

    # Candidate splits: ensure both sides >= min_seg_len
    ks = np.arange(min_seg_len, n - min_seg_len + 1)  # split after ks-1

    sumL, sum2L, nL = c1[ks - 1], c2[ks - 1], ks
    sseL = _segment_sse(sumL, sum2L, nL)

    sumR, sum2R, nR = sum_all - sumL, sum2_all - sum2L, n - nL
    sseR = _segment_sse(sumR, sum2R, nR)

    gains = sse_all - (sseL + sseR)
    i = int(np.argmax(gains))
    k_best = int(ks[i])
    gain_best = float(gains[i])
    if gain_best <= 0:
        return -1, 0.0
    return k_best, gain_best


def _binary_segment(
    y: np.ndarray,
    start: int,
    end: int,
    penalty: float,
    min_seg_len: int,
    changes: List[int]
):
    """
    Recursively split [start:end) using SSE gain with penalty.
    Accept a split if gain > penalty and both sides >= min_seg_len.
    """
    seg = y[start:end]
    k_rel, gain = _best_split_gain(seg, min_seg_len)
    if k_rel < 0 or gain <= penalty:
        return
    k_abs = start + k_rel
    changes.append(k_abs)
    _binary_segment(y, start, k_abs, penalty, min_seg_len, changes)
    _binary_segment(y, k_abs, end, penalty, min_seg_len, changes)


def detect_step_changes(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    penalty: float = 6.0,
    min_seg_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect multiple step changes in y using binary segmentation (least-squares).
    Args:
        y: 1D signal
        x: optional x-axis (same length); default = np.arange(len(y))
        penalty: larger -> fewer splits
        min_seg_len: minimum samples per segment
    Returns:
        change_idx: indices where the split occurs (between k-1 and k)
        change_pos: x positions of those splits
    """
    y = np.asarray(y, dtype=float)
    if x is None:
        x = np.arange(y.size, dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    changes: List[int] = []
    _binary_segment(y, 0, y.size, penalty, min_seg_len, changes)
    changes = sorted(set(changes))
    return np.array(changes, dtype=int), x[changes]

# ---------- demo & plotting ----------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y = pd.read_csv('output.csv')
    y = y['CH2V']
    y = y.values
    y = y[290000:650000]
    # Build a multi-step signal (piecewise constants) + noise
    n = len(y)
    x = np.arange(n)
    # true_breaks = [150, 330, 520, 670]  # indices where steps change
    # levels = [1.0, 2.2, 0.9, 3.0, 1.8]

    # y = np.empty(n)
    # start = 0
    # for b, lev in zip(true_breaks + [n], levels):
    #     y[start:b] = lev
    #     start = b
    # y += rng.normal(0, 0.18, size=n)  # add noise

    # Detect changes
    change_idx, change_pos = detect_step_changes(y, x=x, penalty=0.1, min_seg_len=10)
    # print("Detected change indices:", change_idx)
    print("Detected change positions (x_c):", change_pos)

    # Plot signal and vertical lines at detected changes
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, y, linewidth=1.0, label="Signal (noisy, multi-step)")
    for i, xc in enumerate(change_pos):
        plt.axvline(xc, linestyle="--", linewidth=1.0,
                    label="Detected change" if i == 0 else None)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Multiple step-change detection (binary segmentation, SSE cost)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # print(y)