"""Plot reward curves from episode JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _moving_average(xs: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or xs.size == 0:
        return xs.astype(float)
    w = min(window, xs.size)
    kernel = np.ones(w) / w
    pad = np.full(w - 1, xs[0], dtype=float)
    padded = np.concatenate([pad, xs.astype(float)])
    return np.convolve(padded, kernel, mode="valid")


def load_episodes(path: Path) -> list[dict]:
    episodes: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("event") == "episode":
                episodes.append(row)
    return episodes


def plot_reward_curve(
    episodes_path: Path | str,
    out_path: Path | str,
    *,
    window: int = 50,
    title: str = "Reward curve",
) -> None:
    episodes = load_episodes(Path(episodes_path))
    if not episodes:
        raise ValueError(f"No episodes in {episodes_path}")

    rewards = np.array([e.get("retrieval_reward", e.get("reward", 0.0)) for e in episodes])
    irr = np.array([float(e.get("incorrect_retrieval_dominant", 0)) for e in episodes])
    xs = np.arange(len(rewards))
    ma = _moving_average(rewards, window)
    ma_irr = _moving_average(irr, window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(xs, rewards, alpha=0.15, s=8, label="episode reward")
    ax.plot(xs, ma, color="red", linewidth=2, label=f"reward MA({window})")
    ax2 = ax.twinx()
    ax2.plot(xs, ma_irr, color="blue", linestyle="--", linewidth=1.5, label=f"IRR MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax2.set_ylabel("IRR")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--title", default="Reward curve")
    args = parser.parse_args()
    plot_reward_curve(args.episodes, args.out, window=args.window, title=args.title)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
