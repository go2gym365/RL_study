import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DATA_DIR = "data"
Y_TAG = "Eval_AverageReturn"
X_TAG = "Train_EnvstepsSoFar"


GROUPS = {
    "Default (bs=5000)": r"q2_pg_pendulum_default_s[1-5]_InvertedPendulum-v4",
    "Tuned (bs=2000, lr=1e-2)": r"q2_pg_pendulum_tuned_bs2000_lr1e2_s[1-5]_InvertedPendulum-v4",
    "Tuned + GAE (λ=0.95)": r"q2_pg_pendulum_tuned_bs2000_lr1e2_gae095_s[1-5]_InvertedPendulum-v4",
}


def load_xy(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()

    if Y_TAG not in ea.Tags()["scalars"] or X_TAG not in ea.Tags()["scalars"]:
        return None

    y = np.array([e.value for e in ea.Scalars(Y_TAG)])
    x = np.array([e.value for e in ea.Scalars(X_TAG)])

    n = min(len(x), len(y))
    return x[:n], y[:n]


def main():
    runs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)]

    plt.figure(figsize=(9, 5))

    for group_name, pattern in GROUPS.items():
        series = []

        for r in runs:
            if re.search(pattern, os.path.basename(r)):
                xy = load_xy(r)
                if xy is not None:
                    series.append(xy)

        if len(series) == 0:
            print(f"[WARN] No runs for {group_name}")
            continue

        # 공통 x축 (env steps)
        x_min = max(s[0][0] for s in series)
        x_max = min(s[0][-1] for s in series)
        x_grid = np.linspace(x_min, x_max, 200)

        ys = []
        for x, y in series:
            ys.append(np.interp(x_grid, x, y))

        ys = np.stack(ys)
        mean = ys.mean(axis=0)
        std = ys.std(axis=0)

        plt.plot(x_grid, mean, label=f"{group_name} (n={len(series)})")
        plt.fill_between(x_grid, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.title("Experiment 4: InvertedPendulum-v4 (Average over 5 seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiment4_envsteps.png", dpi=200)
    print("Saved: experiment4_envsteps.png")


if __name__ == "__main__":
    main()