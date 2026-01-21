import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalars(logdir, tags):
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in tags:
        if tag not in ea.Tags().get("scalars", []):
            out[tag] = None
            continue
        events = ea.Scalars(tag)
        out[tag] = pd.DataFrame({
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        })
    return out

def plot_run_list(data_root, run_names, out_png, legend_labels=None):
    plt.figure()

    for i, name in enumerate(run_names):
        rd = os.path.join(data_root, name)
        scalars = load_scalars(rd, ["Eval_AverageReturn", "Train_EnvstepsSoFar"])
        df_ret = scalars["Eval_AverageReturn"]
        df_env = scalars["Train_EnvstepsSoFar"]

        if df_ret is None or df_env is None:
            print(f"[skip] missing tags in {name}")
            continue

        df = pd.merge(df_ret, df_env, on="step", suffixes=("_ret", "_env"))
        x = df["value_env"]
        y = df["value_ret"]

        label = legend_labels[i] if legend_labels else name
        plt.plot(x, y, label=label)

    plt.xlabel("Train_EnvstepsSoFar")
    plt.ylabel("Eval_AverageReturn")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("saved:", out_png)

if __name__ == "__main__":
    data_root = "data"

    # ===== Small batch (b=1000) 4개 =====
    small_runs = [
        "q2_pg_cartpole_CartPole-v0_21-01-2026_11-04-02",
        "q2_pg_cartpole_rtg_CartPole-v0_21-01-2026_11-06-13",
        "q2_pg_cartpole_na_CartPole-v0_21-01-2026_11-08-55",
        "q2_pg_cartpole_rtg_na_CartPole-v0_21-01-2026_11-10-53",
    ]
    small_labels = ["b=1000", "b=1000 +rtg", "b=1000 +na", "b=1000 +rtg +na"]
    plot_run_list(data_root, small_runs, "cartpole_b1000_envsteps.png", small_labels)

    # ===== Large batch (b=4000) 4개 =====
    large_runs = [
        "q2_pg_cartpole_lb_CartPole-v0_21-01-2026_11-12-41",
        "q2_pg_cartpole_lb_rtg_CartPole-v0_21-01-2026_11-21-39",
        "q2_pg_cartpole_lb_na_CartPole-v0_21-01-2026_11-27-58",
        "q2_pg_cartpole_lb_rtg_na_CartPole-v0_21-01-2026_11-33-38",
    ]
    large_labels = ["b=4000", "b=4000 +rtg", "b=4000 +na", "b=4000 +rtg +na"]
    plot_run_list(data_root, large_runs, "cartpole_b4000_envsteps.png", large_labels)