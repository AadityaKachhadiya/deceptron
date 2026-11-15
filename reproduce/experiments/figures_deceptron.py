"""Regenerate plots from the archived figure-generation notebook."""

from pathlib import Path

from _output import prepare_output

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRACES = RESULTS / "traces"

prepare_output("figures")

import matplotlib

matplotlib.use("Agg")



# Original figure notebook, cell 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

# fig 1

# csvs required
files = {
    "Heat-3D": str(TRACES / "heat3d_traces.csv"),
    "Advection–Diffusion-2D": str(TRACES / "advdiff_traces.csv"),
    "Allen–Cahn-2D": str(TRACES / "allen_cahn_main_traces.csv"),
}

EPS = 0.30
TAU_MAX_TIME = 120.0
TAU_MAX_ITER = 120.0
N_TAU = 1000
N_BUDGET = 400


dfs = {}
for task_name, fname in files.items():
    df = pd.read_csv(fname)
    if "seed" not in df.columns:
        df["seed"] = 0

    required = ["method", "sample_id", "iteration", "residual", "rmse_x", "seed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fname} missing columns: {missing}")

    if "time_sec" not in df.columns:
        raise ValueError(f"{fname} needs a 'time_sec' column for time-based profiles")

    dfs[task_name] = df.copy()


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


METHOD_ORDER = [
    "D-IPG (+JCP)",
    "GN",
    "LM",
    "L-BFGS",
    "GD",
]

METHOD_LABELS = {
    "D-IPG (+JCP)": "D-IPG",
    "GN": "GN",
    "LM": "LM",
    "L-BFGS": "L-BFGS",
    "GD": "GD",
}

METHOD_COLORS = {
    "D-IPG (+JCP)": "#000000",
    "GN": "#4C72B0",
    "LM": "#DD8452",
    "L-BFGS": "#8172B3",
    "GD": "#55A868",
}

METHOD_WIDTHS = {
    "D-IPG (+JCP)": 2.5,
    "GN": 2.1,
    "LM": 2.1,
    "L-BFGS": 2.1,
    "GD": 2.1,
}

METHOD_LINESTYLES = {
    "D-IPG (+JCP)": "-",
    "GN": ":",
    "LM": "-.",
    "L-BFGS": "--",
    "GD": (0, (4, 2)),
}

# helpers
def canonicalize_method_name(raw_name):
    s = str(raw_name).strip()
    s_low = s.lower()

    if s == "D-IPG (+JCP)":
        return "D-IPG (+JCP)"
    if s in {"GN", "LM", "L-BFGS", "GD"}:
        return s
    if s == "Gradient Descent":
        return "GD"
    if "d-ipg" in s_low and "+jcp" in s_low:
        return "D-IPG (+JCP)"
    if s_low in {"gradient descent", "gd"}:
        return "GD"
    return None

def style_box_axes(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#555555")
        spine.set_visible(True)

def normalize_rmse_per_trajectory(df_method):
    parts = []
    for (_, _), sub in df_method.groupby(["seed", "sample_id"]):
        sub = sub.sort_values("iteration").copy()
        r0 = float(sub["rmse_x"].iloc[0])
        sub["norm_rmse"] = sub["rmse_x"] / max(r0, 1e-12)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def aggregate_vs_iteration(df_method, value_col):
    grp = (
        df_method.groupby("iteration")[value_col]
        .agg(
            mean="mean",
            std=lambda x: np.std(x, ddof=0),
        )
        .reset_index()
    )
    grp["low"] = np.maximum(grp["mean"] - grp["std"], 1e-12)
    grp["high"] = np.maximum(grp["mean"] + grp["std"], 1e-12)
    return grp

def first_hit_time_and_iter(sub, eps):
    sub = sub.sort_values("iteration")
    r = sub["residual"].to_numpy(dtype=float)
    r0 = max(float(r[0]), 1e-12)

    hit = np.where((r / r0) <= eps)[0]
    if len(hit) == 0:
        return np.inf, np.inf

    k = hit[0]
    t_hit = float(sub["time_sec"].iloc[k])
    it_hit = float(sub["iteration"].iloc[k])
    return t_hit, it_hit

def compute_performance_profile(dfs_dict, methods, eps, measure="time", tau_max=120.0, n_tau=1000):
    ratios = {m: [] for m in methods}
    total_instances = 0

    for _, df_raw in dfs_dict.items():
        df = df_raw.copy()
        df["canon_method"] = df["method"].map(canonicalize_method_name)

        for (seed, sid), _ in df.groupby(["seed", "sample_id"]):
            perf = {}

            for m in methods:
                sub = df[
                    (df["canon_method"] == m) &
                    (df["seed"] == seed) &
                    (df["sample_id"] == sid)
                ].sort_values("iteration")

                if sub.empty:
                    perf[m] = np.inf
                    continue

                t_hit, it_hit = first_hit_time_and_iter(sub, eps)
                perf[m] = t_hit if measure == "time" else it_hit

            finite_vals = [v for v in perf.values() if np.isfinite(v)]
            best = min(finite_vals) if finite_vals else np.inf

            for m in methods:
                v = perf[m]
                if np.isfinite(v) and np.isfinite(best) and best > 1e-12:
                    ratios[m].append(v / best)
                else:
                    ratios[m].append(np.inf)

            total_instances += 1

    tau_grid = np.logspace(0, np.log10(tau_max), n_tau)
    profiles = {}
    for m in methods:
        arr = np.array(ratios[m], dtype=float)
        profiles[m] = np.array([
            np.sum(arr <= tau) / total_instances for tau in tau_grid
        ])

    return tau_grid, profiles, total_instances

def compute_data_profile(dfs_dict, methods, eps, measure="time", n_budget=400):
    solved_costs = {m: [] for m in methods}
    total_instances = 0

    for _, df_raw in dfs_dict.items():
        df = df_raw.copy()
        df["canon_method"] = df["method"].map(canonicalize_method_name)

        for (seed, sid), _ in df.groupby(["seed", "sample_id"]):
            total_instances += 1

            for m in methods:
                sub = df[
                    (df["canon_method"] == m) &
                    (df["seed"] == seed) &
                    (df["sample_id"] == sid)
                ].sort_values("iteration")

                if sub.empty:
                    solved_costs[m].append(np.inf)
                    continue

                t_hit, it_hit = first_hit_time_and_iter(sub, eps)
                solved_costs[m].append(t_hit if measure == "time" else it_hit)

    finite_all = []
    for m in methods:
        finite_all.extend([v for v in solved_costs[m] if np.isfinite(v)])

    if len(finite_all) == 0:
        raise ValueError("No successful solves found for data profile.")

    bmin = max(min(finite_all), 1e-12)
    bmax = max(finite_all)

    if bmax <= bmin:
        budgets = np.array([bmin, bmin * 1.01])
    else:
        budgets = np.logspace(np.log10(bmin), np.log10(bmax), n_budget)

    profiles = {}
    for m in methods:
        arr = np.array(solved_costs[m], dtype=float)
        profiles[m] = np.array([
            np.sum(arr <= b) / total_instances for b in budgets
        ])

    return budgets, profiles, total_instances


available_methods = set()
for df in dfs.values():
    canon = df["method"].map(canonicalize_method_name)
    available_methods.update(set(canon.dropna().unique()))
available_methods = [m for m in METHOD_ORDER if m in available_methods]

fig = plt.figure(figsize=(15.2, 8.0), dpi=240)
gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[1.35, 1.0],
    hspace=0.42,
    wspace=0.28
)

axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])

task_names = list(files.keys())

# row 1
for col, task_name in enumerate(task_names):
    df_task_raw = dfs[task_name].copy()
    ax_top = axes_top[col]

    df_task_raw["canon_method"] = df_task_raw["method"].map(canonicalize_method_name)

    top_series = {}
    max_iter_seen = 0

    for method in available_methods:
        sub = df_task_raw[df_task_raw["canon_method"] == method].copy()
        if sub.empty:
            continue
        subn = normalize_rmse_per_trajectory(sub)
        if subn.empty:
            continue
        agg = aggregate_vs_iteration(subn, "norm_rmse")
        if agg.empty:
            continue
        top_series[method] = agg
        max_iter_seen = max(max_iter_seen, int(agg["iteration"].max()))

    plot_order = [m for m in available_methods if m in top_series and m != "D-IPG (+JCP)"] + \
                 (["D-IPG (+JCP)"] if "D-IPG (+JCP)" in top_series else [])

    handles_local = []
    labels_local = []

    for method in plot_order:
        agg = top_series[method]

        if method == "D-IPG (+JCP)":
            ax_top.fill_between(
                agg["iteration"], agg["low"], agg["high"],
                color=METHOD_COLORS[method], alpha=0.10, linewidth=0, zorder=3
            )

        line, = ax_top.plot(
            agg["iteration"],
            agg["mean"],
            color=METHOD_COLORS[method],
            linewidth=METHOD_WIDTHS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
            zorder=4 if method == "D-IPG (+JCP)" else 2,
            solid_capstyle="round",
        )
        handles_local.append(line)
        labels_local.append(METHOD_LABELS[method])

    ax_top.set_title(task_name, pad=6)
    ax_top.set_xlabel("Iteration")
    ax_top.set_ylabel("Normalized RMSE" if col == 0 else "")
    ax_top.set_xlim(left=0.8, right=max_iter_seen + 0.5)
    ax_top.set_ylim(0.0, 1.05)
    ax_top.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    style_box_axes(ax_top)

    ax_top.legend(
        handles_local,
        labels_local,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        borderpad=0.35,
        handlelength=2.0,
        labelspacing=0.25,
    )

    ax_top.text(
        -0.13, 1.05, f"({chr(ord('a') + col)})",
        transform=ax_top.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

# row 2
tau_time, prof_time, N_time = compute_performance_profile(
    dfs, available_methods, EPS,
    measure="time", tau_max=TAU_MAX_TIME, n_tau=N_TAU
)

order_bottom = [m for m in available_methods if m != "D-IPG (+JCP)"] + \
               (["D-IPG (+JCP)"] if "D-IPG (+JCP)" in available_methods else [])

for m in order_bottom:
    ax_d.semilogx(
        tau_time, prof_time[m],
        color=METHOD_COLORS[m],
        lw=METHOD_WIDTHS[m],
        ls=METHOD_LINESTYLES[m],
        label=METHOD_LABELS[m],
        zorder=5 if m == "D-IPG (+JCP)" else 3,
        solid_capstyle="round"
    )

ax_d.axhline(1.0, color="#CCCCCC", lw=0.8, zorder=0)
ax_d.axvline(1.0, color="#E0E0E0", lw=0.8, zorder=0)
ax_d.set_xlim(1.0, TAU_MAX_TIME)
ax_d.set_ylim(-0.02, 1.08)
ax_d.set_title("Dolan–Moré (time)", pad=6)
ax_d.set_xlabel(r"Performance ratio $\tau$")
ax_d.set_ylabel(r"Fraction solved $\rho_s(\tau)$")
ax_d.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax_d.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: ("1×" if abs(x - 1.0) < 1e-12 else f"{int(x)}×"))
)
style_box_axes(ax_d)
ax_d.text(
    -0.15, 1.04, "(d)",
    transform=ax_d.transAxes,
    fontsize=12,
    fontweight="bold",
    va="bottom",
    ha="left"
)

# Dolan–More by iterations

tau_iter, prof_iter, N_iter = compute_performance_profile(
    dfs, available_methods, EPS,
    measure="iter", tau_max=TAU_MAX_ITER, n_tau=N_TAU
)

for m in order_bottom:
    ax_e.semilogx(
        tau_iter, prof_iter[m],
        color=METHOD_COLORS[m],
        lw=METHOD_WIDTHS[m],
        ls=METHOD_LINESTYLES[m],
        zorder=5 if m == "D-IPG (+JCP)" else 3,
        solid_capstyle="round"
    )

ax_e.axhline(1.0, color="#CCCCCC", lw=0.8, zorder=0)
ax_e.axvline(1.0, color="#E0E0E0", lw=0.8, zorder=0)
ax_e.set_xlim(1.0, TAU_MAX_ITER)
ax_e.set_ylim(-0.02, 1.08)
ax_e.set_title("Dolan–Moré (iterations)", pad=6)
ax_e.set_xlabel(r"Performance ratio $\tau$")
ax_e.set_ylabel("")
ax_e.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax_e.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: ("1×" if abs(x - 1.0) < 1e-12 else f"{int(x)}×"))
)
style_box_axes(ax_e)
ax_e.text(
    -0.15, 1.04, "(e)",
    transform=ax_e.transAxes,
    fontsize=12,
    fontweight="bold",
    va="bottom",
    ha="left"
)

# data profile by time budget

budgets, data_prof, N_data = compute_data_profile(
    dfs, available_methods, EPS,
    measure="time", n_budget=N_BUDGET
)

for m in order_bottom:
    ax_f.semilogx(
        budgets, data_prof[m],
        color=METHOD_COLORS[m],
        lw=METHOD_WIDTHS[m],
        ls=METHOD_LINESTYLES[m],
        zorder=5 if m == "D-IPG (+JCP)" else 3,
        solid_capstyle="round"
    )

ax_f.axhline(1.0, color="#CCCCCC", lw=0.8, zorder=0)
ax_f.set_ylim(-0.02, 1.08)
ax_f.set_title("Data profile (time budget)", pad=6)
ax_f.set_xlabel("Wall-clock budget (s)")
ax_f.set_ylabel("")
ax_f.yaxis.set_major_locator(MaxNLocator(nbins=5))
style_box_axes(ax_f)
ax_f.text(
    -0.15, 1.04, "(f)",
    transform=ax_f.transAxes,
    fontsize=12,
    fontweight="bold",
    va="bottom",
    ha="left"
)


ax_f.legend(
    [plt.Line2D([0], [0],
                color=METHOD_COLORS[m],
                lw=METHOD_WIDTHS[m],
                ls=METHOD_LINESTYLES[m]) for m in order_bottom],
    [METHOD_LABELS[m] for m in order_bottom],
    loc="lower right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#DDDDDD",
    borderpad=0.40,
    handlelength=2.2,
    labelspacing=0.28,
    fontsize=9,
    ncol=1
)

ax_d.text(
    0.98, 0.05,
    rf"$N={N_time}$ pooled instances",
    transform=ax_d.transAxes,
    fontsize=8,
    color="#777777",
    va="bottom",
    ha="right"
)

ax_e.text(
    0.98, 0.05,
    "Higher and further left is better",
    transform=ax_e.transAxes,
    fontsize=8,
    color="#777777",
    va="bottom",
    ha="right"
)

ax_f.text(
    0.98, 0.05,
    "Higher curve = more instances solved within budget",
    transform=ax_f.transAxes,
    fontsize=8,
    color="#777777",
    va="bottom",
    ha="right"
)

plt.tight_layout()
plt.close('all')


fig.savefig("figure1_dolan_more.pdf", format="pdf", bbox_inches="tight", transparent=True)
fig.savefig("figure1_dolan_more.png", dpi=600, bbox_inches="tight")


# Original figure notebook, cell 5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# fig 2

TRACE_MAIN = str(TRACES / "allen_cahn_main_traces.csv")
SUCCESS_THRESH = 0.095
N_ITERS = 6
BOOTSTRAPS = 500
RNG_SEED = 7


ac = pd.read_csv(TRACE_MAIN)
required = ["method", "sample_id", "iteration", "rmse_x", "residual"]
missing = [c for c in required if c not in ac.columns]
if missing:
    raise ValueError(f"{TRACE_MAIN} missing columns: {missing}")


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "D-IPG (+JCP)": "#111111",
    "D-IPG (-JCP)": "#111111",
    "GN": "#4C72B0",
    "L-BFGS": "#8172B3",
}

LINESTYLES = {
    "D-IPG (+JCP)": "-",
    "D-IPG (-JCP)": "--",
    "GN": ":",
}

LINEWIDTHS = {
    "D-IPG (+JCP)": 2.2,
    "D-IPG (-JCP)": 2.0,
    "GN": 1.8,
}

METHOD_ORDER = ["D-IPG (+JCP)", "D-IPG (-JCP)", "GN", "L-BFGS"]

PANEL_A_BAR = "#4C72B0"

def style_box_axes(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#555555")
        spine.set_visible(True)

def add_panel_label(ax, label):
    ax.text(
        -0.14, 1.04, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0:
        return np.nan
    return abs(mx - my) / np.sqrt(pooled)



# build 6-step RMSE trajectories per trial


results = {}
for method in ["D-IPG (+JCP)", "D-IPG (-JCP)", "GN", "L-BFGS"]:
    sub = ac[ac["method"] == method].copy()
    traj = []
    meta = []
    for sid, grp in sub.groupby("sample_id"):
        grp = grp.sort_values("iteration")
        rmse_by_iter = grp.set_index("iteration")["rmse_x"]
        vec = np.array([rmse_by_iter.get(i, rmse_by_iter.iloc[-1]) for i in range(1, N_ITERS + 1)], dtype=float)
        min_rmse = float(rmse_by_iter.min())
        success = bool(min_rmse < SUCCESS_THRESH)
        traj.append(vec)
        meta.append({
            "sample_id": sid,
            "success": success,
            "min_rmse": min_rmse,
            "final_rmse": float(rmse_by_iter.iloc[-1]),
            "n_iters": int(grp["iteration"].max()),
        })
    results[method] = {
        "traj": np.array(traj, dtype=float),
        "meta": pd.DataFrame(meta)
    }

# recovered statistics

minus = results["D-IPG (-JCP)"]
plus  = results["D-IPG (+JCP)"]
gn    = results["GN"]

minus_success = minus["meta"]["success"].to_numpy()
minus_fail = ~minus_success

conv_traj = minus["traj"][minus_success]
fail_traj = minus["traj"][minus_fail]
plus_traj = plus["traj"]

# panel (a)
sep_rows = []
for i in range(N_ITERS):
    cm, cs = conv_traj[:, i].mean(), conv_traj[:, i].std(ddof=0)
    fm, fs = fail_traj[:, i].mean(), fail_traj[:, i].std(ddof=0)
    d = cohen_d(conv_traj[:, i], fail_traj[:, i])
    sep_rows.append((i + 1, cm, cs, fm, fs, d))
d_df = pd.DataFrame(sep_rows, columns=["iteration", "conv_mean", "conv_std", "fail_mean", "fail_std", "cohen_d"])

# panel (b)
strip_methods = ["D-IPG (+JCP)", "D-IPG (-JCP)", "GN", "L-BFGS"]
data_list = [results[m]["meta"]["min_rmse"].to_numpy(dtype=float) for m in strip_methods]

minus_conv_center = minus["meta"].loc[minus["meta"]["success"], "min_rmse"].mean()
minus_fail_center = minus["meta"].loc[~minus["meta"]["success"], "min_rmse"].mean()

# panel (c)
# +JCP = all 80 trials
# -JCP = only successful 15 trials
# GN   = all 80 trials

curve_data = {
    "D-IPG (+JCP)": {
        "mean": plus_traj.mean(axis=0),
        "std": plus_traj.std(axis=0, ddof=0),
    },
    "D-IPG (-JCP)": {
        "mean": conv_traj.mean(axis=0),
        "std": conv_traj.std(axis=0, ddof=0),
    },
    "GN": {
        "mean": gn["traj"].mean(axis=0),
        "std": gn["traj"].std(axis=0, ddof=0),
    },
}

# ratio strip = successful -JCP / +JCP
rng = np.random.default_rng(RNG_SEED)
ratio_rows = []
for i in range(N_ITERS):
    plus_vals = plus_traj[:, i]
    minus_vals = conv_traj[:, i]
    ratio_mean = minus_vals.mean() / max(plus_vals.mean(), 1e-12)

    boot = np.empty(BOOTSTRAPS, dtype=float)
    for b in range(BOOTSTRAPS):
        s_plus = rng.choice(plus_vals, size=len(plus_vals), replace=True)
        s_minus = rng.choice(minus_vals, size=len(minus_vals), replace=True)
        boot[b] = s_minus.mean() / max(s_plus.mean(), 1e-12)

    ratio_rows.append({
        "iteration": i + 1,
        "mean": ratio_mean,
        "std": boot.std(ddof=1),
    })
ratio_df = pd.DataFrame(ratio_rows)

ref_slope = -0.49


fig = plt.figure(figsize=(14.0, 4.0), dpi=140)
outer = gridspec.GridSpec(
    1, 3,
    width_ratios=[1.35, 1.35, 1.15],
    wspace=0.38
)

left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[0],
    height_ratios=[1.0, 3.2], hspace=0.05
)

right = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[2],
    height_ratios=[3.0, 1.0], hspace=0.05
)

ax_a_top = fig.add_subplot(left[0])
ax_a_bot = fig.add_subplot(left[1], sharex=ax_a_top)
ax_b = fig.add_subplot(outer[1])
ax_c = fig.add_subplot(right[0])
ax_c_ratio = fig.add_subplot(right[1], sharex=ax_c)

# (a) Broken-axis Cohen's d bar chart

x = d_df["iteration"].to_numpy(dtype=int)
y = d_df["cohen_d"].to_numpy(dtype=float)

ax_a_bot.bar(x, y, color=PANEL_A_BAR, edgecolor="#2F4B7C", linewidth=0.6, width=0.68, zorder=3)
ax_a_top.bar(x, y, color=PANEL_A_BAR, edgecolor="#2F4B7C", linewidth=0.6, width=0.68, zorder=3)

ax_a_bot.set_ylim(0, 3.0)
ax_a_top.set_ylim(40, max(50, float(np.nanmax(y)) + 1.5))

ax_a_bot.axhline(2.0, color="#BDBDBD", lw=0.8, ls="--", zorder=1)
ax_a_bot.text(
    5.95, 2.05, "2σ detectable difference",
    fontsize=8, color="#777777", ha="right", va="bottom"
)

last_it = int(d_df["iteration"].iloc[-1])
last_y = float(d_df["cohen_d"].iloc[-1])
ax_a_top.text(last_it, last_y + 0.5, f"{last_y:.1f}σ",
              ha="center", va="bottom", fontsize=8.5, color="#1F1F1F")

for ax in (ax_a_top, ax_a_bot):
    style_box_axes(ax)
    ax.set_xlim(0.4, N_ITERS + 0.6)
    ax.set_xticks(np.arange(1, N_ITERS + 1))

ax_a_top.spines["bottom"].set_visible(False)
ax_a_bot.spines["top"].set_visible(False)
ax_a_top.tick_params(labelbottom=False, bottom=False)

ax_a_bot.set_xlabel("Iteration")
ax_a_bot.set_ylabel("Separation (σ)")
ax_a_top.set_title("Late-stage bifurcation of −JCP trials", pad=4)

# break marks
d = 0.007
kwargs = dict(transform=ax_a_top.transAxes, color="#555555", clip_on=False, lw=0.8)
ax_a_top.plot((-d, +d), (-d, +d), **kwargs)
ax_a_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs = dict(transform=ax_a_bot.transAxes, color="#555555", clip_on=False, lw=0.8)
ax_a_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_a_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax_a_bot.axvspan(0.5, 5.5, color="#F8F8F8", zorder=0)
ax_a_bot.text(3.0, 2.72, "pre-bifurcation regime",
              ha="center", va="top", fontsize=8, color="#707070")

add_panel_label(ax_a_top, "(a)")


# (b) Violin + strip plot

vp = ax_b.violinplot(
    data_list,
    positions=np.arange(1, len(strip_methods) + 1),
    widths=0.68,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for i, body in enumerate(vp["bodies"]):
    m = strip_methods[i]
    alpha = 0.10 if m == "L-BFGS" else 0.16
    face = "#A79BCB" if m == "L-BFGS" else COLORS[m]
    edge = "#A79BCB" if m == "L-BFGS" else COLORS[m]
    body.set_facecolor(face)
    body.set_edgecolor(edge)
    body.set_alpha(alpha)

rng2 = np.random.default_rng(RNG_SEED)
for pos, m in zip(np.arange(1, len(strip_methods) + 1), strip_methods):
    vals = results[m]["meta"]["min_rmse"].to_numpy(dtype=float)
    jitter = rng2.normal(0, 0.050, size=len(vals))
    point_alpha = 0.28 if m == "L-BFGS" else 0.40
    point_color = "#9A8CC4" if m == "L-BFGS" else COLORS[m]
    ax_b.scatter(
        np.full(len(vals), pos) + jitter,
        vals,
        s=15,
        alpha=point_alpha,
        color=point_color,
        edgecolors="none",
        zorder=3
    )

minus_pos = 2
ax_b.hlines(minus_conv_center, minus_pos - 0.24, minus_pos + 0.24,
            color="#666666", ls="--", lw=0.9, zorder=4)
ax_b.hlines(minus_fail_center, minus_pos - 0.24, minus_pos + 0.24,
            color="#666666", ls="--", lw=0.9, zorder=4)

ax_b.annotate(
    "good basin",
    xy=(minus_pos + 0.22, minus_conv_center),
    xytext=(2.62, minus_conv_center - 0.001),
    fontsize=7.7, color="#555555",
    ha="left", va="center",
    arrowprops=dict(arrowstyle="-", lw=0.7, color="#777777")
)
ax_b.annotate(
    "bad basin",
    xy=(minus_pos + 0.22, minus_fail_center),
    xytext=(2.62, minus_fail_center - 0.0005),
    fontsize=7.7, color="#555555",
    ha="left", va="center",
    arrowprops=dict(arrowstyle="-", lw=0.7, color="#777777")
)

ax_b.set_xticks(np.arange(1, len(strip_methods) + 1))
ax_b.set_xticklabels(["+JCP", "−JCP", "GN", "L-BFGS"])
ax_b.set_ylabel("Best attained RMSE per trial")
ax_b.set_title("Basin occupancy across trials", pad=4)

finite_vals = np.concatenate([v[np.isfinite(v)] for v in data_list if len(v)])
ymin = float(np.min(finite_vals)) - 0.01
ymax = float(np.max(finite_vals)) + 0.01
ax_b.set_ylim(ymin, ymax)
style_box_axes(ax_b)
add_panel_label(ax_b, "(b)")

# (c) GN-equivalence

for m in ["GN", "D-IPG (-JCP)", "D-IPG (+JCP)"]:
    mean = curve_data[m]["mean"]
    std = curve_data[m]["std"]

    if m == "GN":
        band_alpha = 0.025
        lw = 1.7
        z = 2
    elif m == "D-IPG (-JCP)":
        band_alpha = 0.06
        lw = LINEWIDTHS[m]
        z = 3
    else:
        band_alpha = 0.10
        lw = LINEWIDTHS[m]
        z = 4

    ax_c.fill_between(
        np.arange(1, N_ITERS + 1),
        np.maximum(mean - std, 1e-12),
        np.maximum(mean + std, 1e-12),
        color=COLORS[m], alpha=band_alpha, linewidth=0, zorder=1
    )
    ax_c.semilogy(
        np.arange(1, N_ITERS + 1), mean,
        color=COLORS[m],
        linewidth=lw,
        linestyle=LINESTYLES[m],
        label=m,
        zorder=z,
        solid_capstyle="round"
    )

# reference slope line
y0 = curve_data["D-IPG (+JCP)"]["mean"][0]
y_ref = y0 * np.exp(ref_slope * (np.arange(1, N_ITERS + 1) - 1.0))
ax_c.semilogy(np.arange(1, N_ITERS + 1), y_ref, color="#A0A0A0", lw=0.75, ls="--", zorder=2)

ax_c.set_title("Conditional GN-like convergence", pad=4)
ax_c.set_ylabel("RMSE")
ax_c.set_xlim(0.8, N_ITERS + 0.2)
ax_c.set_ylim(0.075, 1.15)
style_box_axes(ax_c)
ax_c.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#DDDDDD",
    borderpad=0.32,
    handlelength=2.0,
    labelspacing=0.24
)
ax_c.text(
    0.05, 0.08,
    "Among successful −JCP trials,\ncontraction closely matches +JCP",
    transform=ax_c.transAxes,
    fontsize=7.7,
    color="#333333"
)
add_panel_label(ax_c, "(c)")

# ratio strip: tighter and more quantitative
ax_c_ratio.axhline(1.0, color="#BBBBBB", lw=0.8, ls="--", zorder=0)
ax_c_ratio.errorbar(
    ratio_df["iteration"],
    ratio_df["mean"],
    yerr=ratio_df["std"],
    color="#111111",
    marker="o",
    markersize=3.8,
    lw=1.1,
    capsize=1.8,
    zorder=3
)
ax_c_ratio.set_ylim(0.84, 1.02)
ax_c_ratio.set_yticks([0.85, 0.90, 0.95, 1.00])
ax_c_ratio.set_ylabel("−JCP /\n+JCP")
ax_c_ratio.set_xlabel("Iteration")
style_box_axes(ax_c_ratio)
ax_c.tick_params(axis="x", which="both", labelbottom=False)

plt.tight_layout(pad=0.6)
plt.close('all')


fig.savefig("figure2_jcp_mechanism.pdf", format="pdf", bbox_inches="tight", transparent=True)
fig.savefig("figure2_jcp_mechanism.png", dpi=600, bbox_inches="tight")


# Original figure notebook, cell 7

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

# fig 3

# csvs required
summary_files = {
    "Heat-3D": str(RESULTS / "heat3d_cnn_main_results.csv"),
    "Advection–Diffusion-2D": str(RESULTS / "advdiff_degraded_main_results.csv"),
    "Allen–Cahn-2D": str(RESULTS / "allen_cahn2d_main_results.csv"),
    "Navier–Stokes-2D": str(RESULTS / "navier_stokes_main_results.csv"),
    "Darcy-2D": str(RESULTS / "darcy_multisource_main_results.csv"),
    "Heat-2D": str(RESULTS / "heat2d_cnn_main_results.csv"),
}


method_order = ["D-IPG", "GN", "LM", "L-BFGS", "Gradient Descent"]
method_labels = {
    "D-IPG": "D-IPG",
    "GN": "GN",
    "LM": "LM",
    "L-BFGS": "L-BFGS",
    "Gradient Descent": "GD",
}

method_bar_colors = {
    "D-IPG": "#111111",
    "GN": "#4C72B0",
    "LM": "#DD8452",
    "L-BFGS": "#8172B3",
    "Gradient Descent": "#55A868",
}


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def style_box_axes(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#555555")
        spine.set_visible(True)


frames = []
for display_name, path in summary_files.items():
    if not os.path.exists(path):
        print(f"Skipping missing file: {path}")
        continue
    df = pd.read_csv(path)
    needed = ["ablation", "method", "success_rate", "mean_time_sec", "mean_final_rjcp"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.copy()
    df["display_name"] = display_name
    frames.append(df)

if not frames:
    raise ValueError("No summary CSVs were found.")

sum_df = pd.concat(frames, ignore_index=True)

def canon_method(x):
    x = str(x).strip()
    if x == "D-IPG":
        return "D-IPG"
    if x == "Gradient Descent":
        return "Gradient Descent"
    if x in {"GN", "LM", "L-BFGS"}:
        return x
    return x

sum_df["method_clean"] = sum_df["method"].map(canon_method)

# Problem ordering by D-IPG(+JCP) RJCP

dipg_plus = sum_df[(sum_df["method_clean"] == "D-IPG") & (sum_df["ablation"] == "+JCP")].copy()
if dipg_plus.empty:
    raise ValueError("No D-IPG +JCP rows found; cannot order problems by RJCP.")

problem_order = (
    dipg_plus[["display_name", "mean_final_rjcp"]]
    .drop_duplicates(subset=["display_name"])
    .sort_values("mean_final_rjcp")
    ["display_name"]
    .tolist()
)

rjcp_by_problem = (
    dipg_plus[["display_name", "mean_final_rjcp"]]
    .drop_duplicates(subset=["display_name"])
    .set_index("display_name")["mean_final_rjcp"]
    .to_dict()
)


n_methods = len(method_order)
n_probs = len(problem_order)

success_mat = np.full((n_methods, n_probs), np.nan, dtype=float)
time_mat = np.full((n_methods, n_probs), np.nan, dtype=float)

for j, prob in enumerate(problem_order):
    sub_prob = sum_df[sum_df["display_name"] == prob].copy()

    for i, m in enumerate(method_order):
        sub_m = sub_prob[sub_prob["method_clean"] == m].copy()
        if sub_m.empty:
            continue

        if m == "D-IPG":
            row = sub_m[sub_m["ablation"] == "+JCP"]
            row = row.iloc[0] if len(row) else sub_m.iloc[0]
        else:
            row = sub_m[sub_m["ablation"] == "Baseline"]
            row = row.iloc[0] if len(row) else sub_m.iloc[0]

        success_mat[i, j] = float(row["success_rate"])
        time_mat[i, j] = float(row["mean_time_sec"])

# Aggregate success summary for right marginal
# arithmetic mean across available problems

agg_success = np.nanmean(success_mat, axis=1)

cmap = LinearSegmentedColormap.from_list(
    "reliability_map",
    ["#C44E52", "#F7F7F7", "#2C5AA0"],
    N=256
)
norm = Normalize(vmin=0.0, vmax=1.0)


fig = plt.figure(figsize=(14.0, 4.8), dpi=160)
outer = gridspec.GridSpec(
    2, 2,
    width_ratios=[5.2, 1.4],
    height_ratios=[4.2, 1.0],
    wspace=0.28,
    hspace=0.12
)

ax_heat = fig.add_subplot(outer[0, 0])
ax_rjcp = fig.add_subplot(outer[1, 0], sharex=ax_heat)
ax_bar = fig.add_subplot(outer[:, 1])

# heatmap
im = ax_heat.imshow(
    success_mat,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    interpolation="nearest",
    origin="upper"
)

for x in np.arange(-0.5, n_probs, 1):
    ax_heat.axvline(x, color="#E6E6E6", lw=0.8, zorder=2)
for y in np.arange(-0.5, n_methods, 1):
    ax_heat.axhline(y, color="#E6E6E6", lw=0.8, zorder=2)

for i in range(n_methods):
    for j in range(n_probs):
        s = success_mat[i, j]
        t = time_mat[i, j]

        if np.isnan(s):
            ax_heat.add_patch(
                Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F0F0F0", edgecolor="none", zorder=1)
            )
            ax_heat.text(j, i, "—", ha="center", va="center", fontsize=12, color="#888888", zorder=3)
            continue

        txt_color = "white" if (s > 0.72 or s < 0.18) else "black"

        ax_heat.text(
            j, i - 0.10, f"{100*s:.0f}%",
            ha="center", va="center",
            fontsize=10.5,
            fontweight="bold" if i == 0 else "normal",
            color=txt_color,
            zorder=3
        )
        ax_heat.text(
            j, i + 0.18, f"{t:.3g}s",
            ha="center", va="center",
            fontsize=7.8,
            color="#E8E8E8" if txt_color == "white" else "#666666",
            zorder=3
        )

ax_heat.set_xticks(np.arange(n_probs))
ax_heat.set_xticklabels(problem_order, rotation=22, ha="right")
ax_heat.set_yticks(np.arange(n_methods))
ax_heat.set_yticklabels([method_labels[m] for m in method_order])
ax_heat.set_title("Reliability matrix across the RJCP-ordered problem spectrum", pad=8)
style_box_axes(ax_heat)

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.028, pad=0.015)
cbar.set_label("Success rate")
cbar.ax.tick_params(labelsize=8)

# Bottom RJCP strip

rjcp_vals = np.array([rjcp_by_problem[p] for p in problem_order], dtype=float)
xpos = np.arange(n_probs)

ax_rjcp.bar(
    xpos,
    rjcp_vals,
    color="#B8C2CC",
    edgecolor="#7F8C99",
    linewidth=0.7,
    width=0.72
)

for j, v in enumerate(rjcp_vals):
    ax_rjcp.text(j, v * 1.08, f"{v:.2g}", ha="center", va="bottom", fontsize=7.5, color="#444444")

ax_rjcp.set_yscale("log")
ax_rjcp.set_ylabel("RJCP")
ax_rjcp.set_xlabel("Problems ordered by D-IPG(+JCP) RJCP")
ax_rjcp.set_xticks(np.arange(n_probs))
ax_rjcp.set_xticklabels(problem_order, rotation=22, ha="right")
style_box_axes(ax_rjcp)

# Right marginal: mean success

ypos = np.arange(n_methods)[::-1]
bar_vals = agg_success[::-1]
bar_methods = method_order[::-1]

bars = ax_bar.barh(
    ypos,
    bar_vals,
    color=[method_bar_colors[m] for m in bar_methods],
    edgecolor="none",
    height=0.7
)

for y, v in zip(ypos, bar_vals):
    ax_bar.text(
        v + 0.02, y, f"{100*v:.0f}%",
        va="center", ha="left",
        fontsize=9, color="#333333"
    )

ax_bar.set_yticks(ypos)
ax_bar.set_yticklabels([method_labels[m] for m in bar_methods])
ax_bar.set_xlim(0, 1.0)
ax_bar.set_xlabel("Mean success")
ax_bar.set_title("Aggregate reliability", pad=8)
style_box_axes(ax_bar)

plt.tight_layout()
plt.close('all')


fig.savefig("figure3_matrix.png", dpi=600, bbox_inches="tight")
fig.savefig("figure3_matrix.pdf", bbox_inches="tight")


# Original figure notebook, cell 9

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from matplotlib.ticker import LogLocator, NullLocator, MaxNLocator
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Appendix A1


files = {
    "Heat-1D": str(TRACES / "heat1d_traces.csv"),
    "Heat-2D": str(TRACES / "heat2d_traces.csv"),
    "Heat-3D": str(TRACES / "heat3d_traces.csv"),
    "Darcy-2D": str(TRACES / "darcy_traces.csv"),
    "Advection–Diffusion-2D": str(TRACES / "advdiff_traces.csv"),
    "Allen–Cahn-2D": str(TRACES / "allen_cahn_main_traces.csv"),
    "Navier–Stokes-2D": str(TRACES / "navier_stokes_traces.csv"),
}

EPS = 0.30

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
})

METHODS = ["D-IPG (+JCP)", "GN", "LM", "L-BFGS", "Gradient Descent"]
LABELS = {
    "D-IPG (+JCP)": "D-IPG",
    "GN": "Gauss–Newton",
    "LM": "Levenberg–Marquardt",
    "L-BFGS": "L-BFGS",
    "Gradient Descent": "Gradient Descent",
}
COLORS = {
    "D-IPG (+JCP)": "#111111",
    "GN": "#2166AC",
    "LM": "#C94040",
    "L-BFGS": "#7B4898",
    "Gradient Descent": "#3A924A",
}
LW = {
    "D-IPG (+JCP)": 2.4,
    "GN": 1.7,
    "LM": 1.7,
    "L-BFGS": 1.7,
    "Gradient Descent": 1.7,
}
LS = {
    "D-IPG (+JCP)": "-",
    "GN": ":",
    "LM": "-.",
    "L-BFGS": "--",
    "Gradient Descent": "-",
}

def canonicalize_method_name(s):
    s = str(s).strip()
    if s in METHODS:
        return s
    if s == "D-IPG":
        return "D-IPG (+JCP)"
    if s == "GD":
        return "Gradient Descent"
    return s

def first_hit_time_and_iter(grp, eps=EPS):
    grp = grp.sort_values("iteration")
    r = grp["residual"].to_numpy(dtype=float)
    r0 = max(float(r[0]), 1e-12)
    rn = r / r0
    t = grp["time_sec"].to_numpy(dtype=float)
    it = grp["iteration"].to_numpy(dtype=float)
    hit = np.where(rn <= eps)[0]
    if len(hit) == 0:
        return np.inf, np.inf
    k = hit[0]
    return float(t[k]), float(it[k])

def compute_perf_profile(dfs, methods, metric="time", tau_max=120, n_tau=1000):
    ratios = {m: [] for m in methods}
    total = 0

    for task, df in dfs.items():
        grouped_trials = list(df.groupby(["seed", "sample_id"], sort=False))
        for (seed, sid), _ in grouped_trials:
            vals = {}
            for m in methods:
                sub = df[
                    (df["method_clean"] == m) &
                    (df["seed"] == seed) &
                    (df["sample_id"] == sid)
                ].copy()

                if sub.empty:
                    vals[m] = np.inf
                    continue

                hit_time, hit_iter = first_hit_time_and_iter(sub, eps=EPS)
                vals[m] = hit_time if metric == "time" else hit_iter

            finite = [v for v in vals.values() if np.isfinite(v)]
            best = min(finite) if finite else np.inf

            for m in methods:
                v = vals[m]
                if np.isfinite(v) and np.isfinite(best) and best > 1e-12:
                    ratios[m].append(v / best)
                else:
                    ratios[m].append(np.inf)

            total += 1

    tau_grid = np.logspace(0, np.log10(tau_max), n_tau)
    profiles = {
        m: np.array([np.mean(np.array(ratios[m]) <= tau) for tau in tau_grid], dtype=float)
        for m in methods
    }
    return tau_grid, profiles, total

def compute_data_profile(dfs, methods, t_max=None, n_t=1000):
    solved_times = {m: [] for m in methods}
    total = 0
    max_time_seen = 0.0

    for task, df in dfs.items():
        grouped_trials = list(df.groupby(["seed", "sample_id"], sort=False))
        for (seed, sid), _ in grouped_trials:
            for m in methods:
                sub = df[
                    (df["method_clean"] == m) &
                    (df["seed"] == seed) &
                    (df["sample_id"] == sid)
                ].copy()

                if sub.empty:
                    solved_times[m].append(np.inf)
                    continue

                hit_time, _ = first_hit_time_and_iter(sub, eps=EPS)
                solved_times[m].append(hit_time)
                if np.isfinite(hit_time):
                    max_time_seen = max(max_time_seen, hit_time)
            total += 1

    if t_max is None:
        t_max = max_time_seen * 1.25 if max_time_seen > 0 else 1.0

    t_grid = np.logspace(np.log10(max(1e-4, min(1e-2, t_max / 1000))), np.log10(t_max), n_t)
    profiles = {
        m: np.array([np.mean(np.array(solved_times[m]) <= t) for t in t_grid], dtype=float)
        for m in methods
    }
    return t_grid, profiles, total


dfs = {}
missing_files = []

for name, fpath in files.items():
    path = Path(fpath)
    if not path.exists():
        missing_files.append(str(path))
        continue

    df = pd.read_csv(path)
    if "seed" not in df.columns:
        df["seed"] = 0

    required = ["method", "sample_id", "iteration", "residual", "time_sec", "seed"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{fpath} missing columns: {miss}")

    df = df.copy()
    df["method_clean"] = df["method"].map(canonicalize_method_name)
    df = df[df["method_clean"].isin(METHODS)].copy()
    dfs[name] = df

if missing_files:
    print("Missing trace files:")
    for p in missing_files:
        print("  ", p)

if not dfs:
    raise ValueError("No trace CSVs were loaded.")

# pooled N
N_total = sum(len(list(df.groupby(['seed', 'sample_id'], sort=False))) for df in dfs.values())

available_methods = [m for m in METHODS if any((df["method_clean"] == m).any() for df in dfs.values())]


tau_time, prof_time, N_time = compute_perf_profile(dfs, available_methods, metric="time", tau_max=120)
tau_iter, prof_iter, N_iter = compute_perf_profile(dfs, available_methods, metric="iter", tau_max=120)
t_grid, prof_data, N_data = compute_data_profile(dfs, available_methods)


fig = plt.figure(figsize=(13.6, 3.9), dpi=240)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

plot_order = [m for m in available_methods if m != "D-IPG (+JCP)"] + (["D-IPG (+JCP)"] if "D-IPG (+JCP)" in available_methods else [])

# (a) wall-clock Dolan–More
for m in plot_order:
    ax1.semilogx(
        tau_time, prof_time[m],
        color=COLORS[m], lw=LW[m], ls=LS[m],
        solid_capstyle="round", label=LABELS[m],
        zorder=5 if m == "D-IPG (+JCP)" else 3
    )

ax1.axhline(1.0, color="#CCCCCC", lw=0.7, zorder=0)
ax1.axvline(1.0, color="#E0E0E0", lw=0.6, zorder=0)
ax1.set_xlim(1.0, 120)
ax1.set_ylim(-0.02, 1.05)
ax1.set_xlabel(r"Performance ratio $\tau$ (wall-clock time)")
ax1.set_ylabel(r"Fraction of instances $\rho_s(\tau)$")
ax1.set_title("Wall-clock performance profile", pad=4)
ax1.text(0.98, 0.06, f"$N={N_time}$ pooled instances", transform=ax1.transAxes,
         ha="right", va="bottom", fontsize=7.0, color="#666666")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: "1×" if x < 2 else f"{int(x)}×"))
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
style_box_axes(ax1)
ax1.text(-0.14, 1.04, "(a)", transform=ax1.transAxes, fontsize=10.5, fontweight="bold")

# (b) iteration Dolan–More
for m in plot_order:
    ax2.semilogx(
        tau_iter, prof_iter[m],
        color=COLORS[m], lw=LW[m], ls=LS[m],
        solid_capstyle="round",
        zorder=5 if m == "D-IPG (+JCP)" else 3
    )

ax2.axhline(1.0, color="#CCCCCC", lw=0.7, zorder=0)
ax2.axvline(1.0, color="#E0E0E0", lw=0.6, zorder=0)
ax2.set_xlim(1.0, 120)
ax2.set_ylim(-0.02, 1.05)
ax2.set_xlabel(r"Performance ratio $\tau$ (iterations)")
ax2.set_ylabel(r"Fraction of instances $\rho_s(\tau)$")
ax2.set_title("Iteration performance profile", pad=4)
ax2.text(0.98, 0.06, f"$N={N_iter}$ pooled instances", transform=ax2.transAxes,
         ha="right", va="bottom", fontsize=7.0, color="#666666")
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: "1×" if x < 2 else f"{int(x)}×"))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
style_box_axes(ax2)
ax2.text(-0.14, 1.04, "(b)", transform=ax2.transAxes, fontsize=10.5, fontweight="bold")

# (c) data profile
for m in plot_order:
    ax3.semilogx(
        t_grid, prof_data[m],
        color=COLORS[m], lw=LW[m], ls=LS[m],
        solid_capstyle="round",
        zorder=5 if m == "D-IPG (+JCP)" else 3
    )

ax3.axhline(1.0, color="#CCCCCC", lw=0.7, zorder=0)
ax3.set_ylim(-0.02, 1.05)
ax3.set_xlabel(r"Wall-clock budget $t$ (s)")
ax3.set_ylabel("Solved fraction")
ax3.set_title("Data profile", pad=4)
ax3.text(0.98, 0.06, f"$N={N_data}$ pooled instances", transform=ax3.transAxes,
         ha="right", va="bottom", fontsize=7.0, color="#666666")
ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
style_box_axes(ax3)
ax3.text(-0.14, 1.04, "(c)", transform=ax3.transAxes, fontsize=10.5, fontweight="bold")

handles = [
    Line2D([0], [0], color=COLORS[m], lw=LW[m], ls=LS[m], label=LABELS[m])
    for m in plot_order
]
fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5),
           bbox_to_anchor=(0.5, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.close('all')

fig.savefig("appendix_A1_pooled_dolan_more.pdf", bbox_inches="tight", transparent=True)
fig.savefig("appendix_A1_pooled_dolan_more.png", dpi=600, bbox_inches="tight")


# Original figure notebook, cell 10

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Appendix A3

summary_files = {
    "Heat-1D": str(RESULTS / "heat1d_main_results.csv"),
    "Heat-2D": str(RESULTS / "heat2d_cnn_main_results.csv"),
    "Heat-3D": str(RESULTS / "heat3d_cnn_main_results.csv"),
    "Darcy-2D": str(RESULTS / "darcy_multisource_main_results.csv"),
    "Advection–Diffusion-2D": str(RESULTS / "advdiff_degraded_main_results.csv"),
    "Allen–Cahn-2D": str(RESULTS / "allen_cahn2d_main_results.csv"),
    "Navier–Stokes-2D": str(RESULTS / "navier_stokes_main_results.csv"),
}


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def style_box_axes(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#555555")
        spine.set_visible(True)

def add_panel_label(ax, label):
    ax.text(
        -0.14, 1.04, label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left"
    )


rows = []
missing_summary = []

for problem, path in summary_files.items():
    if not os.path.exists(path):
        missing_summary.append(path)
        continue

    df = pd.read_csv(path)
    required = ["ablation", "method", "success_rate"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        print(f"Skipping {problem}; missing summary columns: {miss}")
        continue

    plus = df[(df["method"] == "D-IPG") & (df["ablation"] == "+JCP")].copy()
    if len(plus) == 0:
        continue
    plus = plus.iloc[0]

    if "mean_cosine_with_neg_grad" not in plus.index:
        print(f"Skipping {problem}; cosine column not recorded")
        continue

    rows.append({
        "problem": problem,
        "cosine": float(plus["mean_cosine_with_neg_grad"]),
        "success_rate": float(plus["success_rate"]),
    })

if missing_summary:
    print("Missing summary files for panel (b):")
    for p in missing_summary:
        print("  ", p)

cos_df = pd.DataFrame(rows)
if len(cos_df) == 0:
    raise ValueError("No usable cosine summary rows found for panel (b).")

cos_df = cos_df.sort_values("cosine", ascending=True).reset_index(drop=True)

cmap = LinearSegmentedColormap.from_list(
    "sr_map",
    ["#C44E52", "#F7F7F7", "#2C5AA0"],
    N=256
)
norm = Normalize(vmin=0.0, vmax=1.0)

fig = plt.figure(figsize=(6.6, 4.2), dpi=220)
gs = gridspec.GridSpec(1, 1, figure=fig)

ax2 = fig.add_subplot(gs[0, 0])

# Cosine spectrum

y = np.arange(len(cos_df))
colors = [cmap(norm(v)) for v in cos_df["success_rate"].to_numpy(dtype=float)]

ax2.barh(
    y,
    cos_df["cosine"].to_numpy(dtype=float),
    color=colors,
    edgecolor="none",
    height=0.68
)

ax2.axvline(1.0, color="#BDBDBD", lw=0.8, ls="--", zorder=1)
ax2.text(
    0.995, 1.02, "gradient descent",
    transform=ax2.transAxes,
    ha="right", va="bottom",
    fontsize=7.4, color="#666666"
)

for i, row in cos_df.iterrows():
    ax2.text(
        row["cosine"] + 0.015, i,
        f"{row['cosine']:.3f}",
        va="center", ha="left",
        fontsize=8, color="#333333"
    )
    ax2.text(
        1.085, i,
        f"SR={100*row['success_rate']:.0f}%",
        va="center", ha="left",
        fontsize=8, color="#333333"
    )

if (cos_df["problem"] == "Navier–Stokes-2D").any():
    idx = cos_df.index[cos_df["problem"] == "Navier–Stokes-2D"][0]
    xv = float(cos_df.loc[idx, "cosine"])
    ax2.annotate(
        "strongly corrective,\nyet fully reliable",
        xy=(xv, idx),
        xytext=(0.30, idx - 0.50),
        fontsize=7.2,
        color="#333333",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666")
    )

ax2.set_yticks(y)
ax2.set_yticklabels(cos_df["problem"])
ax2.invert_yaxis()
ax2.set_xlim(0.0, 1.22)
ax2.set_xlabel(r"Cosine similarity with $-\nabla f$")
ax2.set_title("Cosine spectrum of the learned pullback", pad=4)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.03)
cbar.set_label("D-IPG success rate")
cbar.ax.tick_params(labelsize=7)

style_box_axes(ax2)
add_panel_label(ax2, "(b)")

plt.tight_layout()
plt.close('all')

fig.savefig("appendix_A3_cosine.pdf", bbox_inches="tight", transparent=True)
fig.savefig("appendix_A3_cosine.png", dpi=600, bbox_inches="tight")


# Original figure notebook, cell 11

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# Appendix A3

SUMMARY_CANDIDATES = {
    "Heat-1D": [
        str(RESULTS / "heat1d_main_results.csv"),
    ],
    "Heat-2D": [
        str(RESULTS / "heat2d_cnn_main_results.csv"),
    ],
    "Heat-3D": [
        str(RESULTS / "heat3d_cnn_main_results.csv"),
        str(RESULTS / "heat3d_main_results.csv"),
    ],
    "Darcy-2D": [
        str(RESULTS / "darcy_multisource_main_results.csv"),
    ],
    "Advection–Diffusion-2D": [
        str(RESULTS / "advdiff_degraded_main_results.csv"),
        str(RESULTS / "advdiff_main_results.csv"),
    ],
    "Allen–Cahn-2D": [
        str(RESULTS / "allen_cahn2d_main_results.csv"),
    ],
    "Navier–Stokes-2D": [
        str(RESULTS / "navier_stokes_main_results.csv"),
    ],
}

TRACE_CANDIDATES = {
    "Heat-1D": [
        str(TRACES / "heat1d_traces.csv"),
    ],
    "Heat-2D": [
        str(TRACES / "heat2d_traces.csv"),
    ],
    "Heat-3D": [
        str(TRACES / "heat3d_traces.csv"),
    ],
    "Darcy-2D": [
        str(TRACES / "darcy_traces.csv"),
    ],
    "Advection–Diffusion-2D": [
        str(TRACES / "advdiff_traces.csv"),
    ],
    "Allen–Cahn-2D": [
        str(TRACES / "allen_cahn_main_traces.csv"),
    ],
    "Navier–Stokes-2D": [
        str(TRACES / "navier_stokes_traces.csv"),
    ],
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def style_box_axes(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#555555")
        spine.set_visible(True)

def add_panel_label(ax, label):
    ax.text(
        -0.14, 1.04, label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

def resolve_file(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def infer_jcp_condition(df):
    cond = pd.Series([None] * len(df), index=df.index, dtype=object)

    method_str = df["method"].astype(str).str.strip()
    cond[method_str.str.contains(r"\(\+JCP\)", regex=True, na=False)] = "+JCP"
    cond[method_str.str.contains(r"\(-JCP\)", regex=True, na=False)] = "-JCP"

    if "ablation" in df.columns:
        ab = df["ablation"].astype(str).str.strip()
        cond[(cond.isna()) & (ab == "+JCP")] = "+JCP"
        cond[(cond.isna()) & (ab == "-JCP")] = "-JCP"

    return cond

def final_rmse_per_trial(df_cond):
    if "seed" not in df_cond.columns:
        df_cond = df_cond.copy()
        df_cond["seed"] = 0

    vals = []
    for (_, _), grp in df_cond.groupby(["seed", "sample_id"], sort=False):
        grp = grp.sort_values("iteration")
        vals.append(float(grp["rmse_x"].iloc[-1]))
    return np.asarray(vals, dtype=float)

def cv(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    m = arr.mean()
    s = arr.std(ddof=0)
    if not np.isfinite(m) or abs(m) < 1e-12:
        return np.nan
    return s / abs(m)

# Panel (a): RJCP vs CV(final RMSE)

rows = []
for problem in SUMMARY_CANDIDATES.keys():
    summary_path = resolve_file(SUMMARY_CANDIDATES[problem])
    trace_path = resolve_file(TRACE_CANDIDATES[problem])

    if summary_path is None or trace_path is None:
        continue

    sdf = pd.read_csv(summary_path)
    tdf = pd.read_csv(trace_path)

    if "seed" not in tdf.columns:
        tdf["seed"] = 0

    sreq = ["ablation", "method", "mean_final_rjcp"]
    treq = ["method", "sample_id", "iteration", "rmse_x", "seed"]
    if not set(sreq).issubset(sdf.columns):
        raise ValueError(f"{summary_path} missing one of required columns: {sreq}")
    if not set(treq).issubset(tdf.columns):
        raise ValueError(f"{trace_path} missing one of required columns: {treq}")

    tdf = tdf.copy()
    tdf["jcp_cond"] = infer_jcp_condition(tdf)

    for cond in ["+JCP", "-JCP"]:
        srow = sdf[(sdf["method"] == "D-IPG") & (sdf["ablation"] == cond)]
        if len(srow) == 0:
            continue
        rjcp = float(srow.iloc[0]["mean_final_rjcp"])

        tsub = tdf[tdf["jcp_cond"] == cond].copy()
        if len(tsub) == 0:
            continue

        finals = final_rmse_per_trial(tsub)
        cv_final = cv(finals)

        rows.append({
            "problem": problem,
            "cond": cond,
            "rjcp": rjcp,
            "cv_final_rmse": cv_final,
        })

scatter_df = pd.DataFrame(rows)
if len(scatter_df) == 0:
    raise ValueError("No usable rows found for panel (a).")

counts = scatter_df.groupby("problem")["cond"].nunique()
good_probs = counts[counts == 2].index.tolist()
scatter_df = scatter_df[scatter_df["problem"].isin(good_probs)].copy()

fit_df = scatter_df[np.isfinite(scatter_df["rjcp"]) & np.isfinite(scatter_df["cv_final_rmse"])].copy()
fit_df = fit_df[(fit_df["rjcp"] > 0) & (fit_df["cv_final_rmse"] > 0)].copy()

xlog = np.log(fit_df["rjcp"].to_numpy(dtype=float))
ylog = np.log(fit_df["cv_final_rmse"].to_numpy(dtype=float))

slope, intercept = np.polyfit(xlog, ylog, deg=1)

xgrid = np.linspace(xlog.min(), xlog.max(), 200)
ygrid = intercept + slope * xgrid

# Panel (b): NS trajectory bundle width

ns_path = resolve_file(TRACE_CANDIDATES["Navier–Stokes-2D"])
if ns_path is None:
    raise FileNotFoundError("Navier–Stokes trace file not found.")

ns_df = pd.read_csv(ns_path)
if "seed" not in ns_df.columns:
    ns_df["seed"] = 0

ns_req = ["method", "sample_id", "iteration", "residual", "seed"]
if not set(ns_req).issubset(ns_df.columns):
    raise ValueError(f"{ns_path} missing one of required columns: {ns_req}")

ns_df = ns_df.copy()
ns_df["jcp_cond"] = infer_jcp_condition(ns_df)

def make_bundle_df(df_cond):
    trajs = []
    for (_, _), grp in df_cond.groupby(["seed", "sample_id"], sort=False):
        grp = grp.sort_values("iteration").copy()
        r = grp["residual"].to_numpy(dtype=float)
        r0 = max(float(r[0]), 1e-12)
        log_rn = np.log(np.maximum(r / r0, 1e-12))
        trajs.append(pd.DataFrame({
            "iteration": grp["iteration"].to_numpy(dtype=int),
            "log_residual_norm": log_rn,
        }))
    if not trajs:
        return pd.DataFrame(columns=["iteration", "mean", "std", "count"])
    cat = pd.concat(trajs, ignore_index=True)
    agg = cat.groupby("iteration")["log_residual_norm"].agg(["mean", "std", "count"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    return agg

bundle_plus = make_bundle_df(ns_df[ns_df["jcp_cond"] == "+JCP"].copy())
bundle_minus = make_bundle_df(ns_df[ns_df["jcp_cond"] == "-JCP"].copy())

support_frac = 0.85
n_plus0 = int(bundle_plus["count"].max())
n_minus0 = int(bundle_minus["count"].max())
min_plus = max(1, int(np.floor(support_frac * n_plus0)))
min_minus = max(1, int(np.floor(support_frac * n_minus0)))

common_iters = np.intersect1d(bundle_plus["iteration"], bundle_minus["iteration"])

stable_iters = []
for it in common_iters:
    cp = int(bundle_plus.loc[bundle_plus["iteration"] == it, "count"].iloc[0])
    cm = int(bundle_minus.loc[bundle_minus["iteration"] == it, "count"].iloc[0])
    if cp >= min_plus and cm >= min_minus:
        stable_iters.append(it)

stable_iters = np.asarray(stable_iters)

bundle_plus_st = bundle_plus.set_index("iteration").loc[stable_iters].reset_index()
bundle_minus_st = bundle_minus.set_index("iteration").loc[stable_iters].reset_index()

iter_ref = 6
if iter_ref not in set(stable_iters.tolist()):
    iter_ref = int(stable_iters[np.argmin(np.abs(stable_iters - 6))])

std_plus_ref = float(bundle_plus_st.loc[bundle_plus_st["iteration"] == iter_ref, "std"].iloc[0])
std_minus_ref = float(bundle_minus_st.loc[bundle_minus_st["iteration"] == iter_ref, "std"].iloc[0])
pct_wider = 100.0 * (std_minus_ref - std_plus_ref) / max(std_plus_ref, 1e-12)

fig = plt.figure(figsize=(13.8, 4.6), dpi=220)
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.0], wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

for problem in sorted(good_probs):
    pair = scatter_df[scatter_df["problem"] == problem]
    if len(pair) != 2:
        continue
    m = {row["cond"]: row for _, row in pair.iterrows()}
    if "+JCP" not in m or "-JCP" not in m:
        continue

    x0, y0 = float(m["-JCP"]["rjcp"]), float(m["-JCP"]["cv_final_rmse"])
    x1, y1 = float(m["+JCP"]["rjcp"]), float(m["+JCP"]["cv_final_rmse"])

    ax1.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#9A9A9A", alpha=0.85))

plus_df = scatter_df[scatter_df["cond"] == "+JCP"]
minus_df = scatter_df[scatter_df["cond"] == "-JCP"]

ax1.scatter(plus_df["rjcp"], plus_df["cv_final_rmse"],
            s=42, color="#111111", edgecolor="white", linewidth=0.7, label=r"$+$JCP")
ax1.scatter(minus_df["rjcp"], minus_df["cv_final_rmse"],
            s=42, facecolor="white", edgecolor="#777777", linewidth=1.0, label=r"$-$JCP")

xgrid_raw = np.exp(xgrid)
ygrid_raw = np.exp(ygrid)
ax1.plot(xgrid_raw, ygrid_raw, color="#666666", lw=1.4)

mask_ac = (scatter_df["problem"] == "Allen–Cahn-2D") & (scatter_df["cond"] == "-JCP")
if mask_ac.any():
    row = scatter_df[mask_ac].iloc[0]
    ax1.annotate(
        "Allen–Cahn $-\\mathrm{JCP}$:\nRJCP understated by selection bias",
        xy=(float(row["rjcp"]), float(row["cv_final_rmse"])),
        xytext=(0.50, 0.10),
        textcoords="axes fraction",
        fontsize=7.0,
        color="#444444",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666")
    )

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("RJCP")
ax1.set_ylabel("CV of final RMSE")
ax1.set_title("Composition error predicts outcome variability", pad=4)
ax1.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#DDDDDD")

ax2.plot(bundle_plus_st["iteration"], bundle_plus_st["std"],
         color="#111111", lw=2.2, ls="-", label=r"$+$JCP")
ax2.plot(bundle_minus_st["iteration"], bundle_minus_st["std"],
         color="#777777", lw=2.0, ls="--", label=r"$-$JCP")

bp = bundle_plus_st["std"].to_numpy()
bm = bundle_minus_st["std"].to_numpy()
its = bundle_plus_st["iteration"].to_numpy()

ax2.fill_between(its, bp, bm, color="#BDBDBD", alpha=0.25)

ax2.plot([iter_ref, iter_ref], [std_plus_ref, std_minus_ref], color="#666666", lw=0.9)
ax2.text(iter_ref + 0.6, 0.5 * (std_plus_ref + std_minus_ref),
         f"+{pct_wider:.0f}%\nat iter {iter_ref}", fontsize=7.8)

ax2.set_xlabel("Iteration")
ax2.set_ylabel(r"$\mathrm{std}_{\mathrm{trials}}(\log(r_t/r_1))$")
ax2.set_title("Geometric fingerprint on Navier–Stokes-2D", pad=4)
ax2.legend()

for ax, lab in zip([ax1, ax2], ["(a)", "(b)"]):
    style_box_axes(ax)
    add_panel_label(ax, lab)

plt.tight_layout()
plt.close('all')

fig.savefig("appendix_A3_reproducibility_geometry.pdf", bbox_inches="tight", transparent=True)
fig.savefig("appendix_A3_reproducibility_geometry.png", dpi=600, bbox_inches="tight")


# Original figure notebook, cell 12

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Appendix A4

probe_path = str(TRACES / "allen_cahn_probe_ablations_results.csv")
probe_traces_path = str(TRACES / "allen_cahn_probe_ablation_traces.csv")
alpha_results_path = str(TRACES / "allen_cahn_alpha_ablation_results.csv")
alpha_traces_path = str(TRACES / "allen_cahn_alpha_ablation_traces.csv")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def style_box_axes(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#555555")
        spine.set_visible(True)

def add_panel_label(ax, label):
    ax.text(
        -0.14, 1.04, label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left"
    )

if not os.path.exists(probe_path):
    raise FileNotFoundError(probe_path)

probe_df = pd.read_csv(probe_path)

probe_need = ["num_probes", "+JCP_success_rate", "-JCP_success_rate", "+JCP_val_rjcp"]
probe_miss = [c for c in probe_need if c not in probe_df.columns]
if probe_miss:
    raise ValueError(f"{probe_path} missing columns: {probe_miss}")

probe_df = probe_df.sort_values("num_probes").reset_index(drop=True)

# approximate binomial standard errors
n_probe_trials = 80
probe_df["+JCP_se"] = np.sqrt(
    probe_df["+JCP_success_rate"] * (1.0 - probe_df["+JCP_success_rate"]) / n_probe_trials
)
probe_df["-JCP_se"] = np.sqrt(
    probe_df["-JCP_success_rate"] * (1.0 - probe_df["-JCP_success_rate"]) / n_probe_trials
)


if not os.path.exists(alpha_results_path):
    raise FileNotFoundError(alpha_results_path)

alpha_df = pd.read_csv(alpha_results_path)

alpha_need = ["dipg_step_size", "+JCP_success_rate", "+JCP_mean_iters", "+JCP_mean_final_rjcp"]
alpha_miss = [c for c in alpha_need if c not in alpha_df.columns]
if alpha_miss:
    raise ValueError(f"{alpha_results_path} missing columns: {alpha_miss}")

alpha_df = alpha_df.sort_values("dipg_step_size").reset_index(drop=True)


if not os.path.exists(alpha_traces_path):
    raise FileNotFoundError(alpha_traces_path)

trace_df = pd.read_csv(alpha_traces_path)

trace_need = ["dipg_step_size", "sample_id", "iteration", "rmse_x", "ablation", "method"]
trace_miss = [c for c in trace_need if c not in trace_df.columns]
if trace_miss:
    raise ValueError(f"{alpha_traces_path} missing columns: {trace_miss}")

trace_df = trace_df.copy()
trace_df = trace_df[
    (trace_df["ablation"].astype(str).str.strip() == "+JCP") &
    (trace_df["method"].astype(str).str.strip() == "D-IPG")
].copy()

if len(trace_df) == 0:
    raise ValueError("No +JCP D-IPG rows found in alpha traces.")

trace_df["seed"] = 0
trace_df = trace_df.sort_values(["dipg_step_size", "seed", "sample_id", "iteration"]).reset_index(drop=True)

# Build normalized RMSE curves from traces

def normalized_rmse_curve(df_alpha):
    parts = []
    for (_, _), grp in df_alpha.groupby(["seed", "sample_id"], sort=False):
        grp = grp.sort_values("iteration").copy()
        r0 = max(float(grp["rmse_x"].iloc[0]), 1e-12)
        grp["norm_rmse"] = grp["rmse_x"] / r0
        parts.append(grp[["iteration", "norm_rmse"]])

    if not parts:
        return pd.DataFrame(columns=["iteration", "mean", "std", "low", "high"])

    out = pd.concat(parts, ignore_index=True)
    agg = out.groupby("iteration")["norm_rmse"].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    agg["low"] = np.maximum(agg["mean"] - agg["std"], 1e-12)
    agg["high"] = np.maximum(agg["mean"] + agg["std"], 1e-12)
    return agg

alpha_curves = {}
for step_size, sub in trace_df.groupby("dipg_step_size", sort=True):
    alpha_curves[float(step_size)] = normalized_rmse_curve(sub)

if len(alpha_curves) == 0:
    raise ValueError("No usable step-size trajectories found.")


step_vals_sorted = sorted(alpha_curves.keys())
cmap = matplotlib.colormaps["Greys"]

if len(step_vals_sorted) == 1:
    color_list = ["#111111"]
else:
    color_list = [cmap(0.30 + 0.55 * i / (len(step_vals_sorted) - 1)) for i in range(len(step_vals_sorted))]

step_to_color = {a: c for a, c in zip(step_vals_sorted, color_list)}


fig = plt.figure(figsize=(13.8, 5.2), dpi=220)
outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.15], wspace=0.28)

left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[0], height_ratios=[1.0, 0.95], hspace=0.08
)
right = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[1], height_ratios=[1.0, 0.85], hspace=0.10
)

ax1 = fig.add_subplot(left[0])   # probe success
ax2 = fig.add_subplot(left[1], sharex=ax1)  # probe rjcp
ax3 = fig.add_subplot(right[0])  # step-size trajectories
ax4 = fig.add_subplot(right[1])  # step-size success bars

# (a) Probe count: success rate vs k
xk = probe_df["num_probes"].to_numpy(dtype=float)

ax1.errorbar(
    xk,
    probe_df["+JCP_success_rate"].to_numpy(dtype=float),
    yerr=probe_df["+JCP_se"].to_numpy(dtype=float),
    color="#111111",
    linestyle="-",
    marker="o",
    markersize=4.5,
    linewidth=2.0,
    capsize=2.0,
    label=r"$+$JCP",
    zorder=3,
)

ax1.errorbar(
    xk,
    probe_df["-JCP_success_rate"].to_numpy(dtype=float),
    yerr=probe_df["-JCP_se"].to_numpy(dtype=float),
    color="#7A7A7A",
    linestyle="--",
    marker="o",
    markersize=4.5,
    linewidth=2.0,
    capsize=2.0,
    label=r"$-$JCP",
    zorder=3,
)

best_idx = probe_df["+JCP_success_rate"].to_numpy(dtype=float).argmax()
best_k = float(probe_df.iloc[best_idx]["num_probes"])
best_sr = float(probe_df.iloc[best_idx]["+JCP_success_rate"])

ax1.annotate(
    "k=4 optimal" if np.isclose(best_k, 4.0) else f"k={best_k:g} optimal",
    xy=(best_k, best_sr),
    xytext=(best_k, min(1.02, best_sr + 0.13)),
    ha="center",
    va="bottom",
    fontsize=7.6,
    color="#333333",
    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#666666")
)

ax1.axhline(1.0, color="#CCCCCC", lw=0.7, ls="--", zorder=1)
ax1.set_ylim(0.0, 1.05)
ax1.set_ylabel("Success rate")
ax1.set_title("Probe-count sensitivity", pad=4)
ax1.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#DDDDDD")

# (b) Probe count: validation RJCP vs k

ax2.plot(
    xk,
    probe_df["+JCP_val_rjcp"].to_numpy(dtype=float),
    color="#4C72B0",
    linestyle="-",
    marker="o",
    linewidth=2.0,
    markersize=4.5,
    zorder=3
)

min_idx = probe_df["+JCP_val_rjcp"].to_numpy(dtype=float).argmin()
min_k = float(probe_df.iloc[min_idx]["num_probes"])
min_rjcp = float(probe_df.iloc[min_idx]["+JCP_val_rjcp"])

ax2.annotate(
    "val RJCP minimum",
    xy=(min_k, min_rjcp),
    xytext=(min_k + 0.35, min_rjcp + 0.015),
    fontsize=7.4,
    color="#333333",
    arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666")
)

# vertical reference at the empirically best k
ax1.axvline(best_k, color="#C7C7C7", lw=0.8, ls="--", zorder=1)
ax2.axvline(best_k, color="#C7C7C7", lw=0.8, ls="--", zorder=1)

ax2.set_xlabel(r"Number of Hutchinson probes $k$")
ax2.set_ylabel("Validation RJCP")
ax2.set_xticks(xk)
ax2.set_xticklabels([str(int(v)) for v in xk])


# (c) Step size: normalized RMSE trajectories

for step_size in step_vals_sorted:
    curve = alpha_curves[step_size]
    color = step_to_color[step_size]
    lw = 2.4 if np.isclose(step_size, 1.0) else 1.9

    ax3.plot(
        curve["iteration"].to_numpy(dtype=float),
        curve["mean"].to_numpy(dtype=float),
        color=color,
        linewidth=lw,
        label=rf"$\alpha={step_size:g}$",
        zorder=4 if np.isclose(step_size, 1.0) else 3
    )

    if np.isclose(step_size, 1.0):
        ax3.fill_between(
            curve["iteration"].to_numpy(dtype=float),
            curve["low"].to_numpy(dtype=float),
            curve["high"].to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2
        )

ax3.set_ylabel("Normalized RMSE")
ax3.set_title("Step-size sensitivity", pad=4)
ax3.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

if 0.25 in alpha_curves:
    c = alpha_curves[0.25]
    ax3.annotate(
        r"$\alpha=0.25$: long unsuccessful trajectory",
        xy=(float(c["iteration"].iloc[-1]), float(c["mean"].iloc[-1])),
        xytext=(0.40, 0.83),
        textcoords="axes fraction",
        fontsize=7.2,
        color="#444444",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666")
    )

if 1.0 in alpha_curves:
    c = alpha_curves[1.0]
    idx = min(len(c) - 1, 5)
    ax3.annotate(
        r"$\alpha=1.0$: fast successful descent",
        xy=(float(c["iteration"].iloc[idx]), float(c["mean"].iloc[idx])),
        xytext=(0.47, 0.18),
        textcoords="axes fraction",
        fontsize=7.2,
        color="#444444",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666")
    )

if 1.5 in alpha_curves:
    c = alpha_curves[1.5]
    idx = min(len(c) - 1, 4)
    ax3.annotate(
        r"$\alpha=1.5$: fast but wrong basin",
        xy=(float(c["iteration"].iloc[idx]), float(c["mean"].iloc[idx])),
        xytext=(0.56, 0.50),
        textcoords="axes fraction",
        fontsize=7.2,
        color="#444444",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666")
    )

ax3.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#DDDDDD", ncol=2)

# (d) Step size: success-rate bars

alpha_df["dipg_step_size"] = alpha_df["dipg_step_size"].astype(float)
alpha_df = alpha_df.sort_values("dipg_step_size").reset_index(drop=True)

bar_colors = [step_to_color.get(float(a), "#999999") for a in alpha_df["dipg_step_size"].to_numpy(dtype=float)]
x_alpha = np.arange(len(alpha_df))
sr_vals = alpha_df["+JCP_success_rate"].to_numpy(dtype=float)

ax4.bar(
    x_alpha,
    sr_vals,
    color=bar_colors,
    edgecolor="none",
    width=0.68,
    zorder=2
)

for xi, sr in enumerate(sr_vals):
    ax4.text(
        xi,
        sr + 0.03,
        f"{100*sr:.0f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#333333"
    )


ax4.axhline(1.0, color="#CCCCCC", lw=0.7, ls="--", zorder=1)
ax4.set_ylim(0.0, 1.05)
ax4.set_ylabel("Success rate")
ax4.set_xlabel(r"Step size $\alpha$")
ax4.set_xticks(x_alpha)
ax4.set_xticklabels([f"{a:g}" for a in alpha_df["dipg_step_size"].to_numpy(dtype=float)])

for ax, lab in zip([ax1, ax2, ax3, ax4], ["(a)", "(b)", "(c)", "(d)"]):
    style_box_axes(ax)
    add_panel_label(ax, lab)

plt.tight_layout()
plt.close('all')

fig.savefig("appendix_A4_hyperparameter_analysis.pdf", bbox_inches="tight", transparent=True)
fig.savefig("appendix_A4_hyperparameter_analysis.png", dpi=600, bbox_inches="tight")
