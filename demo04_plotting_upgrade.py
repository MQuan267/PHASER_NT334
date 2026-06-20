from demo00_conf_upgrade import *
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from phaser.evaluation import dist_stats, MetricMaker
from phaser.plotting import kde_ax, cm_ax, hist_fig, eer_ax, roc_ax

print("Running script.")

# Load
le = load("./demo_outputs/LabelEncoders_upgrade.bz2")
df_h = load("./demo_outputs/Hashes.df.bz2")
df_d = load("./demo_outputs/Distances_upgrade.df.bz2")

# Split into intra and inter
intra_df = df_d[df_d["class"] == 1]
inter_df = df_d[df_d["class"] == 0]

# Generate triplet combinations without 'orig'
triplets = np.array(
    np.meshgrid(
        le["a"].classes_,
        [t for t in le["t"].classes_ if t != "orig"],
        le["m"].classes_
    )
).T.reshape(-1, 3)

# ── Histogram: 1 hình per transform ──────────────────────────────────────
for transform in [t for t in le["t"].classes_ if t != "orig"]:
    print(f"\nGenerating macro stats for '{transform}'")
    stats = dist_stats(intra_df, le, transform, style=False)
    print(stats.to_latex())

    fig = hist_fig(intra_df, le, transform)
    fig.savefig(fname=f"./demo_outputs/figs/04up-hist_intra_{transform}.png")
    plt.close()

    fig = hist_fig(inter_df, le, transform)
    fig.savefig(fname=f"./demo_outputs/figs/04up-hist_inter_{transform}.png")
    plt.close()

# ── KDE, CM, EER, ROC: loop qua tất cả triplets ──────────────────────────
max_fpr = 0.01

for a_s, t_s, m_s in triplets:
    print(f"\nAnalysing '{a_s}_{t_s}_{m_s}'")

    a_label = le["a"].transform(np.array(a_s).ravel())
    m_label = le["m"].transform(np.array(m_s).ravel())

    data = df_d.query(f"algo == {a_label[0]} and metric == {m_label[0]}").copy()

    if data.empty:
        print(f"  No data for {a_s}_{t_s}_{m_s}, skipping.")
        continue

    # KDE plot
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
    ax = kde_ax(data, t_s, le, fill=True, title=f"{a_s} - {m_s} - {t_s}", ax=ax)
    fig.savefig(fname=f"./demo_outputs/figs/04up-{a_s}_{m_s}_{t_s}_kde.png")
    plt.close()

    y_true = data["class"]
    y_similarity = data[t_s]

    mm = MetricMaker(y_true=y_true, y_similarity=y_similarity, weighted=False)

    # CM tại ngưỡng EER
    cm_eer = mm.get_cm(threshold=mm.eer_thresh, normalize=None)
    print(f"  CM EER@{mm.eer_thresh:.4f}, EER={mm.eer_score:.4f}")
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
    ax = cm_ax(cm=cm_eer, class_labels=le["c"].classes_, values_format=".0f", ax=ax)
    fig.savefig(f"./demo_outputs/figs/04up-{a_s}_{m_s}_{t_s}_cm_eer_{mm.eer_thresh:.4f}.png")
    plt.close()

    # EER curve
    fpr_threshold = mm.get_fpr_threshold(max_fpr=max_fpr)
    cm_fpr = mm.get_cm(fpr_threshold, normalize="none")
    print(f"  FPR={max_fpr} -> threshold={fpr_threshold:.4f}")

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
    _ = ax.axhline(max_fpr, label=f"FPR={max_fpr:.2f}", color="red")
    _ = ax.axvline(
        float(fpr_threshold),
        label=f"FPR={max_fpr:.2f}@{fpr_threshold:.2f}",
        color="red",
        linestyle="--",
    )
    ax = eer_ax(mm.fpr, mm.tpr, mm.thresholds, threshold=mm.eer_thresh, legend="", ax=ax)
    fig.savefig(fname=f"./demo_outputs/figs/04up-{a_s}_{m_s}_{t_s}_eer.png")
    plt.close()

    # CM tại FPR threshold
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
    ax = cm_ax(cm_fpr, class_labels=le["c"].classes_, values_format=".0f", ax=ax)
    fig.savefig(f"./demo_outputs/figs/04up-{a_s}_{m_s}_{t_s}_cm_fpr_{fpr_threshold:.4f}.png")
    plt.close()

    # ROC curve
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=FIGSIZE, constrained_layout=True)
    ax = roc_ax(mm.fpr, mm.tpr, roc_auc=mm.auc, legend=f"{a_s}_{m_s}_{t_s}", ax=ax)
    fig.savefig(fname=f"./demo_outputs/figs/04up-{a_s}_{m_s}_{t_s}_ROC_AUC{mm.auc:.4f}.png")
    plt.close()

print("\nScript completed.")
