from demo00_conf_upgrade import *
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from phaser.evaluation import ComputeMetrics, make_bit_weights
from phaser.similarities import IntraDistance, InterDistance, find_inter_samplesize
from phaser.plotting import bit_weights_ax, auc_cmp_fig

print("Running script.")

# Load
le = load("./demo_outputs/LabelEncoders_upgrade.bz2")
df_h = load("./demo_outputs/Hashes.df.bz2")
df_d = load("./demo_outputs/Distances_upgrade.df.bz2")
n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))

# Generate triplet combinations without 'orig'
triplets = np.array(
    np.meshgrid(
        le["a"].classes_,
        [t for t in le["t"].classes_ if t != "orig"],
        le["m"].classes_
    )
).T.reshape(-1, 3)

# Compute metrics for all available triplets
print(f"Number of triplets to analyse: {len(triplets)}")
cm = ComputeMetrics(le, df_d, df_h, analyse_bits=True, n_jobs=1)
m, b = cm.fit(triplets=triplets)

print(f"Performance without bit weights:")
print(m.groupby(["Algorithm", "Metric"])[["AUC", "EER"]].agg(["mean", "std"]))
print(m.to_string())

# Plot bit frequency for each triplet
print(f"Plotting bit weights for each triplet")
for triplet in list(b.keys()):
    fig, ax = plt.subplots(1, 1, figsize=(5, 1.5), constrained_layout=True)
    _ = bit_weights_ax(b[triplet], ax=ax)
    fig.savefig(f"./demo_outputs/figs/03up-bit_analysis_{triplet}.png")
    plt.close()

# Create bit_weights (algo, metric)
weights = make_bit_weights(b, le)

# Plot applied bitweights
for pair in list(weights.keys()):
    fig, ax = plt.subplots(1, 1, figsize=(5, 1.5), constrained_layout=True)
    _ = bit_weights_ax(weights[pair].reshape(-1, 1), ax=ax)
    fig.savefig(f"./demo_outputs/figs/03up-bit_weights_{pair}.png")
    plt.close()

intra_df_w = IntraDistance(METR_dict, le, 1, weights, progress_bar=True).fit(df_h)
inter_df_w = InterDistance(METR_dict, le, 0, weights, n_samples, progress_bar=True).fit(df_h)
df_d_w = pd.concat([intra_df_w, inter_df_w])

cm_w = ComputeMetrics(le, df_d_w, df_h, analyse_bits=False, n_jobs=1)
m_w, _ = cm_w.fit(triplets=triplets)
print(f"Performance with bit weights:")
print(m_w.groupby(["Algorithm", "Metric"])[["AUC", "EER"]].agg(["mean", "std"]))
print(m_w.to_string())

# Plot AUC comparison cho cả 3 metric
for metric in list(METR_dict.keys()):
    fig = auc_cmp_fig(m, m_w, metric=metric)
    fig.savefig(f"./demo_outputs/figs/03up_auc_cmp_{metric}.png")
    plt.close()
    print(f"Saved AUC comparison for {metric}")

print("Script finished")
