import pickle
import matplotlib.pyplot as plt

def load_results(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Load your results
results = load_results("nexp/opt2_dist.pkl")

# Group results by distribution type
distributions = {}
for entry in results:
    dist = entry["distribution"]
    if dist not in distributions:
        distributions[dist] = {"gpu_times": []}
    distributions[dist]["gpu_times"].append(entry["gpu_time"])

# These are your full xtick labels
full_xtick_labels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Slice from index 1 onwards
xtick_labels = full_xtick_labels[1:]  # [2, 4, 8, 16, ..., 512]
indices = list(range(len(xtick_labels)))  # New indices: 0,1,2,3,...

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

markers = ["o", "s", "^", "D", "v", ">", "<", "h"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

for (dist_name, data), marker, color in zip(distributions.items(), markers, colors):
    gpu_times = data["gpu_times"][1:]  # Slice from index 1 onwards
    ax.plot(
        indices,
        gpu_times,
        label=f"{dist_name}",
        marker=marker,
        linestyle="-",
        color=color,
    )

# Set xticks to be evenly spaced by index
ax.set_xticks(indices)
ax.set_xticklabels(xtick_labels)

# Labels, title
ax.set_xlabel("Array Size (Millions)")
ax.set_ylabel("GPU Execution Time (seconds)")
ax.set_title("GPU Execution Time vs Array Size (Grouped by Distribution)")

ax.set_yscale("log")   # y-axis in log scale to handle execution time range
ax.legend()
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
