import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_results(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

filenames = {
    "Baseline": "radix_sort_results_v100.pkl",
    "Opt2 64 Threads": "nexp/opt2_64t.pkl",
    "Opt2 128 Threads": "nexp/opt2_128t.pkl",
    "Opt2 256 Threads": "nexp/opt2_256t.pkl",
    "Opt2 512 Threads": "nexp/opt2_512t.pkl",
    "Opt2 1024 Threads": "nexp/opt2_1024t.pkl",
}

results = {key: load_results(file) for key, file in filenames.items()}

sizes, gpu_times = {}, {}
for key, res in results.items():
    sizes[key] = [r["size"] for r in res][1:]
    gpu_times[key] = [r["gpu_time"] / (1000 if "Optimized" in key else 1) for r in res][1:]

# Precompute the baseline GPU times
baseline_times = gpu_times["Baseline"]

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
styles = {
    "Opt2 64 Threads": (">", "-", "tab:blue"),
    "Opt2 128 Threads": ("H", "-", "tab:olive"),
    "Opt2 256 Threads": ("o", "-", "tab:cyan"),
    "Opt2 512 Threads": ("<", "-", "tab:purple"),
    "Opt2 1024 Threads": ("v", "-", "tab:red"),
}

# First plot: GPU Times
for key, (marker, linestyle, color) in styles.items():
    axes[0].plot(sizes[key], gpu_times[key], label=f"{key} GPU Time", marker=marker, linestyle=linestyle, color=color)

# Second plot: Speedup relative to baseline
for key, (marker, linestyle, color) in styles.items():
    speedup = [b/a for a, b in zip(gpu_times[key], baseline_times)]
    axes[1].plot(sizes[key], speedup, label=f"{key} Speedup", marker=marker, linestyle=linestyle, color=color)

axes[0].set_yscale("log")

xaxis = list(map(lambda x: int(x/10**6), sizes['Opt2 64 Threads']))
for ax, ylabel, title in zip(axes, ["Time (seconds)", "Speedup (X)"],
                             ["GPU Execution Time for Radix Sort", "Speedup Relative to Baseline"]):
    ax.set_xlabel("Array Size (Millions)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(sizes['Opt2 64 Threads'])
    ax.set_xticklabels(xaxis)
    ax.legend()
    ax.set_title(title)
    ax.grid(True)

plt.tight_layout()
plt.show()
