import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle

matplotlib.rcParams['figure.dpi'] = 300


def plot_execution_times_and_speedups(df, target_sorter, label_prefix, baseline_sorters, multi_threaded_baselines=None):
    if multi_threaded_baselines is None:
        multi_threaded_baselines = []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))

    all_labels = set()

    for sorter in baseline_sorters:
        if sorter in multi_threaded_baselines:
            for thread_count in sorted(df[df['Sorter'] == sorter]['Thread Count'].unique()):
                all_labels.add(f'{sorter} ({thread_count}t)')
        else:
            all_labels.add(sorter)

    for thread_count in sorted(df[df['Sorter'] == target_sorter]['Thread Count'].unique()):
        all_labels.add(f'{label_prefix} ({thread_count}t)')

    color_cycle = cycle(plt.cm.tab10.colors)
    label_to_color = {label: next(color_cycle) for label in sorted(all_labels)}

    seen_labels = set()

    # execution time plot
    target_df = plot_execution_time(ax1, baseline_sorters, df, label_prefix, label_to_color, multi_threaded_baselines,
                                    seen_labels, target_sorter)

    # speedup plot
    handles, labels = plot_speedup(ax1, ax2, baseline_sorters, df, label_prefix, label_to_color, seen_labels,
                                   target_df)

    fig.legend(handles, labels, loc='upper center', ncols=3, fontsize='small')
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    plt.savefig(f'{target_sorter}_execution_time_speedup.png', dpi=300)
    plt.show()


def plot_execution_time(ax1, baseline_sorters, df, label_prefix, label_to_color, multi_threaded_baselines, seen_labels,
                        target_sorter):
    for sorter in baseline_sorters:
        sorter_df = df[df['Sorter'] == sorter]
        if sorter in multi_threaded_baselines:
            for thread_count in sorted(sorter_df['Thread Count'].unique()):
                data = sorter_df[sorter_df['Thread Count'] == thread_count]
                label = f'{sorter} ({thread_count}t)'
                ax1.plot(data['Input Size'], data['Average Execution Time [s]'],
                         label=label if label not in seen_labels else None,
                         color=label_to_color[label],
                         linestyle='--', marker='x', alpha=0.5)
                seen_labels.add(label)
        else:
            data = sorter_df
            label = sorter
            ax1.plot(data['Input Size'], data['Average Execution Time [s]'],
                     label=label if label not in seen_labels else None,
                     color=label_to_color[label],
                     linestyle='--', marker='x', alpha=0.5)
            seen_labels.add(label)
    target_df = df[df['Sorter'] == target_sorter]
    for thread_count in sorted(target_df['Thread Count'].unique()):
        data = target_df[target_df['Thread Count'] == thread_count]
        label = f'{label_prefix} ({thread_count}t)'
        ax1.plot(data['Input Size'], data['Average Execution Time [s]'],
                 label=label if label not in seen_labels else None,
                 color=label_to_color[label],
                 linestyle='--', marker='x', alpha=0.5)
        seen_labels.add(label)

    ax1.set_ylabel('Avg Execution Time [s]')
    ax1.set_title('Execution Time vs Input Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks(sorted(df['Input Size'].unique()))
    ax1.set_xticklabels([f"{int(x):,}" for x in sorted(df['Input Size'].unique())])
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
    return target_df


def plot_speedup(ax1, ax2, baseline_sorters, df, label_prefix, label_to_color, seen_labels, target_df):
    baseline_sorter = baseline_sorters[0]
    baseline_df = df[df['Sorter'] == baseline_sorter]
    baseline_single_thread = baseline_df[baseline_df['Thread Count'] == 1]
    baseline_time_map = dict(zip(
        baseline_single_thread['Input Size'],
        baseline_single_thread['Average Execution Time [s]']
    ))
    for thread_count in sorted(target_df['Thread Count'].unique()):
        data = target_df[target_df['Thread Count'] == thread_count]
        speedups = []
        for _, row in data.iterrows():
            input_size = row['Input Size']
            baseline_time = baseline_time_map.get(input_size)
            if baseline_time is not None and row['Average Execution Time [s]'] > 0:
                speedup = baseline_time / row['Average Execution Time [s]']
                speedups.append((input_size, speedup))

        if speedups:
            input_sizes, speedup_values = zip(*sorted(speedups))
            label = f'{label_prefix} ({thread_count}t)'
            ax2.plot(input_sizes, speedup_values,
                     label=label if label not in seen_labels else None,
                     color=label_to_color[label],
                     linestyle='--', marker='x', alpha=0.5)
            seen_labels.add(label)
    ax2.set_xlabel('Input Size [million]')
    ax2.set_ylabel(f'Speedup (vs {baseline_sorter})')
    ax2.set_title('Speedup vs Input Size')
    ax2.set_xscale('log')
    ax2.set_xticks(sorted(df['Input Size'].unique()))
    ax2.set_xticklabels([f"{int(x):,}" for x in sorted(df['Input Size'].unique())])
    ax2.grid(True, which='major', linestyle='--', linewidth=0.5)
    handles, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels


def plot_distribution_differences(df):
    plt.figure(figsize=(5, 5))

    for distribution, data in df.groupby('Input Distribution'):
        plt.plot(data['Input Size'], data['Average Execution Time [s]'], label=distribution, linestyle='--', marker='x',
                 alpha=0.5)

    plt.xlabel('Input Size [million]')
    plt.ylabel('Average Execution Time [s]')
    plt.title('Execution Time vs Input Size by Distribution')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(df['Input Size'].unique(), [f"{int(x):,}" for x in df['Input Size'].unique()], rotation=0)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('allOpts_distribution_differences.png', dpi=300)
    plt.show()


def main():
    df = pd.read_csv('../cpu_benchmark_results.csv')
    df['Input Size'] /= 1_000_000.0
    df_grouped_by_distribution = df.groupby(['Sorter', 'Input Size', 'Thread Count'])[
        'Average Execution Time [s]'].mean().reset_index()

    target_sorters = ['BaseParallel', 'ParallelOptA', 'ParallelOptB', 'ParallelOptC', 'ParallelOptAC',
                      'ParallelAllOpts']
    label_prefixes = ['BaseParallel', 'OptA', 'OptB', 'OptC', 'OptAC', 'AllOpts']
    baseline_sorters = ['SerialRadixSort', 'std::sort']

    for target_sorter, label_prefix in zip(target_sorters, label_prefixes):
        plot_execution_times_and_speedups(df_grouped_by_distribution, target_sorter, label_prefix, baseline_sorters)

    df_all_opts_distribution = df[df['Sorter'] == 'ParallelAllOpts']
    df_all_opts_distribution = df_all_opts_distribution[df_all_opts_distribution['Thread Count'] == 8]
    plot_distribution_differences(df_all_opts_distribution)


if __name__ == '__main__':
    main()
