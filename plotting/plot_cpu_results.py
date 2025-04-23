import matplotlib.pyplot as plt
import pandas as pd


def plot_execution_time_vs_input_size(df):
    df = df[df['Input Distribution'] == 'Uniform']
    plt.figure(figsize=(7, 7))

    for sorter, style in [('std::sort', 'o-'), ('SerialRadixSort', 'x-')]:
        results = df[df['Sorter'] == sorter]
        plt.plot(results['Input Size'], results['Average Execution Time [s]'], style, label=sorter)

    parallel_radix_results = df[df['Sorter'] == 'ParallelRadixSort']
    for thread_count in parallel_radix_results['Thread Count'].unique():
        thread_data = parallel_radix_results[parallel_radix_results['Thread Count'] == thread_count]
        plt.plot(thread_data['Input Size'], thread_data['Average Execution Time [s]'], 'x--',
                 label=f'Parallel Radix ({thread_count}t)')

    plt.xlabel('Input Size [million]')
    plt.ylabel('Average Execution Time [s]')
    plt.title('Execution Time vs Input Size')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(df['Input Size'].unique(), [f"{int(x):,}" for x in df['Input Size'].unique()], rotation=0)
    plt.legend()
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('execution_time_vs_input_size.png', dpi=300)
    plt.show()


def plot_distribution_comparison(df):
    plt.figure(figsize=(7, 7))
    distribution_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    sorter_styles = {
        'std::sort': {'color': 'tab:blue', 'marker': 'o', 'linestyle': '--'},
        'SerialRadixSort': {'color': 'tab:orange', 'marker': 's', 'linestyle': '-'},
        'ParallelRadixSort': {'color': 'tab:green', 'marker': '^', 'linestyle': '-.'}
    }

    for sorter in ['std::sort', 'SerialRadixSort', 'ParallelRadixSort']:
        sorter_data = df[(df['Sorter'] == sorter) & (df['Thread Count'] == 8)] if sorter == 'ParallelRadixSort' else df[
            df['Sorter'] == sorter]
        for i, dist in enumerate(df['Input Distribution'].unique()):
            dist_data = sorter_data[sorter_data['Input Distribution'] == dist]
            plt.plot(dist_data['Input Size'], dist_data['Average Execution Time [s]'],
                     label=f'{sorter} - {dist}', color=distribution_colors[i],
                     marker=sorter_styles[sorter]['marker'], linestyle=sorter_styles[sorter]['linestyle'], alpha=0.5)

    plt.xlabel('Input Size [million]')
    plt.ylabel('Average Execution Time [s]')
    plt.title('Comparison of Sorting Algorithms by Input Distribution')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(df['Input Size'].unique(), [f"{int(x):,}" for x in df['Input Size'].unique()], rotation=0)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=300)
    plt.show()


def plot_speedup(df, baseline):
    merged = get_speedup(df, baseline)
    plt.figure(figsize=(7, 7))

    if baseline == 'std::sort':
        serial_radix_results = merged[merged['Sorter'] == 'SerialRadixSort']
        plt.plot(serial_radix_results['Input Size'], serial_radix_results['Speedup'], 'o-', label='Serial Radix')

    parallel_radix_results = merged[merged['Sorter'] == 'ParallelRadixSort']
    for thread_count in parallel_radix_results['Thread Count'].unique():
        thread_data = parallel_radix_results[parallel_radix_results['Thread Count'] == thread_count]
        plt.plot(thread_data['Input Size'], thread_data['Speedup'], 'x--', label=f'Parallel Radix ({thread_count}t)')

    plt.xlabel('Input Size [million]')
    plt.ylabel('Speedup [x]')
    plt.title(f'Speedup vs Input Size (Baseline: {baseline})')
    plt.xscale('log')
    plt.ylim(bottom=0)
    plt.xticks(df['Input Size'].unique(), [f"{int(x):,}" for x in df['Input Size'].unique()], rotation=0)
    plt.legend()
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'speedup_comparison_{baseline.replace("::", "_")}.png', dpi=300)
    plt.show()


def get_speedup(df, baseline):
    df = df[df['Input Distribution'] == 'Uniform']
    baseline_results = df[df['Sorter'] == baseline]
    radix_results = df[df['Sorter'] != baseline]

    merged = radix_results.merge(
        baseline_results[['Input Size', 'Input Distribution', 'Average Execution Time [s]']],
        on=['Input Size', 'Input Distribution'],
        suffixes=('', '_Baseline')
    )
    merged['Speedup'] = merged['Average Execution Time [s]_Baseline'] / merged['Average Execution Time [s]']
    return merged


def plot_parallel_efficiency(df):
    merged = get_speedup(df, baseline='SerialRadixSort')
    merged['Efficiency'] = merged['Speedup'] / merged['Thread Count']
    plt.figure(figsize=(7, 7))

    parallel_radix_results = merged[merged['Sorter'] == 'ParallelRadixSort']
    for thread_count in parallel_radix_results['Thread Count'].unique():
        thread_data = parallel_radix_results[parallel_radix_results['Thread Count'] == thread_count]
        plt.plot(thread_data['Input Size'], thread_data['Efficiency'], 'x--', label=f'Parallel Radix ({thread_count}t)')

    plt.xlabel('Input Size [million]')
    plt.ylabel('Efficiency')
    plt.title('Parallel Efficiency vs Input Size')
    plt.xscale('log')
    plt.ylim(bottom=0)
    plt.xticks(df['Input Size'].unique(), [f"{int(x):,}" for x in df['Input Size'].unique()], rotation=0)
    plt.legend()
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('parallel_efficiency.png', dpi=300)
    plt.show()


def main():
    df = pd.read_csv('../cpu/benchmark_results.csv')
    df['Input Size'] /= 1_000_000.0
    plot_execution_time_vs_input_size(df)
    plot_distribution_comparison(df)
    plot_speedup(df, baseline='std::sort')
    plot_speedup(df, baseline='SerialRadixSort')
    plot_parallel_efficiency(df)


if __name__ == '__main__':
    main()
