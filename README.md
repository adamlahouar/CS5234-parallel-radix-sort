# CS5234 Final Project—CPU and GPU Parallel Radix Sort

## Instructions for Building and Running the Project

### CPU

1. Clone the repository and navigate to the project directory
2. Use the included `CMakeLists.txt` file to build the project (`cmake . && make`)
3. Run `./benchmark` to benchmark the CPU sorts, run `./validate_sort` to validate the correctness of the CPU sorts
4. To plot the benchmark results, install the necessary Python packages using the included `requirements.txt` file (
   `pip install -r requirements.txt`) and run `python plotting/plot_cpu_results.py`