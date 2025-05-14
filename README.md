# CS5234 Final Project—CPU and GPU Parallel Radix Sort

## Instructions for Building and Running the Project

### CPU

1. Clone the repository and navigate to the project directory
2. Use the included `CMakeLists.txt` file to build the project (`cmake . && make`)
3. Run `./benchmark` to benchmark the CPU sorts, run `./validate_sort` to validate the correctness of the CPU sorts
4. To plot the benchmark results, install the necessary Python packages using the included `requirements.txt` file (
   `pip install -r requirements.txt`) and run `python plotting/plot_cpu_results.py`


### GPU

You need to have CUDA-12.4 Driver installed and at least 2-4 GB of memory available for experiments.

1. First install numba and numpy by using this command

`$ pip3 install numba numpy`

2. You may run any of opt1 and opt2 codes by using a similar command to:

`$ python3 opt1.py <number of threads per block>`

3. to verify the results of the baseline and the optimization, run check_verification.py by using this command:

`$ python3 check_verification.py`

4. To plot the data, use evaluation.py