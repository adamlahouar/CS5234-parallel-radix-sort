from baseline import radix_sort_gpu as baseline
from opt1 import radix_sort_gpu as opt1
from opt2 import radix_sort_gpu as opt2
import numpy as np
import time
import pickle
import sys


if __name__ == "__main__":

    arr = np.random.randint(0, 10000, 16, dtype=np.int32)

    # Ensure the array is a 1D numpy array
    arr = np.asarray(arr).flatten()

    # Run the baseline and optimized implementations
    baseline_sorted, baseline_time = baseline(arr)
    opt1_sorted, opt1_time = opt1(arr, 256)
    opt2_sorted, opt2_time = opt2(arr, 256)

    # Verify that all implementations produce the same sorted array
    assert np.array_equal(baseline_sorted, opt1_sorted), "Opt1 output does not match Baseline"
    assert np.array_equal(baseline_sorted, opt2_sorted), "Opt2 output does not match Baseline"

    print(f"original array: {arr}")
    print(f"Baseline sorted array: {baseline_sorted}")
    print(f"Opt1 sorted array: {opt1_sorted}")
    print(f"Opt2 sorted array: {opt2_sorted}")

    print("All implementations produce the same sorted array.")