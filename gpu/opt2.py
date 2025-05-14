import numpy as np
import numba
from numba import cuda
import time
import sys
import pickle

@cuda.jit
def counting_sort_kernel(d_arr, d_output, exp, n, threads_per_block):
    shared_count = cuda.shared.array(10, numba.int32)
    shared_offset = cuda.shared.array(10, numba.int32)

    tid = cuda.grid(1)
    lane = cuda.threadIdx.x

    # Initialize shared count for each block
    if lane < 10:
        shared_count[lane] = 0
    cuda.syncthreads()

    # Count occurrences of each digit in the block
    if tid < n:
        digit = (d_arr[tid] // exp) % 10
        cuda.atomic.add(shared_count, digit, 1)
    cuda.syncthreads()

    # Calculate prefix sum within the block
    if lane < 10:
        offset = 0
        for i in range(lane):
            offset += shared_count[i]
        shared_offset[lane] = offset
    cuda.syncthreads()

    # Determine the correct position in the output array
    if tid < n:
        digit = (d_arr[tid] // exp) % 10
        pos = cuda.atomic.add(shared_offset, digit, 1)
        d_output[pos + cuda.blockIdx.x * threads_per_block] = d_arr[tid] 

def radix_sort_gpu(arr, threads_per_block):
    n = arr.size
    d_arr = cuda.to_device(arr)
    d_output = cuda.device_array_like(arr)

    max_element = np.max(arr)
    exp = 1

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()

    temp_d_arr = d_arr

    blockspergrid = (n + threads_per_block - 1) // threads_per_block

    while max_element // exp > 0:
        counting_sort_kernel[blockspergrid, threads_per_block](temp_d_arr, d_output, exp, n, threads_per_block)
        temp_d_arr, d_output = d_output, temp_d_arr
        exp *= 10

    end_event.record()
    end_event.synchronize()
    gpu_time = cuda.event_elapsed_time(start_event, end_event) / 1000.0

    sorted_arr = temp_d_arr.copy_to_host()
    return sorted_arr, gpu_time

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python a.py <threads_per_block>")
        sys.exit(1)

    try:
        threads_per_block = int(sys.argv[1])
        if threads_per_block <= 0:
            raise ValueError
    except ValueError:
        print("Error: <threads_per_block> must be a positive integer.")
        sys.exit(1)

    results = []
    sizes = [2**i * 10**6 for i in range(10)]
    for size in sizes:
        arr = np.random.randint(0, 1e9, size, dtype=np.int32)
        sorted_gpu, gpu_time = radix_sort_gpu(arr.copy(), threads_per_block)

        flops = (size * np.log10(size)) / gpu_time if gpu_time > 0 else 0
        memory_usage = arr.nbytes

        results.append({
            "size": int(size),
            "gpu_time": float(gpu_time),
            "flops": float(flops),
            "memory_usage": int(memory_usage)
        })

        print(f"Array size: {size}")
        print(f"GPU Time: {gpu_time:.6f} sec")
        print(f"Approx FLOPS: {flops:.2f}")
        print(f"Memory Usage: {memory_usage / (1024**2):.2f} MB\n")

    with open("radix_sort_results.pkl", "wb") as f:
        pickle.dump(results, f)