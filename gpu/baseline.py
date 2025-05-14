import numpy as np
from numba import cuda
import time
import pickle

@cuda.jit
def histogram_kernel(d_arr, exp, d_hist):
    tid = cuda.grid(1)
    if tid < d_arr.size:
        digit = (d_arr[tid] // exp) % 10
        cuda.atomic.add(d_hist, digit, 1)

@cuda.jit
def scatter_kernel(d_arr, d_output, exp, d_offsets):
    tid = cuda.grid(1)
    if tid < d_arr.size:
        digit = (d_arr[tid] // exp) % 10
        pos = cuda.atomic.add(d_offsets, digit, 1)
        d_output[pos] = d_arr[tid]

def radix_sort_gpu(arr):
    n = arr.size
    d_arr = cuda.to_device(arr)
    d_output = cuda.device_array_like(arr)

    max_element = np.max(arr)
    exp = 1

    start_event = cuda.event()
    end_event = cuda.event()
    start_event.record()

    while max_element // exp > 0:
        d_hist = cuda.to_device(np.zeros(10, dtype=np.int32))
        histogram_kernel[(n + 255) // 256, 256](d_arr, exp, d_hist)
        cuda.synchronize()

        # Copy histogram to host and compute prefix sum
        h_hist = d_hist.copy_to_host()
        h_offsets = np.zeros_like(h_hist)
        for i in range(1, 10):
            h_offsets[i] = h_offsets[i - 1] + h_hist[i - 1]

        d_offsets = cuda.to_device(h_offsets)
        scatter_kernel[(n + 255) // 256, 256](d_arr, d_output, exp, d_offsets)
        cuda.synchronize()

        d_arr, d_output = d_output, d_arr
        exp *= 10

    end_event.record()
    end_event.synchronize()
    gpu_time = cuda.event_elapsed_time(start_event, end_event) / 1000.0

    sorted_arr = d_arr.copy_to_host()
    return sorted_arr, gpu_time

if __name__ == "__main__":
    results = []
    sizes = [2**i * 10**6 for i in range(10)]
    for size in sizes:
        arr = np.random.randint(0, 1e9, size, dtype=np.int32)
        # print(arr) 
        sorted_gpu, gpu_time = radix_sort_gpu(arr.copy())
        # print(sorted_gpu)
        memory_usage = arr.nbytes
        
        results.append({
            "size": int(size),
            "gpu_time": float(gpu_time),
            "memory_usage": int(memory_usage)
        })
        
        print(f"Array size: {size}")
        print(f"GPU Time: {gpu_time:.6f} sec")
        print(f"Memory Usage: {memory_usage / (1024**2):.2f} MB\n")

    with open("radix_sort_results.pkl", "wb") as f:
        pickle.dump(results, f)
