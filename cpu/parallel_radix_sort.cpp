#include "parallel_radix_sort.h"

#include <omp.h>
#include <memory>
#include <cstring>
#include <iostream>

namespace BaseParallel {
    constexpr int BITS_PER_PASS = 1;
    constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

    void computeLocalHistograms(const int *arr, const int n, int **localHistograms, const int shift) {
        #pragma omp parallel default(none) shared(arr, n, localHistograms, shift)
        {
            const int tid = omp_get_thread_num();
            int *local = localHistograms[tid];

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const int bucket = (arr[i] >> shift) & (NUM_BUCKETS - 1);
                local[bucket]++;
            }
        }
    }

    void computeGlobalHistogram(int **localHistograms, int *globalHistogram, const int numThreads) {
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            for (int thread = 0; thread < numThreads; ++thread) {
                globalHistogram[bucket] += localHistograms[thread][bucket];
            }
        }
    }

    void computePrefixSums(const int *globalHistogram, int *prefixSums) {
        int sum = 0;
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            prefixSums[bucket] = sum;
            sum += globalHistogram[bucket];
        }
    }

    void computeThreadOffsets(int **localHistograms, const int *prefixSums, int **threadOffsets, const int numThreads) {
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            int offset = prefixSums[bucket];

            for (int thread = 0; thread < numThreads; ++thread) {
                threadOffsets[thread][bucket] = offset;
                offset += localHistograms[thread][bucket];
            }
        }
    }


    void scatterToBuffer(const int *arr, const int n, int *buffer, int **threadOffsets, const int shift) {
        #pragma omp parallel default(none) shared(arr, n, buffer, threadOffsets, shift)
        {
            const int tid = omp_get_thread_num();
            int *local = threadOffsets[tid];

            #pragma omp for
            for (int i = 0; i < n; ++i) {
                const int value = arr[i];
                const int bucket = (value >> shift) & (NUM_BUCKETS - 1);
                const int pos = local[bucket]++;
                buffer[pos] = value;
            }
        }
    }

    void sort(const int *inputArray, int *outputArray, const int n, const int numThreads) {
        omp_set_num_threads(numThreads);

        auto *arr = new int[n];
        std::memcpy(arr, inputArray, n * sizeof(int));

        auto buffer = new int[n];

        for (int shift = 0; shift < sizeof(int) * 8; shift += BITS_PER_PASS) {
            auto **localHistograms = new int *[numThreads];
            for (int thread = 0; thread < numThreads; ++thread) {
                localHistograms[thread] = new int[NUM_BUCKETS]{};
            }
            computeLocalHistograms(arr, n, localHistograms, shift);

            auto *globalHistogram = new int[NUM_BUCKETS]{};
            computeGlobalHistogram(localHistograms, globalHistogram, numThreads);

            auto *prefixSums = new int[NUM_BUCKETS]{};
            computePrefixSums(globalHistogram, prefixSums);

            auto **threadOffsets = new int *[numThreads];
            for (int thread = 0; thread < numThreads; ++thread) {
                threadOffsets[thread] = new int[NUM_BUCKETS]{};
            }
            computeThreadOffsets(localHistograms, prefixSums, threadOffsets, numThreads);

            scatterToBuffer(arr, n, buffer, threadOffsets, shift);

            std::swap(arr, buffer);

            for (int thread = 0; thread < numThreads; ++thread) {
                delete[] localHistograms[thread];
                delete[] threadOffsets[thread];
            }

            delete[] localHistograms;
            delete[] threadOffsets;
            delete[] globalHistogram;
            delete[] prefixSums;
        }

        std::memcpy(outputArray, arr, n * sizeof(int));

        delete[] arr;
        delete[] buffer;
    }
}

// 8 bits per pass, single parallel region, better memory management
namespace ParallelOptA {
    constexpr int BITS_PER_PASS = 8;
    constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

    void computeLocalHistograms(const int *__restrict arr, const int n, int *__restrict localHistograms,
                                const int shift) {
        std::memset(localHistograms, 0, NUM_BUCKETS * sizeof(int));
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int bucket = (arr[i] >> shift) & (NUM_BUCKETS - 1);
            localHistograms[bucket]++;
        }
    }


    void computeGlobalHistogram(int **localHistograms, int *globalHistogram, const int numThreads) {
        std::memset(globalHistogram, 0, NUM_BUCKETS * sizeof(int));
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            for (int t = 0; t < numThreads; ++t) {
                globalHistogram[bucket] += localHistograms[t][bucket];
            }
        }
    }

    void computePrefixSums(const int *globalHistogram, int *prefixSums) {
        int sum = 0;
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            prefixSums[bucket] = sum;
            sum += globalHistogram[bucket];
        }
    }

    void computeThreadOffsets(int **localHistograms, const int *prefixSums, int **threadOffsets, const int numThreads) {
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            int offset = prefixSums[bucket];
            for (int t = 0; t < numThreads; ++t) {
                threadOffsets[t][bucket] = offset;
                offset += localHistograms[t][bucket];
            }
        }
    }

    void scatterToBuffer(const int *arr, const int n, int *buffer, int *threadOffsets, const int shift) {
        int privateOffsets[NUM_BUCKETS];
        std::memcpy(privateOffsets, threadOffsets, NUM_BUCKETS * sizeof(int));

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int value = arr[i];
            const int bucket = (value >> shift) & (NUM_BUCKETS - 1);
            buffer[privateOffsets[bucket]++] = value;
        }

        std::memcpy(threadOffsets, privateOffsets, NUM_BUCKETS * sizeof(int));
    }

    void sort(int *inputArray, int *outputArray, const int n, const int numThreads) {
        omp_set_num_threads(numThreads);

        int *arr = inputArray;
        int *buffer = outputArray;

        auto localHistograms = std::make_unique<int *[]>(numThreads);
        const auto globalHistogram = std::make_unique<int[]>(NUM_BUCKETS);
        const auto prefixSums = std::make_unique<int[]>(NUM_BUCKETS);
        auto threadOffsets = std::make_unique<int *[]>(numThreads);

        std::unique_ptr<int[]> histogramArrays[numThreads];
        std::unique_ptr<int[]> offsetArrays[numThreads];

        for (int t = 0; t < numThreads; ++t) {
            histogramArrays[t] = std::make_unique<int[]>(NUM_BUCKETS);
            offsetArrays[t] = std::make_unique<int[]>(NUM_BUCKETS);

            localHistograms[t] = histogramArrays[t].get();
            threadOffsets[t] = offsetArrays[t].get();
        }

        for (int shift = 0; shift < sizeof(int) * 8; shift += BITS_PER_PASS) {
            std::memset(globalHistogram.get(), 0, NUM_BUCKETS * sizeof(int));
            std::memset(prefixSums.get(), 0, NUM_BUCKETS * sizeof(int));

            #pragma omp parallel default(none) shared(arr, buffer, localHistograms, globalHistogram, prefixSums, threadOffsets, shift, n, numThreads)
            {
                const int tid = omp_get_thread_num();

                computeLocalHistograms(arr, n, localHistograms[tid], shift);
                #pragma omp barrier

                #pragma omp single
                {
                    computeGlobalHistogram(localHistograms.get(), globalHistogram.get(), numThreads);
                    computePrefixSums(globalHistogram.get(), prefixSums.get());
                    computeThreadOffsets(localHistograms.get(), prefixSums.get(), threadOffsets.get(), numThreads);
                }
                #pragma omp barrier

                scatterToBuffer(arr, n, buffer, threadOffsets[tid], shift);
            }

            std::swap(arr, buffer);
        }
    }
}

// max value calculation with reduced bit processing accordingly
namespace ParallelOptB {
    constexpr int BITS_PER_PASS = 1;
    constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

    void computeLocalHistograms(const int *arr, const int n, int **localHistograms, const int shift) {
        #pragma omp parallel default(none) shared(arr, n, localHistograms, shift)
        {
            const int tid = omp_get_thread_num();
            int *local = localHistograms[tid];

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const int bucket = (arr[i] >> shift) & (NUM_BUCKETS - 1);
                local[bucket]++;
            }
        }
    }

    auto computeLocalHistogramsWithMax(const int *arr, const int n, int **localHistograms, const int shift) {
        int globalMax = 0;

        #pragma omp parallel default(none) shared(arr, n, localHistograms, shift) reduction(max: globalMax)
        {
            const int tid = omp_get_thread_num();
            int *local = localHistograms[tid];

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const int val = arr[i];
                const int bucket = (val >> shift) & (NUM_BUCKETS - 1);
                local[bucket]++;
                globalMax = std::max(globalMax, val);
            }
        }

        const int bitsInMax = static_cast<int>(sizeof(int)) * 8 - __builtin_clz(globalMax);
        return std::max(bitsInMax, BITS_PER_PASS);
    }

    void computeGlobalHistogram(int **localHistograms, int *globalHistogram, const int numThreads) {
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            for (int thread = 0; thread < numThreads; ++thread) {
                globalHistogram[bucket] += localHistograms[thread][bucket];
            }
        }
    }

    void computePrefixSums(const int *globalHistogram, int *prefixSums) {
        int sum = 0;
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            prefixSums[bucket] = sum;
            sum += globalHistogram[bucket];
        }
    }

    void computeThreadOffsets(int **localHistograms, const int *prefixSums, int **threadOffsets, const int numThreads) {
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            int offset = prefixSums[bucket];

            for (int thread = 0; thread < numThreads; ++thread) {
                threadOffsets[thread][bucket] = offset;
                offset += localHistograms[thread][bucket];
            }
        }
    }


    void scatterToBuffer(const int *arr, const int n, int *buffer, int **threadOffsets, const int shift) {
        #pragma omp parallel default(none) shared(arr, n, buffer, threadOffsets, shift)
        {
            const int tid = omp_get_thread_num();
            int *local = threadOffsets[tid];

            #pragma omp for
            for (int i = 0; i < n; ++i) {
                const int value = arr[i];
                const int bucket = (value >> shift) & (NUM_BUCKETS - 1);
                const int pos = local[bucket]++;
                buffer[pos] = value;
            }
        }
    }

    void sort(const int *inputArray, int *outputArray, const int n, const int numThreads) {
        omp_set_num_threads(numThreads);

        auto arr = new int[n];
        std::memcpy(arr, inputArray, n * sizeof(int));

        auto buffer = new int[n];

        auto numBitsToProcess = sizeof(int) * 8;


        for (int shift = 0; shift < numBitsToProcess; shift += BITS_PER_PASS) {
            auto **localHistograms = new int *[numThreads];
            for (int thread = 0; thread < numThreads; ++thread) {
                localHistograms[thread] = new int[NUM_BUCKETS]{};
            }

            if (shift == 0) {
                numBitsToProcess = computeLocalHistogramsWithMax(arr, n, localHistograms, shift);
            } else {
                computeLocalHistograms(arr, n, localHistograms, shift);
            }

            auto *globalHistogram = new int[NUM_BUCKETS]{};
            computeGlobalHistogram(localHistograms, globalHistogram, numThreads);

            auto *prefixSums = new int[NUM_BUCKETS]{};
            computePrefixSums(globalHistogram, prefixSums);

            auto **threadOffsets = new int *[numThreads];
            for (int thread = 0; thread < numThreads; ++thread) {
                threadOffsets[thread] = new int[NUM_BUCKETS]{};
            }
            computeThreadOffsets(localHistograms, prefixSums, threadOffsets, numThreads);

            scatterToBuffer(arr, n, buffer, threadOffsets, shift);

            std::swap(arr, buffer);

            for (int thread = 0; thread < numThreads; ++thread) {
                delete[] localHistograms[thread];
                delete[] threadOffsets[thread];
            }

            delete[] localHistograms;
            delete[] threadOffsets;
            delete[] globalHistogram;
            delete[] prefixSums;
        }

        std::memcpy(outputArray, arr, n * sizeof(int));


        delete[] arr;
        delete[] buffer;
    }
}

// thread-local output buffers
namespace ParallelOptC {
    constexpr int BITS_PER_PASS = 1;
    constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

    constexpr int LOCAL_BUFFER_SIZE = 128;

    void computeLocalHistograms(const int *arr, const int n, int **localHistograms, const int shift) {
        #pragma omp parallel default(none) shared(arr, n, localHistograms, shift)
        {
            const int tid = omp_get_thread_num();
            int *local = localHistograms[tid];

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const int bucket = (arr[i] >> shift) & (NUM_BUCKETS - 1);
                local[bucket]++;
            }
        }
    }

    void computeGlobalHistogram(int **localHistograms, int *globalHistogram, const int numThreads) {
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            for (int thread = 0; thread < numThreads; ++thread) {
                globalHistogram[bucket] += localHistograms[thread][bucket];
            }
        }
    }

    void computePrefixSums(const int *globalHistogram, int *prefixSums) {
        int sum = 0;
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            prefixSums[bucket] = sum;
            sum += globalHistogram[bucket];
        }
    }

    void computeThreadOffsets(int **localHistograms, const int *prefixSums, int **threadOffsets, const int numThreads) {
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            int offset = prefixSums[bucket];

            for (int thread = 0; thread < numThreads; ++thread) {
                threadOffsets[thread][bucket] = offset;
                offset += localHistograms[thread][bucket];
            }
        }
    }

    void scatterToBuffer(const int *arr, const int n, int *buffer, int **threadOffsets, const int shift) {
        #pragma omp parallel default(none) shared(arr, n, buffer, threadOffsets, shift)
        {
            const int tid = omp_get_thread_num();
            int *privateOffsets = threadOffsets[tid];

            int localBuffers[NUM_BUCKETS][LOCAL_BUFFER_SIZE];
            int bufferCounts[NUM_BUCKETS] = {};

            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const int value = arr[i];
                const int bucket = (value >> shift) & (NUM_BUCKETS - 1);

                localBuffers[bucket][bufferCounts[bucket]++] = value;

                if (bufferCounts[bucket] == LOCAL_BUFFER_SIZE) {
                    std::memcpy(&buffer[privateOffsets[bucket]], localBuffers[bucket], LOCAL_BUFFER_SIZE * sizeof(int));
                    privateOffsets[bucket] += LOCAL_BUFFER_SIZE;
                    bufferCounts[bucket] = 0;
                }
            }

            for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
                for (int j = 0; j < bufferCounts[bucket]; ++j) {
                    buffer[privateOffsets[bucket]++] = localBuffers[bucket][j];
                }
            }
        }
    }


    void sort(const int *inputArray, int *outputArray, const int n, const int numThreads) {
        omp_set_num_threads(numThreads);

        auto arr = new int[n];
        std::memcpy(arr, inputArray, n * sizeof(int));

        auto buffer = new int[n];

        for (int shift = 0; shift < sizeof(int) * 8; shift += BITS_PER_PASS) {
            auto **localHistograms = new int *[numThreads];
            for (int thread = 0; thread < numThreads; ++thread) {
                localHistograms[thread] = new int[NUM_BUCKETS]{};
            }
            computeLocalHistograms(arr, n, localHistograms, shift);

            auto *globalHistogram = new int[NUM_BUCKETS]{};
            computeGlobalHistogram(localHistograms, globalHistogram, numThreads);

            auto *prefixSums = new int[NUM_BUCKETS]{};
            computePrefixSums(globalHistogram, prefixSums);

            auto **threadOffsets = new int *[numThreads];
            for (int thread = 0; thread < numThreads; ++thread) {
                threadOffsets[thread] = new int[NUM_BUCKETS]{};
            }
            computeThreadOffsets(localHistograms, prefixSums, threadOffsets, numThreads);

            scatterToBuffer(arr, n, buffer, threadOffsets, shift);

            std::swap(arr, buffer);

            for (int thread = 0; thread < numThreads; ++thread) {
                delete[] localHistograms[thread];
                delete[] threadOffsets[thread];
            }

            delete[] localHistograms;
            delete[] threadOffsets;
            delete[] globalHistogram;
            delete[] prefixSums;
        }

        std::memcpy(outputArray, arr, n * sizeof(int));

        delete[] arr;
        delete[] buffer;
    }
}

// OptA + OptC
namespace ParallelOptAC {
    constexpr int BITS_PER_PASS = 8;
    constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;
    constexpr int LOCAL_BUFFER_SIZE = 128;


    void computeLocalHistograms(const int *__restrict arr, const int n,
                                int *__restrict localHistograms,
                                const int shift) {
        std::memset(localHistograms, 0, NUM_BUCKETS * sizeof(int));
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int bucket = (arr[i] >> shift) & (NUM_BUCKETS - 1);
            localHistograms[bucket]++;
        }
    }


    void computeGlobalHistogram(int **localHistograms, int *globalHistogram, const int numThreads) {
        std::memset(globalHistogram, 0, NUM_BUCKETS * sizeof(int));
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            for (int t = 0; t < numThreads; ++t) {
                globalHistogram[bucket] += localHistograms[t][bucket];
            }
        }
    }

    void computePrefixSums(const int *globalHistogram, int *prefixSums) {
        int sum = 0;
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            prefixSums[bucket] = sum;
            sum += globalHistogram[bucket];
        }
    }

    void computeThreadOffsets(int **localHistograms, const int *prefixSums, int **threadOffsets, const int numThreads) {
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            int offset = prefixSums[bucket];
            for (int t = 0; t < numThreads; ++t) {
                threadOffsets[t][bucket] = offset;
                offset += localHistograms[t][bucket];
            }
        }
    }

    void scatterToBuffer(const int *arr, const int n, int *buffer, const int *threadOffsets, const int shift) {
        int localBuffers[NUM_BUCKETS][LOCAL_BUFFER_SIZE];
        int bufferCounts[NUM_BUCKETS] = {};

        int privateOffsets[NUM_BUCKETS];
        std::memcpy(privateOffsets, threadOffsets, NUM_BUCKETS * sizeof(int));

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int value = arr[i];
            const int bucket = (value >> shift) & (NUM_BUCKETS - 1);

            localBuffers[bucket][bufferCounts[bucket]++] = value;

            if (bufferCounts[bucket] == LOCAL_BUFFER_SIZE) {
                std::memcpy(&buffer[privateOffsets[bucket]], localBuffers[bucket], LOCAL_BUFFER_SIZE * sizeof(int));
                privateOffsets[bucket] += LOCAL_BUFFER_SIZE;
                bufferCounts[bucket] = 0;
            }
        }

        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            if (bufferCounts[bucket] > 0) {
                for (int j = 0; j < bufferCounts[bucket]; ++j) {
                    buffer[privateOffsets[bucket]++] = localBuffers[bucket][j];
                }
            }
        }
    }

    void sort(int *inputArray, int *outputArray, const int n, const int numThreads) {
        omp_set_num_threads(numThreads);

        int *arr = inputArray;
        int *buffer = outputArray;

        auto localHistograms = std::make_unique<int *[]>(numThreads);
        const auto globalHistogram = std::make_unique<int[]>(NUM_BUCKETS);
        const auto prefixSums = std::make_unique<int[]>(NUM_BUCKETS);
        auto threadOffsets = std::make_unique<int *[]>(numThreads);

        std::unique_ptr<int[]> histogramArrays[numThreads];
        std::unique_ptr<int[]> offsetArrays[numThreads];

        for (int t = 0; t < numThreads; ++t) {
            histogramArrays[t] = std::make_unique<int[]>(NUM_BUCKETS);
            offsetArrays[t] = std::make_unique<int[]>(NUM_BUCKETS);

            localHistograms[t] = histogramArrays[t].get();
            threadOffsets[t] = offsetArrays[t].get();
        }

        for (int shift = 0; shift < sizeof(int) * 8; shift += BITS_PER_PASS) {
            std::memset(globalHistogram.get(), 0, NUM_BUCKETS * sizeof(int));
            std::memset(prefixSums.get(), 0, NUM_BUCKETS * sizeof(int));

            #pragma omp parallel default(none) shared(arr, buffer, localHistograms, globalHistogram, prefixSums, threadOffsets, shift, n, numThreads)
            {
                const int tid = omp_get_thread_num();

                computeLocalHistograms(arr, n, localHistograms[tid], shift);
                #pragma omp barrier

                #pragma omp single
                {
                    computeGlobalHistogram(localHistograms.get(), globalHistogram.get(), numThreads);
                    computePrefixSums(globalHistogram.get(), prefixSums.get());
                    computeThreadOffsets(localHistograms.get(), prefixSums.get(), threadOffsets.get(), numThreads);
                }
                #pragma omp barrier

                scatterToBuffer(arr, n, buffer, threadOffsets[tid], shift);
            }

            std::swap(arr, buffer);
        }
    }
}

// all optimizations
namespace ParallelAllOpts {
    constexpr int BITS_PER_PASS = 8;
    constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;
    constexpr int LOCAL_BUFFER_SIZE = 128;


    void computeLocalHistograms(const int *__restrict arr, const int n, int *__restrict localHistograms,
                                const int shift) {
        std::memset(localHistograms, 0, NUM_BUCKETS * sizeof(int));
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int bucket = (arr[i] >> shift) & (NUM_BUCKETS - 1);
            localHistograms[bucket]++;
        }
    }

    auto computeLocalHistogramsWithMax(const int *arr, const int n, int **localHistograms, const int shift,
                                       const int tid, int *threadLocalMax) {
        int localMax = arr[0];
        std::memset(localHistograms[tid], 0, NUM_BUCKETS * sizeof(int));

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int value = arr[i];
            localMax = std::max(localMax, value);
            const int bucket = (value >> shift) & (NUM_BUCKETS - 1);
            localHistograms[tid][bucket]++;
        }

        threadLocalMax[tid] = localMax;
    }

    auto computeNumBits(auto threadLocalMax, const int numThreads) {
        int globalMax = threadLocalMax[0];
        for (int i = 1; i < numThreads; ++i) {
            globalMax = std::max(globalMax, threadLocalMax[i]);
        }

        const int bitsInMax = static_cast<int>(sizeof(int)) * 8 - __builtin_clz(globalMax);
        return std::max(bitsInMax, BITS_PER_PASS);
    }


    void computeGlobalHistogram(int **localHistograms, int *globalHistogram, const int numThreads) {
        std::memset(globalHistogram, 0, NUM_BUCKETS * sizeof(int));
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            for (int t = 0; t < numThreads; ++t) {
                globalHistogram[bucket] += localHistograms[t][bucket];
            }
        }
    }

    void computePrefixSums(const int *globalHistogram, int *prefixSums) {
        std::memset(prefixSums, 0, NUM_BUCKETS * sizeof(int));
        int sum = 0;
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            prefixSums[bucket] = sum;
            sum += globalHistogram[bucket];
        }
    }

    void computeThreadOffsets(int **localHistograms, const int *prefixSums, int **threadOffsets, const int numThreads) {
        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            int offset = prefixSums[bucket];
            for (int t = 0; t < numThreads; ++t) {
                threadOffsets[t][bucket] = offset;
                offset += localHistograms[t][bucket];
            }
        }
    }


    void scatterToBuffer(const int *arr, const int n, int *buffer, const int *threadOffsets, const int shift) {
        int localBuffers[NUM_BUCKETS][LOCAL_BUFFER_SIZE];
        int bufferCounts[NUM_BUCKETS] = {};

        int privateOffsets[NUM_BUCKETS];
        std::memcpy(privateOffsets, threadOffsets, NUM_BUCKETS * sizeof(int));

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int value = arr[i];
            const int bucket = (value >> shift) & (NUM_BUCKETS - 1);

            localBuffers[bucket][bufferCounts[bucket]++] = value;

            if (bufferCounts[bucket] == LOCAL_BUFFER_SIZE) {
                std::memcpy(&buffer[privateOffsets[bucket]], localBuffers[bucket], LOCAL_BUFFER_SIZE * sizeof(int));
                privateOffsets[bucket] += LOCAL_BUFFER_SIZE;
                bufferCounts[bucket] = 0;
            }
        }

        for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
            if (bufferCounts[bucket] > 0) {
                for (int j = 0; j < bufferCounts[bucket]; ++j) {
                    buffer[privateOffsets[bucket]++] = localBuffers[bucket][j];
                }
            }
        }
    }

    void sort(int *inputArray, int *outputArray, const int n, const int numThreads) {
        omp_set_num_threads(numThreads);

        int *arr = inputArray;
        int *buffer = outputArray;

        int numBits = sizeof(int) * 8;

        const auto threadLocalMax = std::make_unique<int[]>(numThreads);
        auto localHistograms = std::make_unique<int *[]>(numThreads);
        const auto globalHistogram = std::make_unique<int[]>(NUM_BUCKETS);
        const auto prefixSums = std::make_unique<int[]>(NUM_BUCKETS);
        auto threadOffsets = std::make_unique<int *[]>(numThreads);

        std::unique_ptr<int[]> histogramArrays[numThreads];
        std::unique_ptr<int[]> offsetArrays[numThreads];
        for (int t = 0; t < numThreads; ++t) {
            histogramArrays[t] = std::make_unique<int[]>(NUM_BUCKETS);
            offsetArrays[t] = std::make_unique<int[]>(NUM_BUCKETS);

            localHistograms[t] = histogramArrays[t].get();
            threadOffsets[t] = offsetArrays[t].get();
        }

        for (int shift = 0; shift < numBits; shift += BITS_PER_PASS) {
            std::memset(globalHistogram.get(), 0, NUM_BUCKETS * sizeof(int));
            std::memset(prefixSums.get(), 0, NUM_BUCKETS * sizeof(int));

            #pragma omp parallel default(none) shared(arr, buffer, localHistograms, globalHistogram, prefixSums, threadOffsets, shift, n, numThreads, threadLocalMax, numBits)
            {
                const int tid = omp_get_thread_num();
                (shift == 0)
                    ? computeLocalHistogramsWithMax(arr, n, localHistograms.get(), shift, tid, threadLocalMax.get())
                    : computeLocalHistograms(arr, n, localHistograms[tid], shift);

                #pragma omp barrier

                #pragma omp single
                {
                    if (shift == 0) {
                        numBits = computeNumBits(threadLocalMax.get(), numThreads);
                    }

                    computeGlobalHistogram(localHistograms.get(), globalHistogram.get(), numThreads);
                    computePrefixSums(globalHistogram.get(), prefixSums.get());
                    computeThreadOffsets(localHistograms.get(), prefixSums.get(), threadOffsets.get(), numThreads);
                }

                #pragma omp barrier
                scatterToBuffer(arr, n, buffer, threadOffsets[tid], shift);
            }

            std::swap(arr, buffer);
        }
    }
}
