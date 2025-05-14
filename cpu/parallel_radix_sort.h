#pragma once

namespace BaseParallel {
    void sort(const int *inputArray, int *outputArray, int n, int numThreads);
}

namespace ParallelOptA {
    void sort(int *inputArray, int *outputArray, int n, int numThreads);
}

namespace ParallelOptB {
    void sort(const int *inputArray, int *outputArray, int n, int numThreads);
}

namespace ParallelOptC {
    void sort(const int *inputArray, int *outputArray, int n, int numThreads);
}

namespace ParallelAllOpts {
    void sort(int *inputArray, int *outputArray, int n, int numThreads);
}

namespace ParallelOptAC {
    void sort(int *inputArray, int *outputArray, int n, int numThreads);
}
