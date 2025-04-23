#ifndef PARALLEL_RADIX_SORT_H
#define PARALLEL_RADIX_SORT_H


class ParallelRadixSort {
public:
    static void sort(int *arr, int n, int numThreads);

private:
    static void computeLocalHistograms(const int *arr, int n, int **localHistograms, int shift);

    static void computeGlobalHistogram(int **localHistograms, int *globalHistogram, int numThreads);

    static void computePrefixSums(const int *globalHistogram, int *prefixSums);

    static void computeThreadOffsets(int **localHistograms, const int *prefixSums, int **threadOffsets, int numThreads);

    static void scatterToBuffer(const int *arr, int n, int *buffer, int **threadOffsets, int shift);
};


#endif //PARALLEL_RADIX_SORT_H
