#ifndef SERIAL_RADIX_SORT_H
#define SERIAL_RADIX_SORT_H


class SerialRadixSort {
public:
    static void sort(int *arr, int n);

private:
    static void buildHistogram(const int *arr, int n, int *histogram, int shift);

    static void computePrefixSums(const int *histogram, int *prefixSums);

    static void scatterToBuffer(const int *arr, int n, int *buffer, int *prefixSums, int shift);
};


#endif //SERIAL_RADIX_SORT_H
