#include "sum.h"

void omp_sum(double *sum_ret)
{
    double sum = 0;
    #pragma omp parallel for
        for(int i=0; i < size; i++ ) {
            sum += x[i];
        }
    *sum_ret = sum;
}

void omp_critical_sum(double *sum_ret)
{
    double sum = 0;
    #pragma omp parallel for
        for(int i=0; i < size; i++ ) {
            #pragma omp critical
            sum += x[i];
        }
    *sum_ret = sum;
}

void omp_atomic_sum(double *sum_ret)
{
    double sum = 0;
    #pragma omp parallel for
        for(int i=0; i < size; i++ ) {
            #pragma omp atomic
            sum += x[i];
        }
    *sum_ret = sum;
}

void omp_local_sum(double *sum_ret)
{
    const int MAX_THREADS = 32;
    double sum[MAX_THREADS];
    for(int i = 0; i < MAX_THREADS; i++) {
        sum[i] = 0;
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for(int i=0; i < size; i++ ) {
            sum[tid] += x[i];
        }
    }
    double resSum = 0;
    for(int i=0; i < MAX_THREADS; i++ ) {
        resSum += sum[i];
    }
    *sum_ret = resSum;
}

void omp_padded_sum(double *sum_ret)
{
    double sum[320000];
    for(int i = 0; i < 320000; i+=10000) {
        sum[i] = 0;
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num() * 10000;
        #pragma omp for
        for(int i=0; i < size; i++ ) {
            sum[tid] += x[i];
        }
    }
    double resSum = 0;
    for(int i=0; i < 320000; i+=10000 ) {
        resSum += sum[i];
    }
    *sum_ret = resSum;
}

void omp_private_sum(double *sum_ret)
{
    double globalsum = 0;
    #pragma omp parallel
    {
        double sum = 0;
        #pragma omp for
        for(int i=0; i < size; i++ ) {
            sum += x[i];
        }
    #pragma omp critical
    globalsum += sum;
    }
    *sum_ret = globalsum;
}

void omp_reduction_sum(double *sum_ret)
{
    double sum = 0;
    #pragma omp parallel for reduction (+:sum)
        for(int i=0; i < size; i++ ) {
            sum += x[i];
        }
    *sum_ret = sum;
}
