#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

const int N = 40000;
const float T = 0.0000235;
const float E = 0.00001;

void multiplication_matrix_vector(const float *A, const float *x, float *result)
{
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        float *result_tmp = &result[i];
        const float *A_tmp = A + N*i;
        (*result_tmp) = 0;
        for (int j = 0; j < N; j++)
        {
            (*result_tmp) += A_tmp[j] * x[j];
        }
    }
}

void subtraction_vector_vector(float *x, const float *y)
{
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        x[i] -= y[i];
    }
}

void multiplication_scalar_vector(float *x, const float *a)
{
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        x[i] *= (*a);
    }
}

float norm_vector(const float *x)
{
    float norm = 0;
#pragma omp parallel for reduction(+: norm)
    for (int i = 0; i < N; i++)
    {
        norm += x[i]*x[i];
    }
    return sqrt(norm);
}

int main()
{


    float *A = (float*)calloc(N*N, sizeof(float));
    float *x = (float*)calloc(N, sizeof(float));
    float *b = (float*)calloc(N, sizeof(float));
    float *buff = (float*)calloc(N, sizeof(float));
    struct timespec start, end;
    int flag = 0;

    for (int i = 0; i < N; i++)
    {
        x[i] = 0;
        b[i] = N + 1;
        for (int j = 0; j < N; j++)
        {
            if (i == j)
                A[N*i + j] = 2;
            else
                A[N*i + j] = 1;
        }
    }
    for (int z = 100; z <= 100; z++)
    {
        flag = 0;
        omp_set_num_threads(z);
        for (int i = 0; i < N; i++)
        {
            x[i] = 0;
            b[i] = N + 1;
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        for (int i = 0; i < 100000 && flag == 0; i ++)
        {
            multiplication_matrix_vector(A, x, buff);
            subtraction_vector_vector(buff, b);
            double temp = norm_vector(buff)/norm_vector(b);
            //printf("Norm =: %lf\n", temp);
            if (temp < E)
            {
                flag = 1;
            }
            multiplication_scalar_vector(buff, &T);
            subtraction_vector_vector(x, buff);
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        printf("Time taken on %d threads: %f seconds\n", z, end.tv_sec - start.tv_sec + 0.000000001 * (end.tv_nsec - start.tv_nsec));
    }
    /*for (int i = 0; i < N; i ++)
    {
        printf("%lf ", x[i]);
    }*/
    printf("\n");
    return 0;
}


