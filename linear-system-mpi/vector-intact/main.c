#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

const int N = 27000;
const double E = 0.00001;
double T = 0.00001;

void separation(int *vecSize, int *vecBeginPos, int *matrixSize, int *matrixBeginPos, int size)
{
    int offset = 0;
    for (int i = 0; i < size; i++)
    {
        vecSize[i] = N / size;
    }

    for (int i = 0; i < size; i++)
    {
        if (i < N % size)
        {
            vecSize[i]++;
        }
        vecBeginPos[i] = offset;
        offset += vecSize[i];
        matrixBeginPos[i] = vecBeginPos[i] * N;
        matrixSize[i] = vecSize[i] * N;
    }
}

void loadData(double *A, double *b)
{
    for (int i = 0; i < N; i++)
    {
        b[i] = N + 1;
        for (int j = 0; j < N; j++)
        {
            if (i == j)
            {
                A[N * i + j] = 2;
            } else
            {
                A[N * i + j] = 1;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int size;
    int rank;

    double partNorm = 0;
    double sumNorm = 0;
    double normB = 0;

    double start;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *vecSize = (int *) calloc(size, sizeof(int));
    int *vecBeginPos = (int *) calloc(size, sizeof(int));
    int *matrixBeginPos = (int *) calloc(size, sizeof(int));
    int *matrixSize = (int *) calloc(size, sizeof(int));


    double *A = (double *) calloc(N * N, sizeof(double));
    double *x = (double *) calloc(N, sizeof(double));
    double *b = (double *) calloc(N, sizeof(double));


    if (rank == 0)
    {
        loadData(A, b);
    }

    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    separation(vecSize, vecBeginPos, matrixSize, matrixBeginPos, size);

    printf("I'm %d from %d processes and my lines: %d-%d (%d lines)\n", rank, size, matrixBeginPos[rank] / N,
           (matrixBeginPos[rank] + matrixSize[rank]) / N, matrixSize[rank] / N);

    double *buf = (double *) calloc(matrixSize[rank] / N, sizeof(double));
    double *buf1 = (double *) calloc(N, sizeof(double));
    double *partA = (double *) calloc(vecSize[rank] * N, sizeof(double));
    double *partX = (double *) calloc(vecSize[rank], sizeof(double));

    MPI_Scatterv(A, matrixSize, matrixBeginPos, MPI_DOUBLE, partA, matrixSize[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < N; ++i)
        {
            normB += b[i] * b[i];
        }
        normB = sqrt(normB);
        start = MPI_Wtime();
    }

    int flag = 1;

    while (flag)
    {
        int vecBeginPos_tmp = vecBeginPos[rank];

        for (int i = 0; i < vecSize[rank]; i++)
        {
            double * partA_tmp = &partA[i * N];
            double sum = 0;
            for (int j = 0; j < N; j++)
            {
                sum += partA_tmp[j] * x[j];
            }

            buf[i] = sum - b[vecBeginPos_tmp + i];
            partNorm += buf[i] * buf[i];
        }

        MPI_Reduce(&partNorm, &sumNorm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allgatherv(buf, vecSize[rank], MPI_DOUBLE, buf1, vecSize, vecBeginPos, MPI_DOUBLE, MPI_COMM_WORLD);

        for (int i = 0; i < vecSize[rank]; i++)
        {
            partX[i] = x[vecBeginPos_tmp + i] - T * buf1[vecBeginPos_tmp + i];
        }

        MPI_Allgatherv(partX, vecSize[rank], MPI_DOUBLE, x, vecSize, vecBeginPos, MPI_DOUBLE, MPI_COMM_WORLD);

        if (rank == 0)
        {
            sumNorm = sqrt(sumNorm) / normB;
            printf("I'm %d from %d processes and E = : %lf\n", rank, size, sumNorm);
            if (sumNorm < E)
            {
                flag = 0;
            }
        }

        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        partNorm = 0;
        sumNorm = 0;
    }

    if (rank == 0)
    {
        double end = MPI_Wtime();
        for (int i = 0; i < N; i++)
        {
            //printf("%lf ", x[i]);
        }
        printf("\n");
        printf("Time: %lf\n", end - start);
    }

    free(vecSize);
    free(matrixSize);
    free(vecBeginPos);
    free(matrixBeginPos);
    free(x);
    free(buf);
    free(partA);
    free(A);
    free(b);

    MPI_Finalize();
    return 0;
}

