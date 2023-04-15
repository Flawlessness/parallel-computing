#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int DIMENSIONS = 2;
const int X = 0;
const int Y = 1;

void combining_matrix(double *partC, double *C, MPI_Comm grid_comm, const int dims[DIMENSIONS], int coords[DIMENSIONS], int N1, int N3, int size)
{
    int *received_number = (int*)malloc(sizeof(int) * size);
    int *displs = (int*)malloc(sizeof(int) * size);

    MPI_Datatype received_vec;
    MPI_Datatype resized_received_vec;

    MPI_Type_vector(N1 / dims[Y], N3 / dims[X], N3, MPI_DOUBLE, &received_vec);
    MPI_Type_commit(&received_vec);

    MPI_Type_create_resized(received_vec, 0, N3 / dims[X] * sizeof(double), &resized_received_vec);
    MPI_Type_commit(&resized_received_vec);

    for (int i = 0; i < size; i++)
    {
        received_number[i] = 1;
        MPI_Cart_coords(grid_comm, i, DIMENSIONS, coords);
        displs[i] = dims[X] * (N1 / dims[Y]) * coords[Y] + coords[X];
    }

    MPI_Gatherv(partC, N1 * N3 / (dims[X] * dims[Y]), MPI_DOUBLE, C, received_number, displs, resized_received_vec, 0, grid_comm);

    MPI_Type_free(&received_vec);
    MPI_Type_free(&resized_received_vec);
}

void create_comm(MPI_Comm *grid_comm, MPI_Comm *rows_comm, MPI_Comm *col_comm, int *coords, int *dims)
{
    int periods[DIMENSIONS];
    MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONS, dims, periods, 0, grid_comm);
    int rank;
    MPI_Comm_rank(*grid_comm, &rank);
    MPI_Cart_coords(*grid_comm, rank, DIMENSIONS, coords);

    MPI_Comm_split(*grid_comm, coords[Y], coords[X], rows_comm);
    MPI_Comm_split(*grid_comm, coords[X], coords[Y], col_comm);
}

void separation_matrix(const double *A, const double *B, double *partA, double *partB, const MPI_Comm row_comm, const MPI_Comm col_comm, const int coords[DIMENSIONS], const int dims[DIMENSIONS], int N1, int N2, int N3)
{
    if (coords[X] == 0)
    {
        MPI_Scatter(A, N1 * N2 / dims[Y], MPI_DOUBLE, partA, N1 * N2 / dims[Y], MPI_DOUBLE, 0, col_comm);
    }

    if (coords[Y] == 0)
    {
        MPI_Datatype send_time_vec;
        MPI_Datatype resized_send_time_vec;

        MPI_Type_vector(N2, 1, N3, MPI_DOUBLE, &send_time_vec);
        MPI_Type_commit(&send_time_vec);

        MPI_Type_create_resized(send_time_vec, 0, 1 * sizeof(double), &resized_send_time_vec);
        MPI_Type_commit(&resized_send_time_vec);

        MPI_Scatter(B, N3 / dims[X], resized_send_time_vec, partB, N2 * N3 / dims[X], MPI_DOUBLE, 0, row_comm);

        MPI_Type_free(&resized_send_time_vec);
        MPI_Type_free(&send_time_vec);
    }


    MPI_Bcast(partA, N1 * N2 / dims[Y], MPI_DOUBLE, 0, row_comm);
    MPI_Bcast(partB, N2 * N3 / dims[X], MPI_DOUBLE, 0, col_comm);
}

void initialize_matrix(double *matrix, int rows, int cols, double value)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (i == j)
            {
                matrix[i * cols + j] = 5;
            }
            else
            {
                matrix[i * cols + j] = value;
            }
        }
    }
}

void multiplication_matrix(const double *partA, const double *partB, double *partC, int partA_rows_count, int partA_col_count, int partB_col_count)
{
    for(int i = 0; i < partA_rows_count; i++)
    {
        double* tmp_partC = partC + i * partB_col_count;
        const double* tmp_partA = partA + i * partA_col_count;
        for(int j = 0; j < partB_col_count; j++)
        {
            const double* tmp_partB = partB + j * partA_col_count;
            double* c = tmp_partC + j;
            for(int z = 0; z < partA_col_count; z++)
            {
                *c += tmp_partA[z] * tmp_partB[z];
            }
        }
    }
}

void print_matrix(const double *matrix, int N1, int N2)
{
    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N2; j++)
        {
            printf("%lf ", matrix[i * N2 + j]);
        }
        printf("\n");
    }
}

void create_matrix(double **A, double **B, double **C, int N1, int N2, int N3, char **argv, int argc)
{
    *A = (double*)malloc(sizeof(double) * N1 * N2);
    *B = (double*)malloc(sizeof(double) * N2 * N3);
    *C = (double*)malloc(sizeof(double) * N1 * N3);

    initialize_matrix(*A, N1, N2, 3);
    initialize_matrix(*B, N2, N3, 3);

    printf("A:\n");
    //print_matrix(*A, N1, N2);

    printf("B:\n");
    //print_matrix(*B, N2, N3);

    printf("Result:\n");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int dims[2] = {0, 0};
    int coords[DIMENSIONS];
    int size;
    int rank;
    MPI_Comm grid_comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    double start_time;
    int exit_flag = 0;

    int N1 = (int)strtol(argv[1], NULL, 10);
    int N2 = (int)strtol(argv[2], NULL, 10);
    int N3 = (int)strtol(argv[3], NULL, 10);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, DIMENSIONS, dims);
    create_comm(&grid_comm, &row_comm, &col_comm, coords, dims);
    MPI_Comm_rank(grid_comm, &rank);

    if (rank == 0)
    {
        if ((N1 == 0) || (N2 == 0) || (N3 == 0))
        {
            printf("Wrong format of matrix sizes in arguments:\nArguments: N1 N2 N3, matrix A = N1*N2 and B = N2*N3 and C = N1*N3\n");
            exit_flag = 1;
        }
        if ((N1 % dims[Y] != 0) || (N3 % dims[X] != 0))
        {
            printf("Incorrect size of matrices in arguments:\nN1 must be a multiple of dims[1] = %d \nN3 must be a multiple of dims[0] = %d \n", dims[Y], dims[X]);
            exit_flag = 1;
        }
    }
    MPI_Bcast(&exit_flag, 1, MPI_INT, 0, grid_comm);
    if (exit_flag == 1)
    {
        MPI_Finalize();
        return 0;
    }

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    double *partA = (double*)malloc(sizeof(double) * N1 * N2 / dims[1]);
    double *partB = (double*)malloc(sizeof(double) * N2 * N3 / dims[0]);
    double *partC = (double*)calloc(N1 * N3 / (dims[X] * dims[Y]), sizeof(double));

    if (rank == 0)
    {
        create_matrix(&A, &B, &C, N1, N2, N3, argv, argc);
        start_time = MPI_Wtime();
    }

    separation_matrix(A, B, partA, partB, row_comm, col_comm, coords, dims, N1, N2, N3);
    multiplication_matrix(partA, partB, partC, N1/dims[Y], N2, N3/dims[X]);
    combining_matrix(partC, C, grid_comm, dims, coords, N1, N3, size);

    if (rank == 0)
    {
        double end_time = MPI_Wtime();
        //print_matrix(C, N1, N3);
        printf("Time: %lf\n", end_time - start_time);
    }


    free(A);
    free(B);
    free(C);
    free(partA);
    free(partB);
    free(partC);

    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);

    MPI_Finalize();

    return 0;
}

