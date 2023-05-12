#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

const double E = 10e-8;

const int Z_N = 65;
const int Y_N = 65;
const int X_N = 65;
const int A = 1;

const double area_size_x = 2.0;
const double area_size_y = 2.0;
const double area_size_z = 2.0;

double f_z;
double f_y;
double f_x;
double constant_coefficient;

double *function_values[2] = {};
double *buffer_bounds[2] = {};

int previous = 1;
int current = 0;

MPI_Request send_request[2];
MPI_Request receive_request[2];

double required_function(double x, double y, double z)
{
    return x * x + y * y + z * z;
}

double rho(double x, double y, double z)
{
    return 6 - A * required_function(x, y, z);
}

void bounds_initialization(const int *number_per_thread, const int *offset_per_thread, int rank, int Y, int X, double h_x, double h_y, double h_z)
{
    for (int i = 0, start_pos = offset_per_thread[rank]; i <= number_per_thread[rank] - 1; i++, start_pos++)
    {
        for (int j = 0; j <= Y_N; j++)
        {
            for (int k = 0; k <= X_N; k++)
            {
                if ((start_pos != 0) && (j != 0) && (k != 0) && (start_pos != Z_N) && (j != Y_N) && (k != X_N))
                {
                    function_values[0][i * Y * X + j * X + k] = 0;
                    function_values[1][i * Y * X + j * X + k] = 0;
                } else
                {
                    function_values[0][i * Y * X + j * X + k] = required_function(start_pos * h_x, j * h_y, k * h_z);
                    function_values[1][i * Y * X + j * X + k] = required_function(start_pos * h_x, j * h_y, k * h_z);
                }
            }
        }
    }
}

void boundary_calculation(int Y, int X, double h_x, double h_y, double h_z, int rank, int size, const int *number_per_thread,
                          const int *offset_per_thread, double pow_h_x, double pow_h_y, double pow_h_z)
{
    for (int j = 1; j < Y_N; j++)
    {
        for (int k = 1; k < X_N; k++)
        {

            if (rank != 0)
            {
                int i = 0;
                f_z = (function_values[previous][(i + 1) * Y * X + j * X + k] + buffer_bounds[0][j * X + k]) / pow_h_x;
                f_y = (function_values[previous][i * Y * X + (j + 1) * X + k] + function_values[previous][i * Y * X + (j - 1) * X + k]) / pow_h_y;
                f_x = (function_values[previous][i * Y * X + j * X + (k + 1)] + function_values[previous][i * Y * X + j * X + (k - 1)]) / pow_h_z;
                function_values[current][i * Y * X + j * X + k] = (f_z + f_y + f_x - rho((i + offset_per_thread[rank])
                        * h_x, j * h_y, k * h_z)) / constant_coefficient;
            }

            if (rank != size - 1)
            {
                int i = number_per_thread[rank] - 1;
                f_z = (buffer_bounds[1][j * X + k] + function_values[previous][(i - 1) * Y * X + j * X + k]) / pow_h_x;
                f_y = (function_values[previous][i * Y * X + (j + 1) * X + k] + function_values[previous][i * Y * X + (j - 1) * X + k]) / pow_h_y;
                f_x = (function_values[previous][i * Y * X + j * X + (k + 1)] + function_values[previous][i * Y * X + j * X + (k - 1)]) / pow_h_z;
                function_values[current][i * Y * X + j * X + k] = (f_z + f_y + f_x - rho((i + offset_per_thread[rank])
                        * h_x, j * h_y, k * h_z)) / constant_coefficient;
            }

        }
    }
}

void data_send(int Y, int X, int rank, int size, const int *number_per_thread)
{
    if (rank != 0)
    {

        MPI_Isend(&(function_values[previous][0]), X * Y, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_request[0]);
        MPI_Irecv(buffer_bounds[0], X * Y, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &receive_request[1]);
    }
    if (rank != size - 1)
    {
        MPI_Isend(&(function_values[previous][(number_per_thread[rank] - 1) * Y * X]), X * Y, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &send_request[1]);
        MPI_Irecv(buffer_bounds[1], X * Y, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &receive_request[0]);
    }
}

void data_receive(int rank, int size)
{
    if (rank != 0)
    {
        MPI_Wait(&send_request[0], MPI_STATUS_IGNORE);
        MPI_Wait(&receive_request[1], MPI_STATUS_IGNORE);
    }
    if (rank != size - 1)
    {
        MPI_Wait(&send_request[1], MPI_STATUS_IGNORE);
        MPI_Wait(&receive_request[0], MPI_STATUS_IGNORE);
    }
}

void inner_part_calculation(int Y, int X, double h_x, double h_y, double h_z, int rank, const int *number_per_thread,
                            const int *offset_per_thread, double pow_h_x, double pow_h_y, double pow_h_z)
{
    for (int i = 1; i < number_per_thread[rank] - 1; i++)
    {
        for (int j = 1; j < Y_N; j++)
        {
            for (int k = 1; k < X_N; k++)
            {
                f_z = (function_values[previous][(i + 1) * Y * X + j * X + k] + function_values[previous][(i - 1) * Y * X + j * X + k]) / pow_h_x;
                f_y = (function_values[previous][i * Y * X + (j + 1) * X + k] + function_values[previous][i * Y * X + (j - 1) * X + k]) / pow_h_y;
                f_x = (function_values[previous][i * Y * X + j * X + (k + 1)] + function_values[previous][i * Y * X + j * X + (k - 1)]) / pow_h_z;
                function_values[current][i * Y * X + j * X + k] = (f_z + f_y + f_x - rho((i + offset_per_thread[rank])
                        * h_x, j * h_y, k * h_z)) / constant_coefficient;
            }
        }
    }
}

void deviation_calculation(int Y, int X, double h_x, double h_y, double h_z, int rank, const int *number_per_thread, const int *offset_per_thread)
{
    double max = 0;
    double max_tmp = 0;
    double F_tmp = 0;

    for (int i = 1; i < number_per_thread[rank] - 1; i++)
    {
        for (int j = 1; j < Y_N; j++)
        {
            for (int k = 1; k < X_N; k++)
            {
                if ((F_tmp = fabs(function_values[current][i * Y * X + j * X + k] -
                        required_function((i + offset_per_thread[rank]) * h_x, j * h_y, k * h_z))) > max)
                {
                    max = F_tmp;
                }
            }
        }
    }

    MPI_Allreduce(&max, &max_tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0)
    {
        max = max_tmp;
        printf("max = %lf\n", max);
    }
}

void termination_condition(int *flag, int Y, int X, double h_x, double h_y, double h_z, int rank, const int *number_per_thread, const int *offset_per_thread)
{
    double max = 0;
    double max_tmp = 0;
    double F_tmp = 0;

    for (int i = 1; i < number_per_thread[rank] - 1; i++)
    {
        for (int j = 1; j < Y_N; j++)
        {
            for (int k = 1; k < X_N; k++)
            {
                if ((F_tmp = fabs(function_values[current][i * Y * X + j * X + k] - function_values[previous][i * Y * X + j * X + k])) > max)
                {
                    max = F_tmp;
                }
            }
        }
    }

    MPI_Allreduce(&max, &max_tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (max_tmp < E)
    {
        (*flag) = 0;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    int flag = 1;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *number_per_thread = calloc(size, sizeof(int));
    int *offset_per_thread = calloc(size, sizeof(int));

    for (int i = 0, height = X_N + 1, tmp = size - (height % size), currentLine = 0; i < size; i++)
    {
        offset_per_thread[i] = currentLine;
        number_per_thread[i] = height / size;
        if (i >= tmp)
        {
            number_per_thread[i]++;
        }
        currentLine += number_per_thread[i];
    }

    int Z = number_per_thread[rank];
    int Y = (Y_N + 1);
    int X = (X_N + 1);

    function_values[0] = calloc(Z * Y * X, sizeof(double));
    function_values[1] = calloc(Z * Y * X, sizeof(double));

    buffer_bounds[0] = calloc(Y * X, sizeof(double));
    buffer_bounds[1] = calloc(Y * X, sizeof(double));

    double h_x = area_size_x / Z_N;
    double h_y = area_size_y / Y_N;
    double h_z = area_size_z / X_N;

    double pow_h_x = pow(h_x, 2);
    double pow_h_y = pow(h_y, 2);
    double pow_h_z = pow(h_z, 2);
    constant_coefficient = 2 / pow_h_x + 2 / pow_h_y + 2 / pow_h_z + A;

    bounds_initialization(number_per_thread, offset_per_thread, rank, Y, X, h_x, h_y, h_z);

    double start;
    if (rank == 0)
    {
        start = MPI_Wtime();
    }

    while (flag != 0)
    {
        previous = 1 - previous;
        current = 1 - current;

        data_send(Y, X, rank, size, number_per_thread);
        inner_part_calculation(Y, X, h_x, h_y, h_z, rank, number_per_thread, offset_per_thread, pow_h_x, pow_h_y, pow_h_z);
        data_receive(rank, size);

        boundary_calculation(Y, X, h_x, h_y, h_z, rank, size, number_per_thread, offset_per_thread, pow_h_x, pow_h_y, pow_h_z);

        termination_condition(&flag, Y, X, h_x, h_y, h_z, rank, number_per_thread, offset_per_thread);
    }

    if (rank == 0)
    {
        double finish = MPI_Wtime();
        printf("Time: %lf\n", finish - start);
    }

    deviation_calculation(Y, X, h_x, h_y, h_z, rank, number_per_thread, offset_per_thread);

    MPI_Finalize();

    free(buffer_bounds[0]);
    free(buffer_bounds[1]);
    free(function_values[0]);
    free(function_values[1]);
    free(offset_per_thread);
    free(number_per_thread);

    return 0;
}
