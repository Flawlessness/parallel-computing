#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

const int EXIT_CODE = -1;
const int TASK_CODE = 1;
const int TASK_COUNT_CODE = 2;
const int NO_TASK_CODE = -2;

const int DELTA = 100000;
const int LIST_SIZE = 10;
const int TASK_SIZE = 1000;
const int MIN_TASKS_TO_SHARE = 5;

pthread_t threads[2];
pthread_mutex_t mutex;
int *tasks;

double max_diff = 0;
int exit_flag = 0;

int size = 0;
int rank = 0;
int tasks_remaining = 0;
int total_executed_tasks = 0;
int current_executed_tasks = 0;
int current_additional_tasks = 0;
int total_additional_tasks = 0;
double global_res = 0;

void initializeTaskSet(int *task_array, int task_count, int iter_counter)
{
    for (int i = 0; i < task_count; i++)
    {
        task_array[i] = DELTA + abs(rank - (iter_counter % size)) * DELTA;
    }
}

void execute_task_set(const int *task_array)
{
    for (int i = 0; i < tasks_remaining; i++)
    {
        pthread_mutex_lock(&mutex);
        int weight = task_array[i];
        pthread_mutex_unlock(&mutex);
        for (int j = 0; j < weight; j++)
        {
            global_res += 1;
        }
        current_executed_tasks++;
    }
    tasks_remaining = 0;
}

void *executor_start_routine(void *args)
{
    tasks = (int *) calloc(TASK_SIZE, sizeof(int));
    double start;
    double end;
    double iteration_duration;
    double shortest_iteration;
    double longest_iteration;

    for (int i = 0; i < LIST_SIZE; i++)
    {
        start = MPI_Wtime();
        printf("Global iteration: %d\n", i);
        initializeTaskSet(tasks, TASK_SIZE, i);
        total_executed_tasks = 0;
        current_executed_tasks = 0;
        total_additional_tasks = 0;
        tasks_remaining = TASK_SIZE;
        current_additional_tasks = 0;

        execute_task_set(tasks);
        total_executed_tasks += current_executed_tasks;
        printf("The main calculations (%d)*%d are performed by process: %d in %lf\n", tasks[0], current_executed_tasks, rank, MPI_Wtime() - start);
        int thread_response;
        int flag = 1;
        while (flag)
        {
            flag = 0;
            for (int proc_rank = 0; proc_rank < size; proc_rank++)
            {

                if (proc_rank != rank)
                {
                    MPI_Send(&rank, 1, MPI_INT, proc_rank, 10, MPI_COMM_WORLD);

                    MPI_Recv(&thread_response, 1, MPI_INT, proc_rank, TASK_COUNT_CODE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (thread_response != NO_TASK_CODE)
                    {
                        flag = 1;
                        printf("Process %d received %d tasks from %d\n", rank, thread_response, proc_rank);
                        current_additional_tasks = thread_response;
                        total_additional_tasks += current_additional_tasks;
                        memset(tasks, 0, TASK_SIZE);

                        MPI_Recv(tasks, current_additional_tasks, MPI_INT, proc_rank, TASK_CODE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        pthread_mutex_lock(&mutex);
                        tasks_remaining = current_additional_tasks;
                        current_executed_tasks = 0;
                        pthread_mutex_unlock(&mutex);
                        execute_task_set(tasks);
                        total_executed_tasks += current_executed_tasks;
                    }
                }
            }
        }
        end = MPI_Wtime();
        iteration_duration = end - start;

        MPI_Allreduce(&iteration_duration, &longest_iteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iteration_duration, &shortest_iteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        printf("Process %d has calculated %d main and %d additional tasks in %lf, res: %lf\n", rank,  total_executed_tasks, total_additional_tasks, iteration_duration, global_res);
    }
    max_diff = longest_iteration - shortest_iteration;
    pthread_mutex_lock(&mutex);
    exit_flag = 1;
    pthread_mutex_unlock(&mutex);
    int signal = EXIT_CODE;
    MPI_Send(&signal, 1, MPI_INT, rank, 10, MPI_COMM_WORLD);
    free(tasks);
    pthread_exit(NULL);
}

void *receiver_start_routine(void *args)
{
    int asking_proc_rank;
    int tasks_to_share;
    int pending_message;
    MPI_Status status;
    while (!exit_flag)
    {
        MPI_Recv(&pending_message, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status);

        if (pending_message == EXIT_CODE)
        {
            printf("The executor thread has completed the work in process: %d\n", rank);
            break;
        }

        asking_proc_rank = pending_message;
        pthread_mutex_lock(&mutex);
        tasks_to_share = (tasks_remaining - current_executed_tasks) / (size);
        if (tasks_to_share >= MIN_TASKS_TO_SHARE)
        {
            tasks_remaining = tasks_remaining - tasks_to_share;

            MPI_Send(&tasks_to_share, 1, MPI_INT, asking_proc_rank, TASK_COUNT_CODE, MPI_COMM_WORLD);
            MPI_Send(&tasks[tasks_remaining - tasks_to_share], tasks_to_share, MPI_INT, asking_proc_rank, TASK_CODE, MPI_COMM_WORLD);
        }
        else
        {
            tasks_to_share = NO_TASK_CODE;
            MPI_Send(&tasks_to_share, 1, MPI_INT, asking_proc_rank, TASK_COUNT_CODE, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&mutex);
    }

    pthread_exit(NULL);
}


int main(int argc, char *argv[])
{
    int thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_support);
    if (thread_support != MPI_THREAD_MULTIPLE)
    {
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    pthread_mutex_init(&mutex, NULL);
    pthread_attr_t ThreadAttributes;

    double start = MPI_Wtime();
    pthread_attr_init(&ThreadAttributes);
    pthread_attr_setdetachstate(&ThreadAttributes, PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0], &ThreadAttributes, receiver_start_routine, NULL);
    pthread_create(&threads[1], &ThreadAttributes, executor_start_routine, NULL);
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
    pthread_attr_destroy(&ThreadAttributes);
    pthread_mutex_destroy(&mutex);

    double res = 0;

    MPI_Reduce(&global_res, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Time: %lf\n", MPI_Wtime() - start);
        printf("Maximum difference: %lf\n", max_diff);
        printf("Result: %lf\n", res);
    }

    MPI_Finalize();
    return 0;
}
