#include "mpi.h"

NUMBA_EXPORT_FUNC(int)
numba_dist_get_rank()
{
    MPI_Init(NULL,NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // printf("my_rank:%d\n", rank);
    return rank;
}

NUMBA_EXPORT_FUNC(int)
numba_dist_get_size()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("mpi_size:%d\n", size);
    return size;
}
