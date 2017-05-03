#include "mpi.h"

NUMBA_EXPORT_FUNC(int)
numba_get_rank()
{
    MPI_Init(NULL,NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // printf("my_rank:%d\n", rank);
    return rank;
}
