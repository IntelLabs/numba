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

NUMBA_EXPORT_FUNC(int64_t)
numba_dist_get_end(int64_t total, int64_t div_chunk, int num_pes, int node_id)
{
    return ((node_id==num_pes-1) ? total : (node_id+1)*div_chunk);
}
