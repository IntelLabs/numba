#include "hdf5.h"

NUMBA_EXPORT_FUNC(int)
numba_h5_open(char* file_name, char* mode)
{
    // printf("h5_open file_name: %s mode:%s\n", file_name, mode);
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    assert(plist_id != -1);
    herr_t ret;
    hid_t file_id;
    // ret = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    // assert(ret != -1);
    file_id = H5Fopen((const char*)file_name, H5F_ACC_RDONLY, plist_id);
    assert(file_id != -1);
    ret = H5Pclose(plist_id);
    assert(ret != -1);
    return file_id;
}
