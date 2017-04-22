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

NUMBA_EXPORT_FUNC(int64_t)
numba_h5_size(hid_t file_id, char* dset_name, int dim)
{
    hid_t dataset_id;
    dataset_id = H5Dopen2(file_id, dset_name, H5P_DEFAULT);
    assert(dataset_id != -1);
    hid_t space_id = H5Dget_space(dataset_id);
    assert(space_id != -1);
    hsize_t data_ndim = H5Sget_simple_extent_ndims(space_id);
    hsize_t space_dims[data_ndim];
    H5Sget_simple_extent_dims(space_id, space_dims, NULL);
    return space_dims[dim];
}
