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
    H5Dclose(dataset_id);
    return space_dims[dim];
}

NUMBA_EXPORT_FUNC(int)
numba_h5_read(hid_t file_id, char* dset_name, int ndims, int64_t* sizes, void* out)
{
    // printf("ndims: %d sizes:%lld\n", ndims, sizes);
    // fflush(stdout);
    // printf("size:%d\n", sizes[0]);
    // fflush(stdout);
    hid_t dataset_id;
    herr_t ret;
    dataset_id = H5Dopen2(file_id, dset_name, H5P_DEFAULT);
    assert(dataset_id != -1);
    hid_t space_id = H5Dget_space(dataset_id);
    assert(space_id != -1);

    hsize_t HDF5_start[ndims];
    for(int i=0; i<ndims; i++)
        HDF5_start[i] = 0;
    hsize_t* HDF5_count = sizes;
    ret = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, HDF5_start, NULL, HDF5_count, NULL);
    assert(ret != -1);
    hid_t mem_dataspace = H5Screate_simple(ndims, HDF5_count, NULL);
    assert (mem_dataspace != -1);
    ret = H5Dread(dataset_id, H5T_NATIVE_FLOAT, mem_dataspace, space_id, H5P_DEFAULT, out);
    assert(ret != -1);
    H5Dclose(dataset_id);
    return ret;
}
