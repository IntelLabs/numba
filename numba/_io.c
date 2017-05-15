#include "hdf5.h"

NUMBA_EXPORT_FUNC(int)
numba_h5_open(char* file_name, char* mode)
{
    // printf("h5_open file_name: %s mode:%s\n", file_name, mode);
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    assert(plist_id != -1);
    herr_t ret;
    hid_t file_id;
    ret = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    assert(ret != -1);
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
numba_h5_read(hid_t file_id, char* dset_name, int ndims, int64_t* sizes,
    void* out, int typ_enum, int64_t start_ind, int64_t end_ind)
{
    //printf("dset_name:%s ndims:%d size:%d typ:%d\n", dset_name, ndims, sizes[0], typ_enum);
    // fflush(stdout);
    // printf("start %lld end %lld\n", start_ind, end_ind);
    int i;
    hid_t dataset_id;
    herr_t ret;
    dataset_id = H5Dopen2(file_id, dset_name, H5P_DEFAULT);
    assert(dataset_id != -1);
    hid_t space_id = H5Dget_space(dataset_id);
    assert(space_id != -1);

    hsize_t HDF5_start[ndims];
    for(i=0; i<ndims; i++)
        HDF5_start[i] = 0;
    hsize_t* HDF5_count = (hsize_t*)sizes;

    hid_t xfer_plist_id = H5P_DEFAULT;
    if(1)// start_ind!=-1)
    {
        HDF5_start[0] = start_ind;
        HDF5_count[0] = end_ind;
        xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
    }

    ret = H5Sselect_hyperslab(space_id, H5S_SELECT_SET, HDF5_start, NULL, HDF5_count, NULL);
    assert(ret != -1);
    hid_t mem_dataspace = H5Screate_simple((hsize_t)ndims, HDF5_count, NULL);
    assert (mem_dataspace != -1);
    hid_t h5_typ = get_h5_typ(typ_enum);
    ret = H5Dread(dataset_id, h5_typ, mem_dataspace, space_id, xfer_plist_id, out);
    assert(ret != -1);
    // printf("out: %lf %lf ...\n", ((double*)out)[0], ((double*)out)[1]);
    H5Dclose(dataset_id);
    return ret;
}


// _h5_typ_table = {
//     int8:0,
//     uint8:1,
//     int32:2,
//     int64:3,
//     float32:4,
//     float64:5
//     }

hid_t get_h5_typ(typ_enum)
{
    // printf("h5 type enum:%d\n", typ_enum);
    hid_t types_list[] = {H5T_NATIVE_CHAR, H5T_NATIVE_UCHAR,
            H5T_NATIVE_INT, H5T_NATIVE_LLONG, H5T_NATIVE_FLOAT, H5T_NATIVE_DOUBLE};
    return types_list[typ_enum];
}
