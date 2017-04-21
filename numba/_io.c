

NUMBA_EXPORT_FUNC(int)
numba_h5_open(char* file_name, char* mode)
{
    printf("%s %s\n", file_name, mode);
    return 0;
}
