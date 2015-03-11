__kernel void pipe_producer(__global          float *src,
                            __write_only pipe float out_pipe)
{
    int gid = get_global_id(0);

    reserve_id_t res_id;
    res_id = reserve_write_pipe(out_pipe, 1);

    if(is_valid_reserve_id(res_id))
    {
        if(write_pipe(out_pipe, res_id, 0, &src[gid]) != 0)
            return;
        commit_write_pipe(out_pipe, res_id);
    }
}

__kernel void pipe_consumer(__global         float *dst,
                            __read_only pipe float in_pipe)
{
    int gid = get_global_id(0);

    reserve_id_t res_id;
    res_id = reserve_read_pipe(in_pipe, 1);

    if(is_valid_reserve_id(res_id))
    {
        if(read_pipe(in_pipe, res_id, 0, &dst[gid]) != 0)
            return;
        commit_read_pipe(in_pipe, res_id);
    }
}