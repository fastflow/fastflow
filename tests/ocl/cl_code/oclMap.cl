typedef struct _mypair { 
    float a; 
    float b; 
} mypair;

float fmapf(float elem) {
    return (elem+1.0);
}
__kernel void kern_mapf(__global float* input,
			__global float* output,
			const uint inSize,
			const uint maxItems,
			const uint offset,
			const uint pad) {
    int i = get_global_id(0);
    uint gridSize = get_local_size(0)*get_num_groups(0);
    while(i < maxItems)  {
	output[i] = fmapf(input[i]);
	i += gridSize;
    }
}
