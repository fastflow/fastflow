float fmap1f(
	     __global float* in,
	     const uint useless,
	     const int i,
	     const int i_,
	     __global const	 int* k_) {
    (void)useless ; 
    const int k = *k_; 
    return (float)((k+1) + i_);
}	   

__kernel void kern_map1f(
			 __global float* input,
			 __global float* output,
			 const uint inSize,
			 const uint maxItems,
			 const uint offset,
			 const uint pad,
			 __global const int* env1) {
    int i = get_global_id(0);
    int ig = i + offset;
    uint gridSize = get_local_size(0)*get_num_groups(0);
    while(i < maxItems)  {
	output[i+pad] = fmap1f(input+pad,inSize,ig,i,env1);
	i += gridSize;
    }
}
