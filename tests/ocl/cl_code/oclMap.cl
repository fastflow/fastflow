float fmapf1(float elem, const int i) {
    return i;
}

float fmapf2(float elem, const int useless) {
    (void)useless;
    return (elem+1.0);
}

__kernel void kern_mapf1(__global float* input,
			 __global float* output,
			 const uint inSize,
			 const uint maxItems,
			 const uint offset,
			 const uint pad) {
  int i = get_global_id(0);
  uint gridSize = get_local_size(0)*get_num_groups(0);
  while(i < maxItems)  {
    output[i] = fmapf1(input[i], i);
    i += gridSize;
  }
}

__kernel void kern_mapf2(__global float* input,
			 __global float* output,
			 const uint inSize,
			 const uint maxItems,
			 const uint offset,
			 const uint pad) {
  int i = get_global_id(0);
  uint gridSize = get_local_size(0)*get_num_groups(0);
  while(i < maxItems)  {
    output[i] = fmapf2(input[i], i);
    i += gridSize;
  }
}
