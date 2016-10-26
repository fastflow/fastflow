#define GET_IN(i,j) (in[((i)*w+(j))-offset])
#define GET_ENV1(i,j) (env1[((i)*w+(j))])
unsigned char fmapf(
	__global unsigned char* in,
	const uint h,
	const uint w,
	const int r,
	const int c,
	const int offset) {
  unsigned char alive_in = GET_IN(r,c); 
  unsigned char naliven = 0; 
  naliven += r>0 && c>0 ? GET_IN(r-1,c-1) : 0; 
  naliven += r>0 ? GET_IN(r-1,c) : 0; 
  naliven += r>0 && c < w-1 ? GET_IN(r-1,c+1) : 0; 
  naliven += c < w-1 ? GET_IN(r,c+1) : 0; 
  naliven += r < w-1 && c < w-1 ? GET_IN(r+1,c+1) : 0; 
  naliven += r < w-1 ? GET_IN(r+1,c) : 0; 
  naliven += r < w-1 && c>0 ? GET_IN(r+1,c-1) : 0; 
  naliven += c>0 ? GET_IN(r,c-1) : 0; 
  return (naliven == 3 || (alive_in && naliven == 2));
}
	
__kernel void kern_mapf(
	__global unsigned char* input,
	__global unsigned char* output,
	const uint inHeight,
	const uint inWidth,
	const uint maxItems,
	const uint offset,
	const uint halo) {

  size_t i = get_global_id(0);
  size_t ig = i + offset;
  size_t r = ig / inWidth;
  size_t c = ig % inWidth;
  size_t gridSize = get_local_size(0)*get_num_groups(0);
  while(i < maxItems)  {
    output[i+halo] = fmapf(input+halo,inHeight,inWidth,r,c,offset);
    i += gridSize;
  }
}

unsigned char freducef(unsigned char x,
		       unsigned char y) {
  return x||y;
}

__kernel void kern_reducef(__global unsigned char* input, const uint halo, __global unsigned char* output, const uint n, __local unsigned char* sdata, unsigned char idElem) {
  uint blockSize = get_local_size(0);
  uint tid = get_local_id(0);
  uint i = get_group_id(0)*blockSize + get_local_id(0);
  uint gridSize = blockSize*get_num_groups(0);
  unsigned char result = idElem; input += halo;
  if(i < n) { result = input[i]; i += gridSize; }
  while(i < n) {
    result = freducef(result, input[i]);
    i += gridSize;
  }
  sdata[tid] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
  if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
  if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
  if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
  if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
  if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
  if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
  if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
  if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }
  if(tid == 0) output[get_group_id(0)] = sdata[tid];
}
