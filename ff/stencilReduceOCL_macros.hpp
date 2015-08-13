/*
 * stencilReduceOCL_macros.hpp
 *
 *  Created on: Feb 13, 2015
 *      Author: drocco
 */

#ifndef STENCILREDUCEOCL_MACROS_HPP_
#define STENCILREDUCEOCL_MACROS_HPP_

#include <string>
#include <sstream>

namespace ff {


/*
 * An instance of the elemental function is executed
 * for each value in [0, local_size-1] where local_size
 * is the device-local size of 'input' and 'env1' arrays.
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'inT' is the element type of the input
 * 'size' is the global size of the input array
 * 'idx' is the global index
 * 'in' is the device-local input array
 * 'idx_' is the device-local index
 * 'env1T' is the element type of the constant environment array
 * 'env1' is the constant environment array
 * 'code' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC1(name,inT,size,idx,in,idx_,env1T,env1, code)         \
    static char name[] =						             \
        "kern_" #name "|"	 					             \
        #inT "|"                                                                     \
"\n\n" #inT " f" #name "(\n"                                                   	     \
"\t__global " #inT "* " #in ",\n"						     \
"\tconst uint " #size ",\n"                        				     \
"\tconst int " #idx ",\n"                                                            \
"\tconst int " #idx_ ",\n"                                                           \
"\t__global const " #env1T "* " #env1 ") {\n"                                        \
"\t   " #code ";\n"                                                                  \
"}\n\n"                                                                              \
"__kernel void kern_" #name "(\n"                                                    \
"\t__global " #inT  "* input,\n"                                                     \
"\t__global " #inT "* output,\n"                                                     \
"\tconst uint inSize,\n"                                                             \
"\tconst uint maxItems,\n"                                                  	     \
"\tconst uint offset,\n"                                                  	     \
"\tconst uint pad,\n"                                                                \
"\t__global const " #env1T "* env1) {\n"                                             \
"\t    int i = get_global_id(0);\n"                                                  \
"\t    int ig = i + offset;\n"                                              	     \
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"                       \
"\t    while(i < maxItems)  {\n"                                                     \
"\t        output[i+pad] = f" #name "(input+pad,inSize,ig,i,env1);\n"                \
"\t        i += gridSize;\n"                                                         \
"\t    }\n"                                                                          \
"}\n"


/*
 * An instance of the elemental function is executed
 * for each value in [0, local_size-1] where local_size
 * is the device-local size of 'input' and 'env1' arrays.
 *
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'inT' is the element type of the input
 * 'size' is the global size of the input array
 * 'idx' is the global index
 * 'in' is the device-local input array
 * 'idx_' is the device-local index
 * 'env1T' is the element type of the constant environment array
 * 'env1' is the constant environment array
 * 'env2T' is the element type of the constant environment value
 * 'env2' is (a pointer to) the constant environment value
 * 'code' is the OpenCL code of the elemental function
 */
#define FF_OCL_STENCIL_ELEMFUNC2(name,inT,size,idx,in,idx_,env1T,env1,env2T,env2,code)	   \
    static char name[] =	  					                   \
        "kern_" #name "|"                                                        	   \
        #inT "|"                                                                	   \
"\n\n" #inT " f" #name "(\n"                                                   		   \
"\t__global " #inT "* " #in ",\n"							   \
"\tconst uint " #size ",\n"                        					   \
"\tconst int " #idx ",\n"                                                        	   \
"\tconst int " #idx_ ",\n"                                                        	   \
"\t__global const " #env1T "* " #env1 ",\n"                                      	   \
"\t__global const " #env2T "* " #env2 ") {\n"                                    	   \
"\t   " #code ";\n"                                                              	   \
"}\n\n"                                                                          	   \
"__kernel void kern_" #name "(\n"                                                	   \
"\t__global " #inT  "* input,\n"                                                 	   \
"\t__global " #inT "* output,\n"                                                	   \
"\tconst uint inSize,\n"                                                         	   \
"\tconst uint maxItems,\n"                                                  	   	   \
"\tconst uint offset,\n"                                                  	   	   \
"\tconst uint pad,\n"                                                                      \
"\t__global const " #env1T "* env1,\n"                                           	   \
"\t__global const " #env2T "* env2) {\n"                                                   \
"\t    int i = get_global_id(0);\n"                                              	   \
"\t    int ig = i + offset;\n"                                              		   \
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"                   	   \
"\t    while(i < maxItems)  {\n"                                                 	   \
"\t        output[i+pad] = f" #name "(input+pad,inSize,ig,i,env1,env2);\n"                 \
"\t        i += gridSize;\n"                                                     	   \
"\t    }\n"                                                                      	   \
"}\n"



//  x=f(param1,param2)   'x', 'param1', 'param2' have the same type
#define FF_OCL_STENCIL_COMBINATOR(name, basictype, param1, param2, code)             \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #basictype "|"                                                  \
#basictype " f" #name "(" #basictype " " #param1 ",\n"                  \
                          #basictype " " #param2 ") {\n" #code ";\n}\n" \
"__kernel void kern_" #name "(__global " #basictype "* input, const uint pad, __global " #basictype "* output, const uint n, __local " #basictype "* sdata, "#basictype" idElem) {\n" \
"        uint blockSize = get_local_size(0);\n"                         \
"        uint tid = get_local_id(0);\n"                                 \
"        uint i = get_group_id(0)*blockSize + get_local_id(0);\n"       \
"        uint gridSize = blockSize*get_num_groups(0);\n"                \
"        " #basictype " result = idElem; input += pad;\n"               \
"        if(i < n) { result = input[i]; i += gridSize; }\n"             \
"        while(i < n) {\n"                                              \
"          result = f" #name "(result, input[i]);\n"                    \
"          i += gridSize;\n"                                            \
"        }\n"                                                           \
"        sdata[tid] = result;\n"                                        \
"        barrier(CLK_LOCAL_MEM_FENCE);\n"                               \
"        if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(tid == 0) output[get_group_id(0)] = sdata[tid];\n"          \
"}\n";



/* The following OpenCL code macros have been "inspired" by the
 * SkePU OpenCL code
 * http://www.ida.liu.se/~chrke/skepu/
 */

//  x=f(param,idx)   'x' and 'param' have the same type
#define FF_OCL_MAP_ELEMFUNC(name, basictype, param, idx, code)		                   \
    static char name[] =                                                                   \
        "kern_" #name "|"                                                                  \
        #basictype "|"                                                                     \
#basictype " f" #name "(" #basictype " " #param ",const int " #idx ") {\n" #code ";\n}\n"  \
"__kernel void kern_" #name "(\n"                                                	   \
"\t__global " #basictype  "* input,\n"                                                 	   \
"\t__global " #basictype "* output,\n"                                                	   \
"\tconst uint inSize,\n"                                                         	   \
"\tconst uint maxItems,\n"                                                  	   	   \
"\tconst uint offset,\n"                                                  	   	   \
"\tconst uint pad) {\n"                                               	      		   \
"\t    int i = get_global_id(0);\n"                                              	   \
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"                   	   \
"\t    while(i < maxItems)  {\n"                                                 	   \
"\t        output[i] = f" #name "(input[i],i);\n"          			           \
"\t        i += gridSize;\n"                                                     	   \
"\t    }\n"                                                                      	   \
"}\n"


//  x=f(param,idx)   'x' and 'param' have different types
#define FF_OCL_MAP_ELEMFUNC2(name, outT, inT, param, idx, code)		                   \
    static char name[] =                                                                   \
        "kern_" #name "|"                                                                  \
        #outT "|"                                                                          \
#outT " f" #name "(" #inT " " #param ", const int " #idx ") {\n" #code ";\n}\n"            \
"__kernel void kern_" #name "(\n"                                                	   \
"\t__global " #inT  "* input,\n"                                                 	   \
"\t__global " #outT "* output,\n"                                                	   \
"\tconst uint inSize,\n"                                                         	   \
"\tconst uint maxItems,\n"                                                  	   	   \
"\tconst uint offset,\n"                                                  	   	   \
"\tconst uint pad) {\n"                                               	      		   \
"\t    int i = get_global_id(0);\n"                                              	   \
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"                   	   \
"\t    while(i < maxItems)  {\n"                                                 	   \
"\t        output[i] = f" #name "(input[i],i);\n"          			           \
"\t        i += gridSize;\n"                                                     	   \
"\t    }\n"                                                                      	   \
"}\n"



#define FFGENERICFUNC(name, basictype, code)                            \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #basictype "|"                                                  \
        #code ";\n\n"


/* ------------------------------------------------------------------------------------- */



// NOTE: A better check would be needed !
// both GNU g++ and Intel icpc define __GXX_EXPERIMENTAL_CXX0X__ if -std=c++0x or -std=c++11 is used 
// (icpc -E -dM -std=c++11 -x c++ /dev/null | grep GXX_EX)
#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || defined(HAS_CXX11_VARIADIC_TEMPLATES)

template< typename ... Args >
std::string stringer(Args const& ... args ) {
    std::ostringstream stream;
    using List= int[];
    
    // expanding a parameter pack is only valid in contexts 
    // where the parser expects a comma-separated list of entries
    (void)List{0, ( (void)(stream << args), 0 ) ... };
    
    return stream.str();
}
    
#define STRINGER(...) stringer(__VA_ARGS__)
    
#endif // c++11 check




}  // namespace

#endif /* STENCILREDUCEOCL_MACROS_HPP_ */
