#cmake_minimum_required(VERSION 2.8)

include_directories(    ${PROJECT_SOURCE_DIR}/
                        ${CMAKE_CURRENT_SOURCE_DIR}
                        ${CMAKE_CURRENT_SOURCE_DIR}/include
                    )

set ( SHARED_CPPS ccnt_lut.cpp ref_read.cpp alphabet.cpp shmem.cpp 
              edit.cpp bt2_idx.cpp bt2_io.cpp bt2_util.cpp 
              reference.cpp ds.cpp multikey_qsort.cpp limit.cpp 
			  random_source.cpp) 
			  
set ( SEARCH_CPPS qual.cpp pat.cpp sam.cpp 
              read_qseq.cpp aligner_seed_policy.cpp 
              aligner_seed.cpp 
			  aligner_seed2.cpp 
			  aligner_sw.cpp 
			  aligner_sw_driver.cpp aligner_cache.cpp 
			  aligner_result.cpp ref_coord.cpp mask.cpp 
			  pe.cpp aln_sink.cpp dp_framer.cpp 
			  scoring.cpp presets.cpp unique.cpp 
			  simple_func.cpp 
			  random_util.cpp 
			  aligner_bt.cpp sse_util.cpp 
			  aligner_swsse.cpp outq.cpp 
			  aligner_swsse_loc_i16.cpp 
			  aligner_swsse_ee_i16.cpp 
			  aligner_swsse_loc_u8.cpp 
			  aligner_swsse_ee_u8.cpp 
			  aligner_driver.cpp 
			   )



set ( BT2_FF_SRCS bt2_search.cpp 
		${SHARED_CPPS}
		${SEARCH_CPPS}	
                bowtie_main.cpp 		
 )

set ( BT2_SRCS bt2_search.cpp 
		${SHARED_CPPS}
		${SEARCH_CPPS}
		#metrics_utils.cpp	
                bowtie_main.cpp 		
 )			
			
include_directories( 
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
    ${CURRENT_SOURCE_DIR}    
    )


if(CMAKE_SIZEOF_VOID_P EQUAL 8) 
    set(ARCH_64 TRUE) 
else() 
    set(ARCH_64 FALSE) 
endif() 

if ( ARCH_64 )
	set (BITS_FLAG -m64)
else()
	set (BITS_FLAG -m32)
endif()


IF (UNIX)
file( MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bt2 )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY bt2 )
add_executable ( bowtie2.0.6 ${BT2_SRCS} )
file (COPY bowtie2 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bt2)
ENDIF (UNIX)

file( MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bt2ff )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY bt2ff )
add_executable ( bowtie2.0.6-ff ${BT2_FF_SRCS} )
file (COPY bowtie2 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bt2ff)


site_name(HOST_NAME)
string(TIMESTAMP TS "%d/%m/%Y")
SET (CO-BT2-FF " -DBOWTIE2  -DBOWTIE_PTHREADS -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -DFF")
SET (CO-BT2 " -DBOWTIE2 -DBOWTIE_PTHREADS -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE")
#SET(cat_prog cat)

IF(UNIX)
    SET (CO-BT2-FF "${CO-BT2-FF} -funroll-loops -finline-functions -fno-strict-aliasing -msse3 -Wall " )
	SET (CO-BT2 "${CO-BT2} -funroll-loops -finline-functions -fno-strict-aliasing -msse3 -Wall " )
ENDIF(UNIX)

# -DNDEBUG is added by cmake with Debug generation type
# ---- numa
find_package(LibNuma)
if ( ${LIBNUMA_LIBRARIES} MATCHES LIBNUMA_LIBRARIES-NOTFOUND )
message (WARNING  " LibNUMA not found, Bowtie-FF will not benefit from memory affinity") 
target_link_libraries ( bowtie2.0.6-ff  ${CMAKE_THREAD_LIBS_INIT} )
else ()
SET (CO-BT2-FF "${CO-BT2-FF} -DFF_NUMA_ALLOC -DFF_INDEX_INTERLEAVED" )
INCLUDE_DIRECTORIES( ${LIBNUMA_INCLUDE_DIRS} )
target_link_libraries ( bowtie2.0.6-ff  ${LIBNUMA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
endif ()

#EXEC_PROGRAM(${cat_prog}  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION OUTPUT_VARIABLE ver)
FILE(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" ver)

# ----
set_target_properties (bowtie2.0.6-ff PROPERTIES COMPILE_FLAGS "${CO-BT2-FF} -DBOWTIE2_VERSION=\"\\\"${ver}\\\"\" -DBUILD_HOST=\"\\\"${HOST_NAME}\\\"\" -DBUILD_TIME=\"\\\"${TS}\\\"\"  -DCOMPILER_VERSION=\"\\\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\\\"\"  -DCOMPILER_OPTIONS=\"\\\"${CO-BT2-FF}\\\"\" " OUTPUT_NAME bowtie2-align)

IF (UNIX)
set_target_properties (bowtie2.0.6 PROPERTIES COMPILE_FLAGS "${CO-BT2} -DBOWTIE2_VERSION=\"\\\"${ver}\\\"\" -DBUILD_HOST=\"\\\"${HOST_NAME}\\\"\" -DBUILD_TIME=\"\\\"${TS}\\\"\"  -DCOMPILER_VERSION=\"\\\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\\\"\"  -DCOMPILER_OPTIONS=\"\\\"${CO-BT2}\\\"\"  " OUTPUT_NAME bowtie2-align)
# -DFF -DFF_NUMA_ALLOC -DFF_INDEX_INTERLEAVED
target_link_libraries ( bowtie2.0.6 ${CMAKE_THREAD_LIBS_INIT})
add_test( bowtie2.0.6 ${CMAKE_CURRENT_BINARY_DIR}/bt2/bowtie2 -t -p 2 -x ${CMAKE_CURRENT_SOURCE_DIR}/example/index/lambda_virus -U ${CMAKE_CURRENT_SOURCE_DIR}/example/reads/longreads.fq -S ${CMAKE_CURRENT_BINARY_DIR}/out.sam )

add_test( bowtie2.0.6-ff ${CMAKE_CURRENT_BINARY_DIR}/bt2ff/bowtie2 -t -p 2 -x ${CMAKE_CURRENT_SOURCE_DIR}/example/index/lambda_virus -U ${CMAKE_CURRENT_SOURCE_DIR}/example/reads/longreads.fq -S ${CMAKE_CURRENT_BINARY_DIR}/out.ff.sam )
ELSE (UNIX)
# Win32
add_test( bowtie2.0.6-ff ${CMAKE_CURRENT_BINARY_DIR}/bt2ff/bowtie2-align -t -p 2 -x ${CMAKE_CURRENT_SOURCE_DIR}/example/index/lambda_virus -U ${CMAKE_CURRENT_SOURCE_DIR}/example/reads/longreads.fq -S ${CMAKE_CURRENT_BINARY_DIR}/out.ff.sam )
ENDIF (UNIX)
