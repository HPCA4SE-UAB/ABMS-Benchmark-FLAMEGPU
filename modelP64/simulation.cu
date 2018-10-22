
/*
 * FLAME GPU v 1.4.0 for CUDA 6
 * Copyright 2015 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */

//Disable internal thrust warnings about conversions
#pragma warning(push)
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

// includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector_operators.h>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"

#pragma warning(pop)

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

/* Agent Memory */

/* person Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_person_list* d_persons;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_person_list* d_persons_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_person_list* d_persons_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_person_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_person_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_person_values;  /**< Agent sort identifiers value */
    
/* person state variables */
xmachine_memory_person_list* h_persons_start;      /**< Pointer to agent list (population) on host*/
xmachine_memory_person_list* d_persons_start;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_person_start_count;   /**< Agent population size counter */
int agent_number;                           /* Agent number read from 0.xml file */ 

/* person state variables */
xmachine_memory_person_list* h_persons_1;      /**< Pointer to agent list (population) on host*/
xmachine_memory_person_list* d_persons_1;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_person_1_count;   /**< Agent population size counter */ 

/* person state variables */
xmachine_memory_person_list* h_persons_2;      /**< Pointer to agent list (population) on host*/
xmachine_memory_person_list* d_persons_2;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_person_2_count;   /**< Agent population size counter */ 

/* person state variables */
xmachine_memory_person_list* h_persons_3;      /**< Pointer to agent list (population) on host*/
xmachine_memory_person_list* d_persons_3;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_person_3_count;   /**< Agent population size counter */ 


/* Message Memory */

/* agentLocation Message variables */
xmachine_message_agentLocation_list* h_agentLocations;         /**< Pointer to message list on host*/
xmachine_message_agentLocation_list* d_agentLocations;         /**< Pointer to message list on device*/
xmachine_message_agentLocation_list* d_agentLocations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_agentLocation_count;         /**< message list counter*/
int h_message_agentLocation_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_agentLocation_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_agentLocation_unsorted_index;		/**< unsorted index (hash) value for message */
#else
	uint * d_xmachine_message_agentLocation_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_agentLocation_values;  /**< message sort identifier values */
#endif
xmachine_message_agentLocation_PBM * d_agentLocation_partition_matrix;  /**< Pointer to PCB matrix */
float3 h_message_agentLocation_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
float3 h_message_agentLocation_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
int3 h_message_agentLocation_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_agentLocation_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_agentLocation_my_id_offset;
int h_tex_xmachine_message_agentLocation_x_offset;
int h_tex_xmachine_message_agentLocation_y_offset;
int h_tex_xmachine_message_agentLocation_z_offset;
int h_tex_xmachine_message_agentLocation_pbm_start_offset;
int h_tex_xmachine_message_agentLocation_pbm_end_or_count_offset;

/* agentCooperate Message variables */
xmachine_message_agentCooperate_list* h_agentCooperates;         /**< Pointer to message list on host*/
xmachine_message_agentCooperate_list* d_agentCooperates;         /**< Pointer to message list on device*/
xmachine_message_agentCooperate_list* d_agentCooperates_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_agentCooperate_count;         /**< message list counter*/
int h_message_agentCooperate_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_agentCooperate_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_agentCooperate_unsorted_index;		/**< unsorted index (hash) value for message */
#else
	uint * d_xmachine_message_agentCooperate_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_agentCooperate_values;  /**< message sort identifier values */
#endif
xmachine_message_agentCooperate_PBM * d_agentCooperate_partition_matrix;  /**< Pointer to PCB matrix */
float3 h_message_agentCooperate_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
float3 h_message_agentCooperate_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
int3 h_message_agentCooperate_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_agentCooperate_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_agentCooperate_source_id_offset;
int h_tex_xmachine_message_agentCooperate_destination_id_offset;
int h_tex_xmachine_message_agentCooperate_cooperate_offset;
int h_tex_xmachine_message_agentCooperate_x_offset;
int h_tex_xmachine_message_agentCooperate_y_offset;
int h_tex_xmachine_message_agentCooperate_z_offset;
int h_tex_xmachine_message_agentCooperate_message0_offset;
int h_tex_xmachine_message_agentCooperate_message1_offset;
int h_tex_xmachine_message_agentCooperate_message2_offset;
int h_tex_xmachine_message_agentCooperate_message3_offset;
int h_tex_xmachine_message_agentCooperate_message4_offset;
int h_tex_xmachine_message_agentCooperate_message5_offset;
int h_tex_xmachine_message_agentCooperate_message6_offset;
int h_tex_xmachine_message_agentCooperate_message7_offset;
int h_tex_xmachine_message_agentCooperate_message8_offset;
int h_tex_xmachine_message_agentCooperate_message9_offset;
int h_tex_xmachine_message_agentCooperate_message10_offset;
int h_tex_xmachine_message_agentCooperate_message11_offset;
int h_tex_xmachine_message_agentCooperate_message12_offset;
int h_tex_xmachine_message_agentCooperate_pbm_start_offset;
int h_tex_xmachine_message_agentCooperate_pbm_end_or_count_offset;

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** person_walk
 * Agent function prototype for walk function of person agent
 */
void person_walk(cudaStream_t &stream);

/** person_cooperate
 * Agent function prototype for cooperate function of person agent
 */
void person_cooperate(cudaStream_t &stream);

/** person_play
 * Agent function prototype for play function of person agent
 */
void person_play(cudaStream_t &stream);

/** person_compute
 * Agent function prototype for compute function of person agent
 */
void person_compute(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(0);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(0);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int closest_sqr_pow2(int x){
	int h, h_d;
	int l, l_d;
	
	//higher bound
	h = (int)pow(4, ceil(log(x)/log(4)));
	h_d = h-x;
	
	//escape early if x is square power of 2
	if (h_d == x)
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	l_d = x-l;
	
	//closest bound
	if(h_d < l_d)
		return h;
	else 
		return l;
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


void initialise(char * inputfile){

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  

	printf("Allocating Host and Device memeory\n");
  
	/* Agent memory allocation (CPU) */
	int xmachine_person_SoA_size = sizeof(xmachine_memory_person_list);
	h_persons_start = (xmachine_memory_person_list*)malloc(xmachine_person_SoA_size);
	h_persons_1 = (xmachine_memory_person_list*)malloc(xmachine_person_SoA_size);
	h_persons_2 = (xmachine_memory_person_list*)malloc(xmachine_person_SoA_size);
	h_persons_3 = (xmachine_memory_person_list*)malloc(xmachine_person_SoA_size);

	/* Message memory allocation (CPU) */
	int message_agentLocation_SoA_size = sizeof(xmachine_message_agentLocation_list);
	h_agentLocations = (xmachine_message_agentLocation_list*)malloc(message_agentLocation_SoA_size);
	int message_agentCooperate_SoA_size = sizeof(xmachine_message_agentCooperate_list);
	h_agentCooperates = (xmachine_message_agentCooperate_list*)malloc(message_agentCooperate_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outpus
			
	/* Set spatial partitioning agentLocation message variables (min_bounds, max_bounds)*/
	h_message_agentLocation_radius = (float)10;
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentLocation_radius, &h_message_agentLocation_radius, sizeof(float)));	
	    h_message_agentLocation_min_bounds = make_float3((float)0.0, (float)0.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentLocation_min_bounds, &h_message_agentLocation_min_bounds, sizeof(float3)));	
	h_message_agentLocation_max_bounds = make_float3((float)300, (float)300, (float)300);
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentLocation_max_bounds, &h_message_agentLocation_max_bounds, sizeof(float3)));	
	h_message_agentLocation_partitionDim.x = (int)ceil((h_message_agentLocation_max_bounds.x - h_message_agentLocation_min_bounds.x)/h_message_agentLocation_radius);
	h_message_agentLocation_partitionDim.y = (int)ceil((h_message_agentLocation_max_bounds.y - h_message_agentLocation_min_bounds.y)/h_message_agentLocation_radius);
	h_message_agentLocation_partitionDim.z = (int)ceil((h_message_agentLocation_max_bounds.z - h_message_agentLocation_min_bounds.z)/h_message_agentLocation_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentLocation_partitionDim, &h_message_agentLocation_partitionDim, sizeof(int3)));	
	
			
	/* Set spatial partitioning agentCooperate message variables (min_bounds, max_bounds)*/
	h_message_agentCooperate_radius = (float)10;
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentCooperate_radius, &h_message_agentCooperate_radius, sizeof(float)));	
	    h_message_agentCooperate_min_bounds = make_float3((float)0.0, (float)0.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentCooperate_min_bounds, &h_message_agentCooperate_min_bounds, sizeof(float3)));	
	h_message_agentCooperate_max_bounds = make_float3((float)300, (float)300, (float)300);
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentCooperate_max_bounds, &h_message_agentCooperate_max_bounds, sizeof(float3)));	
	h_message_agentCooperate_partitionDim.x = (int)ceil((h_message_agentCooperate_max_bounds.x - h_message_agentCooperate_min_bounds.x)/h_message_agentCooperate_radius);
	h_message_agentCooperate_partitionDim.y = (int)ceil((h_message_agentCooperate_max_bounds.y - h_message_agentCooperate_min_bounds.y)/h_message_agentCooperate_radius);
	h_message_agentCooperate_partitionDim.z = (int)ceil((h_message_agentCooperate_max_bounds.z - h_message_agentCooperate_min_bounds.z)/h_message_agentCooperate_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentCooperate_partitionDim, &h_message_agentCooperate_partitionDim, sizeof(int3)));	
	

	//read initial states
	readInitialStates(inputfile, h_persons_start, &h_xmachine_memory_person_start_count);
	
	
	/* person Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_persons, xmachine_person_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_persons_swap, xmachine_person_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_persons_new, xmachine_person_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_person_keys, xmachine_memory_person_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_person_values, xmachine_memory_person_MAX* sizeof(uint)));
	/* start memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_persons_start, xmachine_person_SoA_size));
	gpuErrchk( cudaMemcpy( d_persons_start, h_persons_start, xmachine_person_SoA_size, cudaMemcpyHostToDevice));
    
	/* 1 memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_persons_1, xmachine_person_SoA_size));
	gpuErrchk( cudaMemcpy( d_persons_1, h_persons_1, xmachine_person_SoA_size, cudaMemcpyHostToDevice));
    
	/* 2 memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_persons_2, xmachine_person_SoA_size));
	gpuErrchk( cudaMemcpy( d_persons_2, h_persons_2, xmachine_person_SoA_size, cudaMemcpyHostToDevice));
    
	/* 3 memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_persons_3, xmachine_person_SoA_size));
	gpuErrchk( cudaMemcpy( d_persons_3, h_persons_3, xmachine_person_SoA_size, cudaMemcpyHostToDevice));
    
	/* agentLocation Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agentLocations, message_agentLocation_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agentLocations_swap, message_agentLocation_SoA_size));
	gpuErrchk( cudaMemcpy( d_agentLocations, h_agentLocations, message_agentLocation_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_agentLocation_partition_matrix, sizeof(xmachine_message_agentLocation_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_agentLocation_local_bin_index, xmachine_message_agentLocation_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_agentLocation_unsorted_index, xmachine_message_agentLocation_MAX* sizeof(uint)));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_agentLocation_keys, xmachine_message_agentLocation_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_agentLocation_values, xmachine_message_agentLocation_MAX* sizeof(uint)));
#endif
	
	/* agentCooperate Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agentCooperates, message_agentCooperate_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agentCooperates_swap, message_agentCooperate_SoA_size));
	gpuErrchk( cudaMemcpy( d_agentCooperates, h_agentCooperates, message_agentCooperate_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_agentCooperate_partition_matrix, sizeof(xmachine_message_agentCooperate_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_agentCooperate_local_bin_index, xmachine_message_agentCooperate_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_agentCooperate_unsorted_index, xmachine_message_agentCooperate_MAX* sizeof(uint)));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_agentCooperate_keys, xmachine_message_agentCooperate_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_agentCooperate_values, xmachine_message_agentCooperate_MAX* sizeof(uint)));
#endif
		

	/*Set global condition counts*/

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */
	initConstants();
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));
} 


void sort_persons_start(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_person_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_person_start_count); 
	gridSize = (h_xmachine_memory_person_start_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_person_keys, d_xmachine_memory_person_values, d_persons_start);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_person_keys),  thrust::device_pointer_cast(d_xmachine_memory_person_keys) + h_xmachine_memory_person_start_count,  thrust::device_pointer_cast(d_xmachine_memory_person_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_person_agents, no_sm, h_xmachine_memory_person_start_count); 
	gridSize = (h_xmachine_memory_person_start_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_person_agents<<<gridSize, blockSize>>>(d_xmachine_memory_person_values, d_persons_start, d_persons_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_person_list* d_persons_temp = d_persons_start;
	d_persons_start = d_persons_swap;
	d_persons_swap = d_persons_temp;	
}

void sort_persons_1(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_person_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_person_1_count); 
	gridSize = (h_xmachine_memory_person_1_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_person_keys, d_xmachine_memory_person_values, d_persons_1);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_person_keys),  thrust::device_pointer_cast(d_xmachine_memory_person_keys) + h_xmachine_memory_person_1_count,  thrust::device_pointer_cast(d_xmachine_memory_person_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_person_agents, no_sm, h_xmachine_memory_person_1_count); 
	gridSize = (h_xmachine_memory_person_1_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_person_agents<<<gridSize, blockSize>>>(d_xmachine_memory_person_values, d_persons_1, d_persons_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_person_list* d_persons_temp = d_persons_1;
	d_persons_1 = d_persons_swap;
	d_persons_swap = d_persons_temp;	
}

void sort_persons_2(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_person_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_person_2_count); 
	gridSize = (h_xmachine_memory_person_2_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_person_keys, d_xmachine_memory_person_values, d_persons_2);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_person_keys),  thrust::device_pointer_cast(d_xmachine_memory_person_keys) + h_xmachine_memory_person_2_count,  thrust::device_pointer_cast(d_xmachine_memory_person_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_person_agents, no_sm, h_xmachine_memory_person_2_count); 
	gridSize = (h_xmachine_memory_person_2_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_person_agents<<<gridSize, blockSize>>>(d_xmachine_memory_person_values, d_persons_2, d_persons_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_person_list* d_persons_temp = d_persons_2;
	d_persons_2 = d_persons_swap;
	d_persons_swap = d_persons_temp;	
}

void sort_persons_3(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_person_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_person_3_count); 
	gridSize = (h_xmachine_memory_person_3_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_person_keys, d_xmachine_memory_person_values, d_persons_3);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_person_keys),  thrust::device_pointer_cast(d_xmachine_memory_person_keys) + h_xmachine_memory_person_3_count,  thrust::device_pointer_cast(d_xmachine_memory_person_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_person_agents, no_sm, h_xmachine_memory_person_3_count); 
	gridSize = (h_xmachine_memory_person_3_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_person_agents<<<gridSize, blockSize>>>(d_xmachine_memory_person_values, d_persons_3, d_persons_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_person_list* d_persons_temp = d_persons_3;
	d_persons_3 = d_persons_swap;
	d_persons_swap = d_persons_temp;	
}


void cleanup(){

	/* Agent data free*/
	
	/* person Agent variables */
	gpuErrchk(cudaFree(d_persons));
	gpuErrchk(cudaFree(d_persons_swap));
	gpuErrchk(cudaFree(d_persons_new));
	
	free( h_persons_start);
	gpuErrchk(cudaFree(d_persons_start));
	
	free( h_persons_1);
	gpuErrchk(cudaFree(d_persons_1));
	
	free( h_persons_2);
	gpuErrchk(cudaFree(d_persons_2));
	
	free( h_persons_3);
	gpuErrchk(cudaFree(d_persons_3));
	

	/* Message data free */
	
	/* agentLocation Message variables */
	free( h_agentLocations);
	gpuErrchk(cudaFree(d_agentLocations));
	gpuErrchk(cudaFree(d_agentLocations_swap));
	gpuErrchk(cudaFree(d_agentLocation_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_agentLocation_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_agentLocation_unsorted_index));
#else
	gpuErrchk(cudaFree(d_xmachine_message_agentLocation_keys));
	gpuErrchk(cudaFree(d_xmachine_message_agentLocation_values));
#endif
	
	/* agentCooperate Message variables */
	free( h_agentCooperates);
	gpuErrchk(cudaFree(d_agentCooperates));
	gpuErrchk(cudaFree(d_agentCooperates_swap));
	gpuErrchk(cudaFree(d_agentCooperate_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_agentCooperate_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_agentCooperate_unsorted_index));
#else
	gpuErrchk(cudaFree(d_xmachine_message_agentCooperate_keys));
	gpuErrchk(cudaFree(d_xmachine_message_agentCooperate_values));
#endif
	
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
}

void singleIteration(){

	/* set all non partitioned and spatial partitionded message counts to 0*/
	h_message_agentLocation_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentLocation_count, &h_message_agentLocation_count, sizeof(int)));
	
	h_message_agentCooperate_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_agentCooperate_count, &h_message_agentCooperate_count, sizeof(int)));
	

	/* Call agent functions in order itterating through the layer functions */

        // Start profiling, JJRG 03-07-2018
        //cudaProfilerStart ();	

	/* Layer 1*/
	person_walk(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	person_cooperate(stream1);
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	person_play(stream1);
	cudaDeviceSynchronize();
        // Stop profiling, JJRG 03-07-2018
        //cudaProfilerStop ();
  
	/* Layer 4*/
	person_compute(stream1);
	cudaDeviceSynchronize();
  
}

/* Environment functions */


void set_height(int* h_height){
	gpuErrchk(cudaMemcpyToSymbol(height, h_height, sizeof(int)));
}

void set_width(int* h_width){
	gpuErrchk(cudaMemcpyToSymbol(width, h_width, sizeof(int)));
}

void set_radius(int* h_radius){
	gpuErrchk(cudaMemcpyToSymbol(radius, h_radius, sizeof(int)));
}


/* Agent data access functions*/

    
int get_agent_person_MAX_count(){
    return xmachine_memory_person_MAX;
}


int get_agent_person_start_count(){
	//continuous agent
	return h_xmachine_memory_person_start_count;
	
}

xmachine_memory_person_list* get_device_person_start_agents(){
	return d_persons_start;
}

xmachine_memory_person_list* get_host_person_start_agents(){
	return h_persons_start;
}

int get_agent_person_1_count(){
	//continuous agent
	return h_xmachine_memory_person_1_count;
	
}

xmachine_memory_person_list* get_device_person_1_agents(){
	return d_persons_1;
}

xmachine_memory_person_list* get_host_person_1_agents(){
	return h_persons_1;
}

int get_agent_person_2_count(){
	//continuous agent
	return h_xmachine_memory_person_2_count;
	
}

xmachine_memory_person_list* get_device_person_2_agents(){
	return d_persons_2;
}

xmachine_memory_person_list* get_host_person_2_agents(){
	return h_persons_2;
}

int get_agent_person_3_count(){
	//continuous agent
	return h_xmachine_memory_person_3_count;
	
}

xmachine_memory_person_list* get_device_person_3_agents(){
	return d_persons_3;
}

xmachine_memory_person_list* get_host_person_3_agents(){
	return h_persons_3;
}


/* Agent functions */


	
/* Shared memory size calculator for agent function */
int person_walk_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** person_walk
 * Agent function prototype for walk function of person agent
 */
void person_walk(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_person_start_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_person_start_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_person_list* persons_start_temp = d_persons;
	d_persons = d_persons_start;
	d_persons_start = persons_start_temp;
	//set working count to current state count
	h_xmachine_memory_person_count = h_xmachine_memory_person_start_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_count, &h_xmachine_memory_person_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_person_start_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_start_count, &h_xmachine_memory_person_start_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_agentLocation_count + h_xmachine_memory_person_count > xmachine_message_agentLocation_MAX){
		printf("Error: Buffer size of agentLocation message will be exceeded in function walk\n");
		exit(0);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_walk, person_walk_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = person_walk_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_agentLocation_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_agentLocation_output_type, &h_message_agentLocation_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (walk)
	//Reallocate   : false
	//Input        : 
	//Output       : agentLocation
	//Agent Output : 
	GPUFLAME_walk<<<g, b, sm_size, stream>>>(d_persons, d_agentLocations, d_rand48);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_agentLocation_count += h_xmachine_memory_person_count;	
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_agentLocation_count, &h_message_agentLocation_count, sizeof(int)));	
	
#ifdef FAST_ATOMIC_SORTING
  //USE ATOMICS TO BUILD PARTITION BOUNDARY
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_agentLocation_partition_matrix, 0, sizeof(xmachine_message_agentLocation_PBM)));
  //
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_agentLocation_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	hist_agentLocation_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_agentLocation_local_bin_index, d_xmachine_message_agentLocation_unsorted_index, d_agentLocation_partition_matrix->end_or_count, d_agentLocations, state_list_size);
	gpuErrchkLaunch();
	
	thrust::device_ptr<int> ptr_count = thrust::device_pointer_cast(d_agentLocation_partition_matrix->end_or_count);
	thrust::device_ptr<int> ptr_index = thrust::device_pointer_cast(d_agentLocation_partition_matrix->start);
	thrust::exclusive_scan(thrust::cuda::par.on(stream), ptr_count, ptr_count + xmachine_message_agentLocation_grid_size, ptr_index); // scan
	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_agentLocation_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize; 	// Round up according to array size 
	reorder_agentLocation_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_agentLocation_local_bin_index, d_xmachine_message_agentLocation_unsorted_index, d_agentLocation_partition_matrix->start, d_agentLocations, d_agentLocations_swap, state_list_size);
	gpuErrchkLaunch();
#else
	//HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	//Get message hash values for sorting
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_agentLocation_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	hash_agentLocation_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_agentLocation_keys, d_xmachine_message_agentLocation_values, d_agentLocations);
	gpuErrchkLaunch();
	//Sort
	thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_agentLocation_keys),  thrust::device_pointer_cast(d_xmachine_message_agentLocation_keys) + h_message_agentLocation_count,  thrust::device_pointer_cast(d_xmachine_message_agentLocation_values));
	gpuErrchkLaunch();
	//reorder and build pcb
	gpuErrchk(cudaMemset(d_agentLocation_partition_matrix->start, 0xffffffff, xmachine_message_agentLocation_grid_size* sizeof(int)));
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_agentLocation_messages, reorder_messages_sm_size, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	int reorder_sm_size = reorder_messages_sm_size(blockSize);
	reorder_agentLocation_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_agentLocation_keys, d_xmachine_message_agentLocation_values, d_agentLocation_partition_matrix, d_agentLocations, d_agentLocations_swap);
	gpuErrchkLaunch();
#endif
	//swap ordered list
	xmachine_message_agentLocation_list* d_agentLocations_temp = d_agentLocations;
	d_agentLocations = d_agentLocations_swap;
	d_agentLocations_swap = d_agentLocations_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_person_1_count+h_xmachine_memory_person_count > xmachine_memory_person_MAX){
		printf("Error: Buffer size of walk agents in state 1 will be exceeded moving working agents to next state in function walk\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_person_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_person_Agents<<<gridSize, blockSize, 0, stream>>>(d_persons_1, d_persons, h_xmachine_memory_person_1_count, h_xmachine_memory_person_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_person_1_count += h_xmachine_memory_person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_1_count, &h_xmachine_memory_person_1_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int person_cooperate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_agentLocation));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** person_cooperate
 * Agent function prototype for cooperate function of person agent
 */
void person_cooperate(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_person_1_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_person_1_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_person_list* persons_1_temp = d_persons;
	d_persons = d_persons_1;
	d_persons_1 = persons_1_temp;
	//set working count to current state count
	h_xmachine_memory_person_count = h_xmachine_memory_person_1_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_count, &h_xmachine_memory_person_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_person_1_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_1_count, &h_xmachine_memory_person_1_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_agentCooperate_count + h_xmachine_memory_person_count > xmachine_message_agentCooperate_MAX){
		printf("Error: Buffer size of agentCooperate message will be exceeded in function cooperate\n");
		exit(0);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_cooperate, person_cooperate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = person_cooperate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//continuous agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_agentLocation_my_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentLocation_my_id_byte_offset, tex_xmachine_message_agentLocation_my_id, d_agentLocations->my_id, sizeof(int)*xmachine_message_agentLocation_MAX));
	h_tex_xmachine_message_agentLocation_my_id_offset = (int)tex_xmachine_message_agentLocation_my_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentLocation_my_id_offset, &h_tex_xmachine_message_agentLocation_my_id_offset, sizeof(int)));
	size_t tex_xmachine_message_agentLocation_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentLocation_x_byte_offset, tex_xmachine_message_agentLocation_x, d_agentLocations->x, sizeof(int)*xmachine_message_agentLocation_MAX));
	h_tex_xmachine_message_agentLocation_x_offset = (int)tex_xmachine_message_agentLocation_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentLocation_x_offset, &h_tex_xmachine_message_agentLocation_x_offset, sizeof(int)));
	size_t tex_xmachine_message_agentLocation_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentLocation_y_byte_offset, tex_xmachine_message_agentLocation_y, d_agentLocations->y, sizeof(int)*xmachine_message_agentLocation_MAX));
	h_tex_xmachine_message_agentLocation_y_offset = (int)tex_xmachine_message_agentLocation_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentLocation_y_offset, &h_tex_xmachine_message_agentLocation_y_offset, sizeof(int)));
	size_t tex_xmachine_message_agentLocation_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentLocation_z_byte_offset, tex_xmachine_message_agentLocation_z, d_agentLocations->z, sizeof(int)*xmachine_message_agentLocation_MAX));
	h_tex_xmachine_message_agentLocation_z_offset = (int)tex_xmachine_message_agentLocation_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentLocation_z_offset, &h_tex_xmachine_message_agentLocation_z_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_agentLocation_pbm_start_byte_offset;
	size_t tex_xmachine_message_agentLocation_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentLocation_pbm_start_byte_offset, tex_xmachine_message_agentLocation_pbm_start, d_agentLocation_partition_matrix->start, sizeof(int)*xmachine_message_agentLocation_grid_size));
	h_tex_xmachine_message_agentLocation_pbm_start_offset = (int)tex_xmachine_message_agentLocation_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentLocation_pbm_start_offset, &h_tex_xmachine_message_agentLocation_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentLocation_pbm_end_or_count_byte_offset, tex_xmachine_message_agentLocation_pbm_end_or_count, d_agentLocation_partition_matrix->end_or_count, sizeof(int)*xmachine_message_agentLocation_grid_size));
  h_tex_xmachine_message_agentLocation_pbm_end_or_count_offset = (int)tex_xmachine_message_agentLocation_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentLocation_pbm_end_or_count_offset, &h_tex_xmachine_message_agentLocation_pbm_end_or_count_offset, sizeof(int)));

	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_agentCooperate_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_agentCooperate_output_type, &h_message_agentCooperate_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (cooperate)
	//Reallocate   : false
	//Input        : agentLocation
	//Output       : agentCooperate
	//Agent Output : 
	GPUFLAME_cooperate<<<g, b, sm_size, stream>>>(d_persons, d_agentLocations, d_agentLocation_partition_matrix, d_agentCooperates, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//continuous agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentLocation_my_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentLocation_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentLocation_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentLocation_z));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentLocation_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentLocation_pbm_end_or_count));
    
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_agentCooperate_count += h_xmachine_memory_person_count;	
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_agentCooperate_count, &h_message_agentCooperate_count, sizeof(int)));	
	
#ifdef FAST_ATOMIC_SORTING
  //USE ATOMICS TO BUILD PARTITION BOUNDARY
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_agentCooperate_partition_matrix, 0, sizeof(xmachine_message_agentCooperate_PBM)));
  //
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_agentCooperate_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	hist_agentCooperate_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_agentCooperate_local_bin_index, d_xmachine_message_agentCooperate_unsorted_index, d_agentCooperate_partition_matrix->end_or_count, d_agentCooperates, state_list_size);
	gpuErrchkLaunch();
	
	thrust::device_ptr<int> ptr_count = thrust::device_pointer_cast(d_agentCooperate_partition_matrix->end_or_count);
	thrust::device_ptr<int> ptr_index = thrust::device_pointer_cast(d_agentCooperate_partition_matrix->start);
	thrust::exclusive_scan(thrust::cuda::par.on(stream), ptr_count, ptr_count + xmachine_message_agentCooperate_grid_size, ptr_index); // scan
	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_agentCooperate_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize; 	// Round up according to array size 
	reorder_agentCooperate_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_agentCooperate_local_bin_index, d_xmachine_message_agentCooperate_unsorted_index, d_agentCooperate_partition_matrix->start, d_agentCooperates, d_agentCooperates_swap, state_list_size);
	gpuErrchkLaunch();
#else
	//HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	//Get message hash values for sorting
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_agentCooperate_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	hash_agentCooperate_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_agentCooperate_keys, d_xmachine_message_agentCooperate_values, d_agentCooperates);
	gpuErrchkLaunch();
	//Sort
	thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_agentCooperate_keys),  thrust::device_pointer_cast(d_xmachine_message_agentCooperate_keys) + h_message_agentCooperate_count,  thrust::device_pointer_cast(d_xmachine_message_agentCooperate_values));
	gpuErrchkLaunch();
	//reorder and build pcb
	gpuErrchk(cudaMemset(d_agentCooperate_partition_matrix->start, 0xffffffff, xmachine_message_agentCooperate_grid_size* sizeof(int)));
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_agentCooperate_messages, reorder_messages_sm_size, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	int reorder_sm_size = reorder_messages_sm_size(blockSize);
	reorder_agentCooperate_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_agentCooperate_keys, d_xmachine_message_agentCooperate_values, d_agentCooperate_partition_matrix, d_agentCooperates, d_agentCooperates_swap);
	gpuErrchkLaunch();
#endif
	//swap ordered list
	xmachine_message_agentCooperate_list* d_agentCooperates_temp = d_agentCooperates;
	d_agentCooperates = d_agentCooperates_swap;
	d_agentCooperates_swap = d_agentCooperates_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_person_2_count+h_xmachine_memory_person_count > xmachine_memory_person_MAX){
		printf("Error: Buffer size of cooperate agents in state 2 will be exceeded moving working agents to next state in function cooperate\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_person_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_person_Agents<<<gridSize, blockSize, 0, stream>>>(d_persons_2, d_persons, h_xmachine_memory_person_2_count, h_xmachine_memory_person_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_person_2_count += h_xmachine_memory_person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_2_count, &h_xmachine_memory_person_2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int person_play_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_agentCooperate));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** person_play
 * Agent function prototype for play function of person agent
 */
void person_play(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_person_2_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_person_2_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_person_list* persons_2_temp = d_persons;
	d_persons = d_persons_2;
	d_persons_2 = persons_2_temp;
	//set working count to current state count
	h_xmachine_memory_person_count = h_xmachine_memory_person_2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_count, &h_xmachine_memory_person_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_person_2_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_2_count, &h_xmachine_memory_person_2_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_play, person_play_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = person_play_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//continuous agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_agentCooperate_source_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_source_id_byte_offset, tex_xmachine_message_agentCooperate_source_id, d_agentCooperates->source_id, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_source_id_offset = (int)tex_xmachine_message_agentCooperate_source_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_source_id_offset, &h_tex_xmachine_message_agentCooperate_source_id_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_destination_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_destination_id_byte_offset, tex_xmachine_message_agentCooperate_destination_id, d_agentCooperates->destination_id, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_destination_id_offset = (int)tex_xmachine_message_agentCooperate_destination_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_destination_id_offset, &h_tex_xmachine_message_agentCooperate_destination_id_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_cooperate_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_cooperate_byte_offset, tex_xmachine_message_agentCooperate_cooperate, d_agentCooperates->cooperate, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_cooperate_offset = (int)tex_xmachine_message_agentCooperate_cooperate_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_cooperate_offset, &h_tex_xmachine_message_agentCooperate_cooperate_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_x_byte_offset, tex_xmachine_message_agentCooperate_x, d_agentCooperates->x, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_x_offset = (int)tex_xmachine_message_agentCooperate_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_x_offset, &h_tex_xmachine_message_agentCooperate_x_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_y_byte_offset, tex_xmachine_message_agentCooperate_y, d_agentCooperates->y, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_y_offset = (int)tex_xmachine_message_agentCooperate_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_y_offset, &h_tex_xmachine_message_agentCooperate_y_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_z_byte_offset, tex_xmachine_message_agentCooperate_z, d_agentCooperates->z, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_z_offset = (int)tex_xmachine_message_agentCooperate_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_z_offset, &h_tex_xmachine_message_agentCooperate_z_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message0_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message0_byte_offset, tex_xmachine_message_agentCooperate_message0, d_agentCooperates->message0, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message0_offset = (int)tex_xmachine_message_agentCooperate_message0_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message0_offset, &h_tex_xmachine_message_agentCooperate_message0_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message1_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message1_byte_offset, tex_xmachine_message_agentCooperate_message1, d_agentCooperates->message1, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message1_offset = (int)tex_xmachine_message_agentCooperate_message1_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message1_offset, &h_tex_xmachine_message_agentCooperate_message1_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message2_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message2_byte_offset, tex_xmachine_message_agentCooperate_message2, d_agentCooperates->message2, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message2_offset = (int)tex_xmachine_message_agentCooperate_message2_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message2_offset, &h_tex_xmachine_message_agentCooperate_message2_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message3_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message3_byte_offset, tex_xmachine_message_agentCooperate_message3, d_agentCooperates->message3, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message3_offset = (int)tex_xmachine_message_agentCooperate_message3_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message3_offset, &h_tex_xmachine_message_agentCooperate_message3_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message4_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message4_byte_offset, tex_xmachine_message_agentCooperate_message4, d_agentCooperates->message4, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message4_offset = (int)tex_xmachine_message_agentCooperate_message4_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message4_offset, &h_tex_xmachine_message_agentCooperate_message4_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message5_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message5_byte_offset, tex_xmachine_message_agentCooperate_message5, d_agentCooperates->message5, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message5_offset = (int)tex_xmachine_message_agentCooperate_message5_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message5_offset, &h_tex_xmachine_message_agentCooperate_message5_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message6_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message6_byte_offset, tex_xmachine_message_agentCooperate_message6, d_agentCooperates->message6, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message6_offset = (int)tex_xmachine_message_agentCooperate_message6_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message6_offset, &h_tex_xmachine_message_agentCooperate_message6_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message7_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message7_byte_offset, tex_xmachine_message_agentCooperate_message7, d_agentCooperates->message7, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message7_offset = (int)tex_xmachine_message_agentCooperate_message7_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message7_offset, &h_tex_xmachine_message_agentCooperate_message7_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message8_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message8_byte_offset, tex_xmachine_message_agentCooperate_message8, d_agentCooperates->message8, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message8_offset = (int)tex_xmachine_message_agentCooperate_message8_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message8_offset, &h_tex_xmachine_message_agentCooperate_message8_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message9_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message9_byte_offset, tex_xmachine_message_agentCooperate_message9, d_agentCooperates->message9, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message9_offset = (int)tex_xmachine_message_agentCooperate_message9_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message9_offset, &h_tex_xmachine_message_agentCooperate_message9_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message10_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message10_byte_offset, tex_xmachine_message_agentCooperate_message10, d_agentCooperates->message10, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message10_offset = (int)tex_xmachine_message_agentCooperate_message10_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message10_offset, &h_tex_xmachine_message_agentCooperate_message10_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message11_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message11_byte_offset, tex_xmachine_message_agentCooperate_message11, d_agentCooperates->message11, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message11_offset = (int)tex_xmachine_message_agentCooperate_message11_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message11_offset, &h_tex_xmachine_message_agentCooperate_message11_offset, sizeof(int)));
	size_t tex_xmachine_message_agentCooperate_message12_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_message12_byte_offset, tex_xmachine_message_agentCooperate_message12, d_agentCooperates->message12, sizeof(int)*xmachine_message_agentCooperate_MAX));
	h_tex_xmachine_message_agentCooperate_message12_offset = (int)tex_xmachine_message_agentCooperate_message12_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_message12_offset, &h_tex_xmachine_message_agentCooperate_message12_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_agentCooperate_pbm_start_byte_offset;
	size_t tex_xmachine_message_agentCooperate_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_pbm_start_byte_offset, tex_xmachine_message_agentCooperate_pbm_start, d_agentCooperate_partition_matrix->start, sizeof(int)*xmachine_message_agentCooperate_grid_size));
	h_tex_xmachine_message_agentCooperate_pbm_start_offset = (int)tex_xmachine_message_agentCooperate_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_pbm_start_offset, &h_tex_xmachine_message_agentCooperate_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_agentCooperate_pbm_end_or_count_byte_offset, tex_xmachine_message_agentCooperate_pbm_end_or_count, d_agentCooperate_partition_matrix->end_or_count, sizeof(int)*xmachine_message_agentCooperate_grid_size));
  h_tex_xmachine_message_agentCooperate_pbm_end_or_count_offset = (int)tex_xmachine_message_agentCooperate_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_agentCooperate_pbm_end_or_count_offset, &h_tex_xmachine_message_agentCooperate_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (play)
	//Reallocate   : false
	//Input        : agentCooperate
	//Output       : 
	//Agent Output : 
	GPUFLAME_play<<<g, b, sm_size, stream>>>(d_persons, d_agentCooperates, d_agentCooperate_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//continuous agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_source_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_destination_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_cooperate));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message0));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message1));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message2));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message3));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message4));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message5));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message6));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message7));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message8));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message9));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message10));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message11));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_message12));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_agentCooperate_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_person_3_count+h_xmachine_memory_person_count > xmachine_memory_person_MAX){
		printf("Error: Buffer size of play agents in state 3 will be exceeded moving working agents to next state in function play\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_person_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_person_Agents<<<gridSize, blockSize, 0, stream>>>(d_persons_3, d_persons, h_xmachine_memory_person_3_count, h_xmachine_memory_person_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_person_3_count += h_xmachine_memory_person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_3_count, &h_xmachine_memory_person_3_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int person_compute_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** person_compute
 * Agent function prototype for compute function of person agent
 */
void person_compute(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_person_3_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_person_3_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_person_list* persons_3_temp = d_persons;
	d_persons = d_persons_3;
	d_persons_3 = persons_3_temp;
	//set working count to current state count
	h_xmachine_memory_person_count = h_xmachine_memory_person_3_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_count, &h_xmachine_memory_person_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_person_3_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_3_count, &h_xmachine_memory_person_3_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_compute, person_compute_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = person_compute_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (compute)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
//	GPUFLAME_compute<<<g, b, sm_size, stream>>>(d_persons, d_rand48);
//	gpuErrchkLaunch();
	
//***********************************************************************************************************
        // Start CUDA Visual Profiler
        //cudaProfilerStart ();

        // Code inserted to make la TF, JJRG 22-03-2018
        int N = 16384;
        int iterations, resta, numStreams;
        numStreams = 3;
        iterations = agent_number / numStreams;
        resta = agent_number % numStreams;

        //cufftHandle plan;
        cufftComplex *data0, *data1, *data2, *data3, *data4, *data5, *data6, *data7;
        cufftComplex *datadev0, *datadev1, *datadev2, *datadev3, *datadev4, *datadev5, *datadev6, *datadev7;

        if (resta != 0)
             iterations++;
        for (int i = 0; i < iterations; i++)
             {
             //CPU Memory Allocate and initializa
             data0 = (cufftComplex *) malloc (sizeof (cufftComplex) * N * 1);
             data1 = (cufftComplex *) malloc (sizeof (cufftComplex) * N * 1);
             data2 = (cufftComplex *) malloc (sizeof (cufftComplex) * N * 1);
             //data3 = (cufftComplex *) malloc (sizeof (cufftComplex) * N * 1);
             //data4 = (cufftComplex *) malloc (sizeof (cufftComplex) * N * 1);
             //data5 = (cufftComplex *) malloc (sizeof (cufftComplex) * N * 1);
             //data6 = (cufftComplex *) malloc (sizeof (cufftComplex) * N * 1);
             //data7 = (cufftComplex *) malloc (sizeof (cufftComplex) * N * 1);

             srand (123456 + i);
             for (int tf = 0; tf < N; tf++)
                  {
                  data0 [tf].x = (float) rand ();
                  data0 [tf].y = (float) rand ();
                  data1 [tf].x = (float) rand ();
                  data1 [tf].y = (float) rand ();
                  data2 [tf].x = (float) rand ();
                  data2 [tf].y = (float) rand ();
                  //data3 [tf].x = (float) rand ();
                  //data3 [tf].y = (float) rand ();
                  //data4 [tf].x = (float) rand ();
                  //data4 [tf].y = (float) rand ();
                  //data5 [tf].x = (float) rand ();
                  //data5 [tf].y = (float) rand ();
                  //data6 [tf].x = (float) rand ();
                  //data6 [tf].y = (float) rand ();
                  //data7 [tf].x = (float) rand ();
                  //data7 [tf].y = (float) rand ();
                  }

             //Registers host memory as page-locked (required for asynch cudaMemcpyAsync)
             gpuErrchk (cudaHostRegister (data0, sizeof (cufftComplex) * N, cudaHostRegisterPortable));
             gpuErrchk (cudaHostRegister (data1, sizeof (cufftComplex) * N, cudaHostRegisterPortable));
             gpuErrchk (cudaHostRegister (data2, sizeof (cufftComplex) * N, cudaHostRegisterPortable));
             //gpuErrchk (cudaHostRegister (data3, sizeof (cufftComplex) * N, cudaHostRegisterPortable));
             //gpuErrchk (cudaHostRegister (data4, sizeof (cufftComplex) * N, cudaHostRegisterPortable));
             //gpuErrchk (cudaHostRegister (data5, sizeof (cufftComplex) * N, cudaHostRegisterPortable));
             //gpuErrchk (cudaHostRegister (data6, sizeof (cufftComplex) * N, cudaHostRegisterPortable));
             //gpuErrchk (cudaHostRegister (data7, sizeof (cufftComplex) * N, cudaHostRegisterPortable));

             //GPU Memory allociate
             gpuErrchk (cudaMalloc ((void**) &datadev0, sizeof (cufftComplex) * N * 1));
             gpuErrchk (cudaMalloc ((void**) &datadev1, sizeof (cufftComplex) * N * 1));
             gpuErrchk (cudaMalloc ((void**) &datadev2, sizeof (cufftComplex) * N * 1));
             //gpuErrchk (cudaMalloc ((void**) &datadev3, sizeof (cufftComplex) * N * 1));
             //gpuErrchk (cudaMalloc ((void**) &datadev4, sizeof (cufftComplex) * N * 1));
             //gpuErrchk (cudaMalloc ((void**) &datadev5, sizeof (cufftComplex) * N * 1));
             //gpuErrchk (cudaMalloc ((void**) &datadev6, sizeof (cufftComplex) * N * 1));
             //gpuErrchk (cudaMalloc ((void**) &datadev7, sizeof (cufftComplex) * N * 1));

             //numStreams = 4;
             printf ("\nAgent number: %d, numStreams: %d\n", agent_number, numStreams);

             //Creates CUDA streams, as many as agent are.
             cudaStream_t streams [numStreams];
             for (int i = 0; i < numStreams; i++)
                  gpuErrchk (cudaStreamCreate (&streams [i]));

             //Creates cuFFT plans and sets them in streams
             cufftHandle *plans = (cufftHandle *) malloc (sizeof (cufftHandle) * numStreams);
             for (int i = 0; i < numStreams; i++)
                  {
                  cufftResult res = cufftPlan1d (&plans [i], N, CUFFT_C2C, 1);
                  cufftSetStream (plans [i], streams [i]);
                  }

             //Async memcopyes and computation
             //for (int i = 0; i < numStreams; i++)
             //     {
                  gpuErrchk (cudaMemcpyAsync (datadev0, data0, sizeof (cufftComplex) * N * 1, cudaMemcpyHostToDevice, streams[0]));
                  gpuErrchk (cudaMemcpyAsync (datadev1, data1, sizeof (cufftComplex) * N * 1, cudaMemcpyHostToDevice, streams[1]));
                  gpuErrchk (cudaMemcpyAsync (datadev2, data2, sizeof (cufftComplex) * N * 1, cudaMemcpyHostToDevice, streams[2]));
                  //gpuErrchk (cudaMemcpyAsync (datadev3, data3, sizeof (cufftComplex) * N * 1, cudaMemcpyHostToDevice, streams[3]));
                  //gpuErrchk (cudaMemcpyAsync (datadev4, data4, sizeof (cufftComplex) * N * 1, cudaMemcpyHostToDevice, streams[4]));
                  //gpuErrchk (cudaMemcpyAsync (datadev5, data5, sizeof (cufftComplex) * N * 1, cudaMemcpyHostToDevice, streams[5]));
                  //gpuErrchk (cudaMemcpyAsync (datadev6, data6, sizeof (cufftComplex) * N * 1, cudaMemcpyHostToDevice, streams[6]));
                  //gpuErrchk (cudaMemcpyAsync (datadev7, data7, sizeof (cufftComplex) * N * 1, cudaMemcpyHostToDevice, streams[7]));
                  //gpuErrchk (cudaMemcpy (datadev, data, sizeof (cufftComplex) * N * 1, cudaMemcpyHostToDevice));
                  //cufftResult res = cufftPlan1d (&plan, N, CUFFT_C2C, 1);
                  cufftExecC2C (plans [0], datadev0, datadev0, CUFFT_FORWARD);
                  cufftExecC2C (plans [1], datadev1, datadev1, CUFFT_FORWARD);
                  cufftExecC2C (plans [2], datadev2, datadev2, CUFFT_FORWARD);
                  //cufftExecC2C (plans [3], datadev3, datadev3, CUFFT_FORWARD);
                  //cufftExecC2C (plans [4], datadev4, datadev4, CUFFT_FORWARD);
                  //cufftExecC2C (plans [5], datadev5, datadev5, CUFFT_FORWARD);
                  //cufftExecC2C (plans [6], datadev6, datadev6, CUFFT_FORWARD);
                  //cufftExecC2C (plans [7], datadev7, datadev7, CUFFT_FORWARD);
             //     }

             for (int i = 0; i < numStreams; i++)
                  gpuErrchk (cudaStreamSynchronize (streams [i]));
                  //cudaDeviceSynchronize ();

             //Releases resources
             gpuErrchk (cudaHostUnregister (data0));
             gpuErrchk (cudaHostUnregister (data1));
             gpuErrchk (cudaHostUnregister (data2));
             //gpuErrchk (cudaHostUnregister (data3));
             //gpuErrchk (cudaHostUnregister (data4));
             //gpuErrchk (cudaHostUnregister (data5));
             //gpuErrchk (cudaHostUnregister (data6));
             //gpuErrchk (cudaHostUnregister (data7));

             //cufftDestroy (plan);
             gpuErrchk (cudaFree (datadev0));
             gpuErrchk (cudaFree (datadev1));
             gpuErrchk (cudaFree (datadev2));
             //gpuErrchk (cudaFree (datadev3));
             //gpuErrchk (cudaFree (datadev4));
             //gpuErrchk (cudaFree (datadev5));
             //gpuErrchk (cudaFree (datadev6));
             //gpuErrchk (cudaFree (datadev7));

             for(int i = 0; i < numStreams; i++)
                  gpuErrchk (cudaStreamDestroy (streams [i]));

             free (data0);
             free (data1);
             free (data2);
             //free (data3);
             //free (data4);
             //free (data5);
             //free (data6);
             //free (data7);
             }



        // Stop CUDA Visual Profiler
        //cudaProfilerStop ();
//********************************************************************************************************
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_person_start_count+h_xmachine_memory_person_count > xmachine_memory_person_MAX){
		printf("Error: Buffer size of compute agents in state start will be exceeded moving working agents to next state in function compute\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_person_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_person_Agents<<<gridSize, blockSize, 0, stream>>>(d_persons_start, d_persons, h_xmachine_memory_person_start_count, h_xmachine_memory_person_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_person_start_count += h_xmachine_memory_person_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_person_start_count, &h_xmachine_memory_person_start_count, sizeof(int)));	
	
	
}


 
extern "C" void reset_person_start_count()
{
    h_xmachine_memory_person_start_count = 0;
}
 
extern "C" void reset_person_1_count()
{
    h_xmachine_memory_person_1_count = 0;
}
 
extern "C" void reset_person_2_count()
{
    h_xmachine_memory_person_2_count = 0;
}
 
extern "C" void reset_person_3_count()
{
    h_xmachine_memory_person_3_count = 0;
}
