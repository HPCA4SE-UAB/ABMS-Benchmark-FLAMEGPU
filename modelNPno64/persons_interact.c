
/*
 * Copyright 2011 University of Sheffield.
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <cufft.h>
#include "header.h"

#define V_HEIGHT 300
#define V_WIDTH 300
#define V_RADIUS 10

// Constants moved from header.h, JJRG 17-12-2017 
__constant__ int height;
__constant__ int width;
__constant__ int radius;


__FLAME_GPU_INIT_FUNC__ void initConstants ()
     {
     int h_height = V_HEIGHT;
     int h_width = V_WIDTH;
     int h_radius = V_RADIUS;

     set_height (&h_height);
     set_width (&h_width);
     set_radius (&h_radius);
     }


__FLAME_GPU_FUNC__ double frand (RNG_rand48 *rand48)
     {
     double value;

     value = ((double) rnd (rand48) / (RAND_MAX));  // What is RAND_MAX value????

     return value;
     }


__FLAME_GPU_FUNC__ int isIntoCircle (int x, int y, int xCircle, int yCircle, int rCircle)
     {
     float v1 = powf (x - xCircle, 2);
     float v2 = powf (y - yCircle, 2);
     int dist = (int) sqrtf (v1 + v2);

     if (dist <= rCircle)
          return 0;
     else
          return 1;
     }


/**
 * walk FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_person. This represents a single agent instance and can be modified directly.
 * @param agentLocation_messages Pointer to output message list of type xmachine_message_agentLocation_list. Must be passed as an argument to the add_agentLocation_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int walk (xmachine_memory_person *agent, 
                             xmachine_message_agentLocation_list *agentLocation_messages, 
                             RNG_rand48 *rand48)
     {
     /* //Template for message output function use 
      * 
      * int my_id = 0;
      * float x = 0;
      * float y = 0;
      * float z = 0;
      * add_agentLocation_message(agentLocation_messages, my_id, x, y, z);
      */

     int newx;      // new posX
     int newy;      // new posY
     int xdir = 0;  // new dirX
     int ydir = 0;  // new dirY

     // Per ajustar el moviment al que fa RepastHPCDemoAgent::move()
     //xdir = rand()%(3-0)-1;
     //ydir = rand()%(3-0)-1;
     
     xdir = (rnd (rand48) / (float) RAND_MAX) < 0.5 ? -1 : 1;
     ydir = (rnd (rand48) / (float) RAND_MAX) < 0.5 ? -1 : 1;
     
     newx = ((int) agent->x) + xdir;
     newy = ((int) agent->y) + ydir;
     
     if (newx < 0)
          newx = 0;
     else if (newx >= height)
          newx = height;
  
     if (newy < 0)
          newy = 0;
     else if (newy >= width)
          newy = width;
     
     agent->x = (float) newx;
     agent->y = (float) newy;
     agent->z = 0.0f;
     
     add_agentLocation_message (agentLocation_messages, agent->my_id, agent->x, agent->y, agent->z);
    
     return 0;
     }


/**
 * cooperate FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_person. This represents a single agent instance and can be modified directly.
 * @param agentLocation_messages  agentLocation_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_agentLocation_message and get_next_agentLocation_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_agentLocation_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param agentCooperate_messages Pointer to output message list of type xmachine_message_agentCooperate_list. Must be passed as an argument to the add_agentCooperate_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int cooperate (xmachine_memory_person *agent, 
                                  xmachine_message_agentLocation_list *agentLocation_messages, 
                                  xmachine_message_agentCooperate_list *agentCooperate_messages, 
                                  RNG_rand48 *rand48)
     {
     /* //Template for input message itteration
      * 
      * xmachine_message_agentLocation* current_message = get_first_agentLocation_message(agentLocation_messages);
      * while (current_message)
      * {
      *     //INSERT MESSAGE PROCESSING CODE HERE
      *     current_message = get_next_agentLocation_message(current_message, agentLocation_messages);
      * }
      */
    
     /* //Template for message output function use 
      * 
      * int source_id = 0;
      * int destination_id = 0;
      * int cooperate = 0;
      * float x = 0;
      * float y = 0;
      * float z = 0;
      * double message = 0;
      * add_agentCooperate_message(agentCooperate_messages, source_id, destination_id, cooperate, x, y, z, message);
      */

     int cooperate_;
     double message = 123456789.0;  // Before a char array "0123456789"

     // Get the first agentLocation message
     xmachine_message_agentLocation *agentLocation_message;
     agentLocation_message = get_first_agentLocation_message (agentLocation_messages);

     while (agentLocation_message)
          {
          if (agent->my_id != agentLocation_message->my_id)
               {
               if (isIntoCircle (agent->x, agent->y, agentLocation_message->x, agentLocation_message->y, radius) == 0)
                    {
                    if (frand (rand48) < (agent->c * 1.0) / agent->total)
                         {
                         cooperate_ = 1;
                         }
                    else
                         cooperate_ = 0;

                    add_agentCooperate_message (agentCooperate_messages, agent->my_id,
                                                agentLocation_message->my_id, cooperate_,
                                                agent->x, agent->y, agent->z,
                                                message);
                    //printf ("Creat missatge agentCooperate, a_id: %d, aL_id: %d, coo: %d, x: %f, y: %f, z: %f, me: %f\n", 
                    //        agent->my_id, agentLocation_message->my_id, cooperate_, agent->x, agent->y, agent->z, message);
                    //printf ("Creat nou missatge agentCooperate, a_id: %d\n", agent->my_id);
                    }
               //else
               //     printf ("No és dins el circle i no creo missatge\n");
               }
          else
               add_agentCooperate_message (agentCooperate_messages, -1, -1, -1, -1.0, -1.0, -1.0,
                                           -1.0);

          agentLocation_message = get_next_agentLocation_message (agentLocation_message, agentLocation_messages);
          }
      
     return 0;
     }


/**
 * play FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_person. This represents a single agent instance and can be modified directly.
 * @param agentCooperate_messages  agentCooperate_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_agentCooperate_message and get_next_agentCooperate_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_agentCooperate_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int play (xmachine_memory_person *agent, 
                             xmachine_message_agentCooperate_list *agentCooperate_messages, 
                             RNG_rand48 *rand48)
     {
     /* //Template for input message itteration
      * 
      * xmachine_message_agentCooperate* current_message = get_first_agentCooperate_message(agentCooperate_messages);
      * while (current_message)
      * {
      *     //INSERT MESSAGE PROCESSING CODE HERE
      *     
      *     current_message = get_next_agentCooperate_message(current_message, agentCooperate_messages);
      * }
      */

     int iCooperated;
     int cPayoff = 0;
     int totalPayoff = 0;
     int i = 0;
     double message = 123456789.0;  // Before a string like "0123456789"
//printf ("Start Llegir missatges agentCooperate.\n"); 
     // Get the first  gentCooperate message
     xmachine_message_agentCooperate *agentCooperate_message;
     agentCooperate_message = get_first_agentCooperate_message (agentCooperate_messages);

     while (agentCooperate_message)
          {
          // Do I cooperate?
          if (frand (rand48) < (agent->c * 1.0) / agent->total)
               iCooperated = 1;
          else
               iCooperated = 0;

          // "if" added to control no useful message, and to solve problem with get_next_agentCooperate_message, JJRG 27-04-2018
          if (agentCooperate_message->cooperate != -1)
               {
               double payoff = (iCooperated ? (agentCooperate_message->cooperate ?  7 : 1) :     // If I cooperated, did my opponent?
                                              (agentCooperate_message->cooperate ? 10 : 3));     // If I didn't cooperate, did my opponent?
               if (iCooperated)
                    cPayoff += payoff;

               totalPayoff += payoff;

               i++;
               }

          // Verify messages
          //if (strncmp (message, agentCooperate_message->message, 10) == 0)
          //     printf ("play missatges IGUALS \n");
          //else
          //     printf( "play missatges DIFERENTS \n");

          agentCooperate_message = get_next_agentCooperate_message (agentCooperate_message, agentCooperate_messages);
          //if (agentCooperate_message)
          //     printf ("agentCooperate_message->source_id: %d, i: %d\n", agentCooperate_message->source_id, i);

          // Commented because it's added in before "if"
          //i++;
          }
//printf ("End llegir missatges agentCooperate.\n");

     agent->c += cPayoff;
     agent->total += totalPayoff;
     //printf ("NumAgents: %d\n", i);
      
     return 0;
     }


/**
 * compute FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_person. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int compute (xmachine_memory_person *agent, RNG_rand48 *rand48)
     {
     int N = 16384;
//     cufftHandle plan;
//     cufftComplex *data;
//
//     //int i = threadIdx.x + blockIdx.x * blockDim.x;
//     //printf ("Posicio i = %d\n", i);
//
//     cudaMalloc ((void**) &data, sizeof (cufftComplex)*N*1);
//
//     cufftResult res = cufftPlan1d (&plan, N, CUFFT_C2C, 1);
//     cufftExecC2C (plan, data, data, CUFFT_FORWARD);
//     cudaDeviceSynchronize();
//     cufftDestroy (plan);
//     cudaFree (data);

     //if (i<N)
     //     {
     //     //set next element
     //     set_person_agent_array_value<float> (agent->inx, i, frand (rand48));
     //     set_person_agent_array_value<float> (agent->iny, i, frand (rand48));
     //     //agent->inx [i] = frand (rand48);
     //     //agent->iny [i] = frand (rand48);
     //     }

     return 0;
     }


/*
int compute2 (xmachine_memory_person *agent, RNG_rand48 *rand48)
     {
     int i;
     //int N = 128;
     int N = 16384;
     int BATCH = 2;

     //fftw_complex *in, *out;
     //fftw_plan p;
     cufftComplex *in, *out;
     cufftHandle p;

     //srand (123456);
     //in = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) * N);
     //out = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) * N);

     for (i = 0; i < N; i++ )
          {
          in [i].x = frand (rand48);
          in [i].y = frand (rand48);
          }

     //p = fftw_plan_dft_1d (N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
     //fftw_execute (p);
     //fftw_destroy_plan (p);
     //fftw_free (in);
     //fftw_free (out);

     cufftPlan1d (&p, N, CUFFT_C2C, BATCH);
     cufftExecC2C (p, in, out, CUFFT_FORWARD);
     cudaDeviceSynchronize ();
     cufftDestroy (p);
     cudaFree (in);
     cudaFree (out);

     return 0;
     }
*/

#endif //_FLAMEGPU_FUNCTIONS
