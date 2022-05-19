//File name: midpoint_sum.cu
// Numerical Integration Using the Midpoint Rule using Unified memory architecture of Cuda
//input n = no. of term, Block_count = No. of Block, Thread_count = No. of thread
//Output: Summantion of the series
//      : Time taken to execute the program
//Compile: $ nvcc -O2 -arch=sm_30 -o midpoint_sum midpoint_sum.cu
//Run: $srun -p gpu --gres gpu:1 -n 1 -N 1 --pty --mem 1000 -t 3:00 --reservation=cscgpu bash
// $ ./midpoint_sum 16 2 4
// Written By Saswati Bhattacharjee 

//Unified midpoint sum
#include<stdio.h>
#include<cuda.h>
#include<math.h>
#include "timer.h"
//Serial sum function prototype declaration
double midpoint(double h_lowerlimit, double h_upperlimit, int h_interval);

__global__ void midpoint_sum(float *a, float *c, float*sum, float step,  int n){


        int tid;
        float local_sum=0.0;
        //Get global thread ID
        tid = blockDim.x*blockIdx.x+threadIdx.x;
        //This program will run if we assign 2 tasks per thread
        //Need to remove the commented portion as per requirement
        int loopstart = 2*tid; //assign 2 task per thread
        int loopend = loopstart+1;
        /* Assign 16 tasks per thread
        int loopstart = 16*tid;
        int loopend = loopstart+15;*/
        /* Assign 256 tasks per thread
           int loopstart=256*tid;
           int loopend = loopstart+255;*/
        /*Assign one task to one thread
          int loopstart = 1*tid;
          int loopend = loopstart+0;*/


        if(tid<n){
        for(int i=loopstart;i<=loopend;i++){
                local_sum+=(step)*(4.0/(1.0+(a[i]*a[i])));

        }

        c[tid]=local_sum;
                atomicAdd(sum,c[tid]);
        }
}

int main(int argc, char* argv[]){

        int thread_count, block_count,n;
        n = strtol(argv[1],NULL,10);
        float *a,  *c, *sum;
        float interval_length[n];
        float mid_values[n-1];
        int j,m; //counter variables
        float lower_limit=0.0, upper_limit=1.0;
        float step;
        double start,finish;

        step =(float) ((upper_limit - lower_limit)/n);


        for( j=0; j<=n;j++){
                interval_length[j] = lower_limit;
                lower_limit= lower_limit + step;

        }

        for(m=0; m<n; m++){
                mid_values[m]=(interval_length[m]+interval_length[m+1])/2.0;
        }
        block_count = strtol(argv[2],NULL,10);
		    thread_count = strtol(argv[3], NULL, 10);
        //shared memory managment
        cudaMallocManaged(&a, n*sizeof(float));
        cudaMallocManaged(&c, n*sizeof(float));
        cudaMallocManaged(&sum, n*sizeof(float));
			for(int i=0; i<n; i++){

              		a[i]= mid_values[i];
			}

		GET_TIME(start);

//Launch Kernel

        midpoint_sum<<<block_count,thread_count>>>(a, c,sum,step, n);

        //Synchronize threads
        cudaDeviceSynchronize();
	 GET_TIME(finish);
			
        printf("\nCuda_parallel_sum=%f\n",*sum);
                        cudaFree(a);
                        cudaFree(c);
                        cudaFree(sum);

       printf("\nElapsed time for cuda=%e seconds\n", finish-start);

      double I_value,h_start,h_finish;//variables to execute serial code
      GET_TIME(h_start);
		  I_value = midpoint(0.0,1.0,16); //serial code execution
      GET_TIME(h_finish);
      printf("\nThe answer of the serial_integration is %lf\n", I_value);
      printf("\nElapsed time for host=%e seconds\n", h_finish-h_start);

        return 0;
}
//Serial code function
double midpoint(double h_a, double h_b, int nk){
    double h_interval_length[nk];
    double h_mid[nk-1];
    double h_interval,h_sum, h_partial_sum=0.0;
    h_interval = (h_b-h_a)/(double)nk;

// If lowerlimit=0, upper limit =1 and f[lowerlimit, upperlimit] is divided into  n  subintervals, each of length (upperlimit-lowerlimit)/n. Store the length of the subinterval into interval_length[] array.

                for(int h_i= 0; h_i<=nk; h_i++){
        				h_interval_length[h_i]= h_a;
        				h_a =h_a + h_interval;
    				}
			//Store the value of the midpoints into mid array
    				for(int h_i=0; h_i<nk; h_i++){
        					h_mid[h_i] = (h_interval_length[h_i]+h_interval_length[h_i+1])/2.0;
			// calculate the intermediate sum
        					h_partial_sum = h_partial_sum + (1/(1+ pow(h_mid[h_i],2)));
				}

    			h_sum = (1.0/16.0)*(4.0)*h_partial_sum;
    return h_sum;
}
