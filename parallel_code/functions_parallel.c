#include <stdio.h>
#include <stdlib.h>
#include "functions.h"
#include <mpi.h>

void allocate_image(image *u, int m, int n){
    u -> image_data = malloc(m * sizeof(u->image_data));
    u -> image_data[0] = malloc(m*n * sizeof(u-> image_data[0]));

    for (int i = 1; i<m ; i++){
      u -> image_data[i]= &(u -> image_data[0][i*n]);
    }
    u->m = m;
    u->n = n;

}
void deallocate_image(image *u){
  free(u->image_data[0]);
  free(u->image_data);

}

void convert_jpeg_to_image(const unsigned char* image_chars, image *u){
  int m = u->m;
  int n = u->n;
  for (int i=0; i<m; i++){
    for (int j=0; j<n; j++){
      u->image_data[i][j] = (float)image_chars[n*i + j];
    }
  }
}
void convert_image_to_jpeg(const image *u, unsigned char* image_chars){
  int m = u->m;
  int n = u->n;
  for (int i=0; i< m; i++){
    for (int j=0; j<n; j++){
      image_chars[n*i + j] = (unsigned char) u->image_data[i][j];
    }
  }
}


void halos_update(image *u){
  MPI_Status status;
  int my_rank, num_procs;
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  int m = u->m;
  int n = u->n;
  /* Starting with the odd numbered processes.
     Checking for boundaries before send and receive. */
  if (my_rank % 2){
    /* Sending */
    MPI_Send(&u->image_data[1][0], n, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD);
    if (my_rank != num_procs-1){
      MPI_Send(&u->image_data[m-2][0], n, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD);
    }
    /* Receiving */
    MPI_Recv(&u->image_data[0][0], n, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);
    if (my_rank != num_procs-1){
      MPI_Recv(&u->image_data[m-1][0], n, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD, &status);
    }
  }
  /* Doing the same in opposite order for
     even numbered processes. */
  else{
    /* Receiving */
    if(my_rank !=0){
      MPI_Recv(&u->image_data[0][0], n, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);
    }
    if(my_rank != num_procs-1){
      MPI_Recv(&u->image_data[m-1][0], n, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD, &status);
    }
    /* Sending */
    if (my_rank != 0){
      MPI_Send(&u->image_data[1][0], n, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD);
    }
    if (my_rank != num_procs-1){
      MPI_Send(&u->image_data[m-2][0], n, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD);
    }
  }
}

void iso_diffusion_denoising_parallel(image *u, image *u_bar, float kappa, int iters){
  int my_rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  halos_update(u);
  int m = u->m;
  int n = u->n;
  float cross_sum, four_times_val, difference;
  /* Minor change in boundaries. j = 0 and j=n-1 needs to be
  set for all. Only the first and last row needs to be set. */
  if (my_rank==0){
    for (int j = 1; j< n-1; j++){
      u_bar->image_data[0][j] = u->image_data[0][j];
    }
  }else if(my_rank == num_procs-1){
    for (int j = 1; j < n-1; j++){
      u_bar->image_data[m-1][j] = u->image_data[m-1][j];
    }
  }
  for (int i = 0; i < m ; i++){
      u_bar->image_data[i][0] = u->image_data[i][0];
      u_bar->image_data[i][n-1] = u->image_data[i][n-1];
  }
  float **temp;
  /* Executing diffusion iters times */
  for(int iteration=0; iteration<iters; iteration++){
    for (int i=1; i < m-1; i++){
      for (int j=1; j < n-1; j++){
        cross_sum = u->image_data[i-1][j] + u->image_data[i][j-1] + u->image_data[i+1][j] + u->image_data[i][j+1];
        four_times_val = 4*u->image_data[i][j];
        difference = cross_sum - four_times_val;

        u_bar->image_data[i][j] = u->image_data[i][j] + kappa*difference;
      }
    }
    /* Swapping u and u_bar between each
       execution except the last. */
    if (iteration<(iters-1)){
      temp = u_bar->image_data;
      u_bar->image_data = u->image_data;
      u->image_data = temp;
    }
    halos_update(u);
  }

}
