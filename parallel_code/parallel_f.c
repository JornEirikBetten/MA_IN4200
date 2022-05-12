#include <stdio.h>
#include <stdlib.h>
#include "parallel_f.h"
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
// nothing has to change in the convert because we are giving the function the correct boundries
void convert_jpeg_to_image(const unsigned char* image_chars, image *u){
  for (int i=0; i<u->m; i++){
    for (int j=0; j<u->n; j++){
      u->image_data[i][j] = (float)image_chars[u->n*i + j];
    }
  }
}
void convert_image_to_jpeg(const image *u, unsigned char* image_chars){
      for (int i=0; i< u->m; i++){
        for (int j=0; j<u->n; j++){
            image_chars[u->n*i + j] = (unsigned char) u->image_data[i][j];
    }
  }
}
// We have one extra row up and down for the ghost rows that needs to be updated and filled with the correct values each iteration. This means row num zero and my_m-1 (last row) are the ghost rows and rows number 1 and my_m-2 are the actual fisrt and last row that we neet to send.
// Additionally we are using odd-even implement of sending and recieving. meaning all odd processors send then recieve while all even processors are recieving then sending. This will prevent the deadlock and all processors will have the values they need.
void ghostrow_update(image *u){
  MPI_Status status;
  int my_rank, num_procs;
  // getting my_rank and num_procs locally for this function
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  if (my_rank % 2){ // odd: if the residues is 1 then it means true, therefore it is odd
    MPI_Send(&u->image_data[1][0], u->n, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD);// values of fisrt row send up to the lower ghost row from the upper neighbours
    if (my_rank != num_procs-1){// we have to make sure that if the last chunck is handled by an odd ranked processor it will not send anything down.
      MPI_Send(&u->image_data[u->m-2][0], u->n, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD);// sending values to the upper ghost row of the down neighbours
    }
    MPI_Recv(&u->image_data[0][0], u->n, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);// receiving for the upper ghost from the upper neighbour
    if (my_rank != num_procs-1){ // making sure the last processor does not receive from a non-existance down neighbour
      MPI_Recv(&u->image_data[u->m-1][0], u->n, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD, &status);// receiving for the down ghost from the down neighbour
    }
  }
  else{// even: now even recieves then sends
    if(my_rank !=0){// processor rank zero should not recieve anything from a non-existant upper neighbour
      MPI_Recv(&u->image_data[0][0], u->n, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);// recieving for the upper ghost row from upper neighbour
    }
    if(my_rank != num_procs-1){// last processor should not recieve from a down neghbour
      MPI_Recv(&u->image_data[u->m-1][0], u->n, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD, &status);// recieving for the down ghost row from down neighbour
    }
    // now we start sending from the even processors
    // We have to make sure processor rank zero does not send up and the last processor does not send down
    if (my_rank != 0){
      MPI_Send(&u->image_data[1][0], u->n, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD);
    }
    if (my_rank != num_procs-1){
      MPI_Send(&u->image_data[u->m-2][0], u->n, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD);
    }
  }
}

void iso_diffusion_denoising_parallel(image *u, image *u_bar, float kappa, int iters){
  int my_rank, num_procs;
  // getting my_rank and num_procs locally for this function
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  ghostrow_update(u);
  // fixing the left and right boundries. That would be column number zero and n-1 of each row.
  if (my_rank==0){ // for processor rank zero and the last one the actual boundry of the picture has to be fixed.
    for (int j = 1; j< u->n-1; j++){
      u_bar->image_data[0][j] = u->image_data[0][j];
    }
  }else if(my_rank == num_procs-1){
    for (int j = 1; j < u->n-1; j++){
      u_bar->image_data[u->m-1][j] = u->image_data[u->m-1][j];
    }
  }
  // In the serial code we filled the whole upper and lower neighbour then the left and right neighbour from 1 to m-2. But here we filled the upper and down neighbour from 1 to n-2 then fill the first and last point using the left and right loop.
  for (int i = 0; i < u->m ; i++){
      u_bar->image_data[i][0] = u->image_data[i][0];
      u_bar->image_data[i][u->n-1] = u->image_data[i][u->n-1];
  }
  float **temp;
  for(int k=0; k<iters; k++){ // start of the interation for noise removing function
    for (int i=1; i < u->m-1; i++){
      for (int j=1; j < u->n-1; j++){
        u_bar->image_data[i][j] = u->image_data[i][j] + kappa * (u->image_data[i-1][j]+u->image_data[i][j-1]-4*u->image_data[i][j]+u->image_data[i][j+1]+u->image_data[i+1][j]);
      }
    }
    temp = u_bar->image_data;
    u_bar->image_data = u->image_data;
    u->image_data = temp;
    ghostrow_update(u);
  }

}
