/* needed header files .... */
#include <stdio.h>
#include <stdlib.h>
#include "functions_parallel.c"
#include <mpi.h>

/* A try at virtual topology */

void import_JPEG_file (const char* filename, unsigned char** image_chars,
                       int* image_height, int* image_width,
                       int* num_components);
void export_JPEG_file (const char* filename, const unsigned char* image_chars,
                       int image_height, int image_width,
                       int num_components, int quality);


#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

/* declarations of functions import_JPEG_file and export_JPEG_file */
int main(int argc, char *argv[])
{
  int m, n, c, iters;
  int my_m, my_n, my_rank, num_procs;
  int my_mstart, my_mstop, my_nstart, my_nstop;
  int coord[2], id, dim[2], period[2], reorder;
  int up = 1;
  int down = -1;
  int left = -1;
  int right = 1;
  float kappa;
  MPI_Comm comm_cart;
  MPI_Status status;
  image u, u_bar, whole_image;
  unsigned char *image_chars, *my_image_chars;
  char *input_jpeg_filename, *output_jpeg_filename;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  /* read from command line: kappa, iters, input_jpeg_filename, output_jpeg_file
  name */
  if (argc<5) {
    printf("Need five input arguments from command line.\n");
    printf("./prog kappa iters input_jpeg_filename output_jpeg_file\n");
    printf("Exiting program.\n");
    MPI_Finalize();
    exit(0);
  }
  if (num_procs != 12) {
    printf("The program needs 12 processes to be initiated.Number of inserted processes: %d.\n Exiting program.\n", num_procs);
    exit(0);
  }
  /* ... */
  if (my_rank==0) {
    /* reading command line arguments */
    kappa = atof(argv[1]);
    iters = atoi(argv[2]);
    input_jpeg_filename = argv[3];
    output_jpeg_filename = argv[4];
    /* reading image into 1D array */
    import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
    printf("M: %d, N: %d\n", m,n);
    /* allocating an image with 2D float array inside */
    //allocate_image (&whole_image, m, n);
  }
  MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  /* 2D decomposition of the m x n pixels evenly among the MPI processes */
  /* Using twelve processes */
  dim[0] = 4; dim[1] = 3;
  period[0] = 0; period[1] = 0; /* Non periodic boundaries */
  reorder=0; /* false */

  int ndims = 2;
  MPI_Dims_create(num_procs, ndims, dim);
  if(my_rank==0){
    printf("PW[%d], CommSz[%d%]: PEdims = [%d x %d] \n",my_rank,num_procs,dim[0],dim[1]);
  }
  MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm_cart);
  MPI_Cart_coords(comm_cart, my_rank, 2, coord);
  printf("Rank %d coordinates are %d %d\n", my_rank, coord[0], coord[1]);fflush(stdout);
  int rank_dirs[4];
  //int dest[4];
  MPI_Cart_shift(comm_cart, 0, 1, &rank_dirs[UP], &rank_dirs[DOWN]);
  MPI_Cart_shift(comm_cart, 1, 1, &rank_dirs[LEFT], &rank_dirs[RIGHT]);
  printf("P[%d]: neighbors(u,d,l,r)=%d %d %d %d\n",my_rank, rank_dirs[UP],rank_dirs[DOWN],rank_dirs[LEFT], rank_dirs[RIGHT]);

  my_mstart = coord[0]*(m-2)/4;
  my_mstop = (coord[0]+1)*(m-2)/4;
  my_m = my_mstop-my_mstart+2;
  my_nstart = coord[1]*(n-2)/3;
  my_nstop = (coord[1]+1)*(n-2)/3;
  my_n = my_nstop-my_nstart+2;
  *my_image_chars = (unsigned char*)malloc(my_m*my_n*sizeof(unsigned char));
  printf("P[%d]: m: start: %d, stop: %d, my_m: %d\n", my_rank, my_mstart, my_mstop, my_m);
  printf("P[%d]: n: start: %d, stop: %d, my_n: %d\n", my_rank, my_nstart, my_nstop, my_n);

  allocate_image (&u, my_m, my_n);
  allocate_image (&u_bar, my_m, my_n);
  /* each process asks process 0 for a partitioned region */
  /* of image_chars and copy the values into u */
  /* ... */
  if (my_rank>0) {
    MPI_Send(&my_mstart, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&my_m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&my_nstart, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&my_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&my_image_chars, my_m*my_n, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);
  }
  if (my_rank==0) {
    int mstart, msize, nstart, nsize;
    for (int i=1; i<12; i++) {
      MPI_Recv(&mstart, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&msize, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&nstart, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&nsize, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      unsigned char *chunk = (unsigned char*)malloc(nsize*msize*sizeof(unsigned char));
      for (int i=mstart; i<mstart + msize; i++) {
        for (int j=nstart; j<nstart + nsize; j++) {
          chunk[(i-mstart)*nsize + (j-nstart)] = image_chars[i*n + j];
        }
      }
      MPI_Send(&chunk, nsize*msize, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
    }
    for (int i=my_mstart; i<my_mstart + my_m; i++) {
      for (int j=my_nstart; j<my_nstart+my_n; j++) {
        my_image_chars[(i-my_mstart)*nsize + (j-nstart)] = image_chars[i*n + j];
      }
    }
  }


  convert_jpeg_to_image (my_image_chars, &u);
  iso_diffusion_denoising_parallel (&u, &u_bar, kappa, iters);
  /* each process sends its resulting content of u_bar to process 0 */
  /* process 0 receives from each process incoming values and */
  /* copy them into the designated region of struct whole_image */
  /* ... */
  if (my_rank>0) {
    MPI_Send(&my_m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&my_mstart, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&my_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&my_nstart, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&u_bar.image_data, my_m*my_n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
  if (my_rank==0){
    for (int i=1; i<12; i++) {
      MPI_Recv(&msize, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&mstart, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&nsize, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&nstart, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      float** temp = (float**)malloc(msize*sizeof(float*));
      for (int i=0; i<msize; i++) {
        *temp[i] = (float*)malloc(nsize*sizeof(float)); 
      }
      MPI_Recv(&temp, msize*nsize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);

      for (int i=mstart+1; i<mstart + msize -1; i++){
        for (int j=nstart+1; j<nstart + nsize -1; j++) {
          whole_image.image_data[i][j] = temp[i-mstart][j-nstart];
        }
      }
    }
  }
  if (my_rank==0) {
    convert_image_to_jpeg(&whole_image, image_chars);
    export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
    deallocate_image (&whole_image);
  }
  deallocate_image (&u);
  deallocate_image (&u_bar);

  MPI_Finalize ();
  return 0;
}
