/* needed header files .... */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>



void import_JPEG_file (const char* filename, unsigned char** image_chars,
                       int* image_height, int* image_width,
                       int* num_components);
void export_JPEG_file (const char* filename, const unsigned char* image_chars,
                       int image_height, int image_width,
                       int num_components, int quality);


typedef struct {
  float** image_data;
  int m; // vertical direction
  int n; // horizontal direction
}
image;
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

void allocate_image(image *u, int m, int n);
void deallocate_image(image *u);
void convert_jpeg_to_image(const unsigned char* image_chars, image *u);
void convert_image_to_jpeg(const image *u, unsigned char* image_chars);
void iso_diffusion_denoising_parallel(image *u, image *u_bar, float k, int iters);
void swap_images(image *u, image *u_bar, int m, int n);

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
  /*
  if (argc<5) {
    printf("Need five input arguments from command line.\n");
    printf("./prog kappa iters input_jpeg_filename output_jpeg_file\n");
    printf("Exiting program.\n");
    MPI_Finalize();
    exit(0);
  }
  */
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
    allocate_image (&whole_image, m, n);
  }
  MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  /* 2D decomposition of the m x n pixels evenly among the MPI processes */
  /* Slicing in the first dimension only */
  /* Using twelve processes */
  int m_slice = m/num_procs;
  int remainder_slice = m%num_procs;
  int *m_array = malloc(num_procs*sizeof(int));
  int *counts = malloc(num_procs*sizeof(int));
  int *displs = malloc(num_procs*sizeof(int));

  for (int i = 0; i < num_procs-1; i++) {
        m_array[i] = m_slice + ((i >= (num_procs - remainder_slice)) ? 1:0);
        counts[i] = m_array[i]*n;
        displs[i+1] = displs[i] + counts[i];
    }
    m_array[num_procs-1] = m_slice + ((num_procs-1) >= (num_procs - remainder_slice) ? 1:0);
    counts[num_procs-1] = m_array[num_procs-1]*n;
  /* Allocating array for saving of number of rows in slices */
  if (my_rank==0 && my_rank==num_procs-1) {
    my_m = m_array[my_rank] + 1;
  }
  else{
    my_m = m_array[my_rank]+2;
  }
  my_n = n;
  allocate_image (&u, my_m, my_n);
  allocate_image (&u_bar, my_m, my_n);
  /* each process asks process 0 for a partitioned region */
  /* of image_chars and copy the values into u */
  /* ... */
  my_image_chars = malloc(my_m*my_n*sizeof(unsigned char));
  /* making sure scatter works fine */
  int start;
  if(my_rank< 0){
    start = n;
  }
  else{
    start = 0;
  }

  MPI_Scatterv(image_chars,                           // Sending image chars from root process
               counts,
               displs,
               MPI_UNSIGNED_CHAR,
               &my_image_chars[start],
               counts[my_rank],
               MPI_UNSIGNED_CHAR,
               0,
               MPI_COMM_WORLD);
  convert_jpeg_to_image (my_image_chars, &u);
  /* need to wait for all processes to be done */
  MPI_Barrier(MPI_COMM_WORLD);
  iso_diffusion_denoising_parallel (&u, &u_bar, kappa, iters);
  /* each process sends its resulting content of u_bar to process 0 */
  /* process 0 receives from each process incoming values and */
  /* copy them into the designated region of struct whole_image */
  /* ... */
  if (my_rank<0){
    start = 1;
  }
  else {
    start = 0;
  }
  MPI_Gatherv((&u)->image_data[start], m_array[my_rank]*n, MPI_FLOAT, (&whole_image)->image_data[0], counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
