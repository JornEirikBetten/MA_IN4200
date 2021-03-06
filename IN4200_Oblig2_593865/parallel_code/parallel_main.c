#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "functions_parallel.h"

void import_JPEG_file (const char* filename, unsigned char** image_chars,
                       int* image_height, int* image_width,
                       int* num_components);

void export_JPEG_file (const char* filename, const unsigned char* image_chars,
                       int image_height, int image_width,
                       int num_components, int quality);


int main(int argc, char *argv[])
    {
    int m, n, c, iters;
    int my_m, my_n, my_rank, num_procs;
    float kappa;
    image u, u_bar, whole_image;
    unsigned char *image_chars, *my_image_chars;
    char *input_jpeg_filename, *output_jpeg_filename;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
    if (argc<5) {
      printf("Command to run: \n");
      printf("./prog.exe (float)kappa (int)iters filename1 filename2");
      exit(0);
    }
    if (my_rank==0) {
      kappa = atof(argv[1]);
      iters = atoi(argv[2]);
      input_jpeg_filename = argv[3];
      output_jpeg_filename = argv[4];
      import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
      allocate_image (&whole_image, m, n);
    }else{
      allocate_image(&whole_image, 0, 0);
    }
    MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *m_array = malloc(num_procs*sizeof *m_array);

    /* arrays used for keeping track of displacements
    and number of elements to send/receive in MPI_Gatherv and
    MPI_Scatterv. */
    int *counts = malloc(num_procs*sizeof *counts);
    int *displs = malloc(num_procs*sizeof *displs);


    int hor_slice = m/num_procs;
    int remainder = m%num_procs;
    displs[0] = 0;

    /* Processes with rank higher than num_procs-remainder
       gain an additional row. */
    for (int my_rank = 0; my_rank < num_procs-1; my_rank++) {
        m_array[my_rank] = hor_slice + ((my_rank >= (num_procs - remainder)) ? 1:0);
        counts[my_rank] = m_array[my_rank]*n;
        displs[my_rank+1] = displs[my_rank] + counts[my_rank];
    }
    m_array[num_procs-1] = hor_slice + ((num_procs-1) >= (num_procs - remainder) ? 1:0);
    counts[num_procs-1] = m_array[num_procs-1]*n;

    if (my_rank==0 || my_rank == num_procs-1){
      my_m = m_array[my_rank]+1;
    }
    else{
      my_m = m_array[my_rank]+2;
    }
    my_n = n;
    allocate_image (&u, my_m, my_n);
    allocate_image (&u_bar, my_m, my_n);
    my_image_chars = malloc(my_m * my_n *sizeof(*my_image_chars));
    int start;
    if(my_rank == 0){
        /* the zeroth process start from the top
           of the image array */
        start = 0;
    }else{
        start = n;
    }
    /* Sending designated chunks of image to
       every */
    MPI_Scatterv(image_chars,
                counts,
                displs,
                MPI_UNSIGNED_CHAR,
                &my_image_chars[start],
                counts[my_rank],
                MPI_UNSIGNED_CHAR,
                0,
                MPI_COMM_WORLD);
    convert_jpeg_to_image (my_image_chars, &u);
    /* Wait for all processes to have converted their part of the image */
    MPI_Barrier(MPI_COMM_WORLD);
    iso_diffusion_denoising_parallel (&u, &u_bar, kappa, iters);
    if (my_rank == 0){
        start = 0; //rank one has no neighbour to the "left"
    }else{
        start = 1;
    }
    MPI_Gatherv((&u_bar)->image_data[start],
                counts[my_rank], MPI_FLOAT,
                (&whole_image)->image_data[0],
                counts,
                displs,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD);
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
