#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "parallel_f.h"

void import_JPEG_file (const char* filename, unsigned char** image_chars,
                       int* image_height, int* image_width,
                       int* num_components);

void export_JPEG_file (const char* filename, const unsigned char* image_chars,
                       int image_height, int image_width,
                       int num_components, int quality);


int main(int argc, char *argv[])
    {
    int m, n, c, iters;
    int my_m, my_n, my_rank, start_point, num_procs;
    float kappa;
    image u, u_bar, whole_image;
    unsigned char *image_chars, *my_image_chars;
    char *input_jpeg_filename, *output_jpeg_filename;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
    kappa = atof(argv[1]);
    iters = atoi(argv[2]);
    input_jpeg_filename = argv[3];
    output_jpeg_filename = argv[4];
    // char *input_jpeg_filename = "mona_lisa_noisy.jpg";
    // char *output_jpeg_filename = "mona_lisa_parallel.jpg";
    // kappa = 0.2;
    // iters = 100;
    if (my_rank==0) {
        import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
        allocate_image (&whole_image, m, n);
    }else{
        allocate_image(&whole_image, 0, 0);
    }
    MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *m_rows = malloc(num_procs*sizeof *m_rows);
    //using for scattering and gathering. From excersice3 week10
    int *sendcounts = malloc(num_procs*sizeof *sendcounts);
    int *Sdispls = malloc(num_procs*sizeof *Sdispls);


    int rows = m/num_procs;
    int remainder = m%num_procs;
    Sdispls[0] = 0;

    // Last remainder processes gets an extra row.
    for (int my_rank = 0; my_rank < num_procs-1; my_rank++) {
        m_rows[my_rank] = rows + ((my_rank >= (num_procs - remainder)) ? 1:0);
        sendcounts[my_rank] = m_rows[my_rank]*n;
        Sdispls[my_rank+1] = Sdispls[my_rank] + sendcounts[my_rank];
    }
    m_rows[num_procs-1] = rows + ((num_procs-1) >= (num_procs - remainder) ? 1:0);
    sendcounts[num_procs-1] = m_rows[num_procs-1]*n;

    my_m = m_rows[my_rank]+1;// all the processors need at least one more row as the ghost row
    if (my_rank != 0 && my_rank != num_procs-1){// all the middle rankes need another ghost row. so two for middle ones and one for rank = 0 and last
        my_m++;
    }
    my_n = n;
    allocate_image (&u, my_m, my_n);
    allocate_image (&u_bar, my_m, my_n);
    my_image_chars = malloc(my_m * my_n *sizeof(*my_image_chars));
    if(my_rank == 0){// to scatter we have to make sure that only rank zero sends out from zero.
        start_point = 0;
    }else{
        start_point = n;
    }

    MPI_Scatterv(image_chars,                 // Sendbuff, matters only for root process.
                sendcounts,
                Sdispls,
                MPI_UNSIGNED_CHAR,
                &my_image_chars[start_point],                 // Recieve buff is the same as sendbuf here.
                sendcounts[my_rank],
                MPI_UNSIGNED_CHAR,
                0,
                MPI_COMM_WORLD);
    convert_jpeg_to_image (my_image_chars, &u);
    MPI_Barrier(MPI_COMM_WORLD); //make sure all arrays are filled before starting computation
    iso_diffusion_denoising_parallel (&u, &u_bar, kappa, iters);
    if (my_rank == 0){ // only rank 0 starts from 0 because it does not have a upper ghost row. the rest start from one to skip the upper ghost row
        start_point = 0;
    }else{
        start_point = 1;
    }
    MPI_Gatherv((&u)->image_data[start_point], m_rows[my_rank]*n, MPI_FLOAT, (&whole_image)->image_data[0], sendcounts, Sdispls, MPI_FLOAT, 0, MPI_COMM_WORLD);
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