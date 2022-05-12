#include <stdio.h>

#include <stdlib.h>
#include "functions.h"

/* The purpose of this program is to demonstrate how the functions
   'import_JPEG_file' & 'export_JPEG_file' can be used. */

void import_JPEG_file (const char* filename, unsigned char** image_chars,
                       int* image_height, int* image_width,
                       int* num_components);
void export_JPEG_file (const char* filename, const unsigned char* image_chars,
                       int image_height, int image_width,
                       int num_components, int quality);

int main(int argc, char *argv[])
{
  int m, n, c, iters;
  c = 1;
  float kappa;
  image u, u_bar;
  unsigned char *image_chars;
  char *input_jpeg_filename, *output_jpeg_filename;
  /* read from command line: kappa, iters, input_jpeg_filename, output_jpeg_filename */
  /* ... */
  if (argc<5) {
    printf("Need four inputs from command line: \n");
    printf("./program.exe kappa iters input_jpeg_filename output_jpeg_filename.\n");
    exit(0);
  }
  else {
    double argument = atof(argv[1]);
    kappa = (float) argument;
    iters = atoi(argv[2]);
    input_jpeg_filename = argv[3];
    output_jpeg_filename = argv[4];
  }
  import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
  printf("Dimensions of image: (%d, %d)\n", m, n);
  printf("kappa: %f\n", kappa);
  printf("Number of iters: %d \n", iters); 


  allocate_image(&u, m, n);
  allocate_image(&u_bar, m, n);
  convert_jpeg_to_image(image_chars, &u);


  iso_diffusion_denoising(&u, &u_bar, kappa, iters);
  convert_image_to_jpeg(&u_bar, image_chars);

  export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);

  deallocate_image(&u);
  deallocate_image(&u_bar);


  return 0;
}
