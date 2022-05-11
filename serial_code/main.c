#include <stdio.h>

#include <stdlib.h>

/* The purpose of this program is to demonstrate how the functions
   'import_JPEG_file' & 'export_JPEG_file' can be used. */

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

void allocate_image(image *u, int m, int n);
void deallocate_image(image *u);
void convert_jpeg_to_image(const unsigned char* image_chars, image *u);
void convert_image_to_jpeg(const image *u, unsigned char* image_chars);
void iso_diffusion_denoising(image *u, image *u_bar, float k, int iters);
void swap_images(image *u, image *u_bar, int m, int n);



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
    printf("Arg 1 as string: %s", argv[1]);
    double argument = atof(argv[1]);
    kappa = (float) argument;
    printf("Arg 1 as float: %f", kappa);
    iters = atoi(argv[2]);
    input_jpeg_filename = argv[3];
    output_jpeg_filename = argv[4];
  }
  import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
  printf("%d\n", m);
  printf("%d\n", n);
  printf("kappa: %f\n", kappa);


  allocate_image(&u, m, n);
  allocate_image(&u_bar, m, n);
  convert_jpeg_to_image(image_chars, &u);
  /*
  for (int i=0; i<m; i++) {
    printf("\n");
    for (int j=0; j<n; j++) {
      printf("%.1f ", u.image_data[i][j]);
    }
  }
  */
  printf("Kappa just before the function: %f\n", kappa);
  iso_diffusion_denoising(&u, &u_bar, kappa, iters);
  printf("Kappa after function: %f\n", kappa);
  convert_image_to_jpeg(&u_bar, image_chars);
  /*
  for (int i=0; i<m; i++) {
    printf("\n");
    for (int j=0; j<n; j++) {
      printf("%.1f ", u_bar.image_data[i][j]);
    }
  }
  */

  export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);

  deallocate_image(&u);
  deallocate_image(&u_bar);


  return 0;
}
