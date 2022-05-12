#ifndef PARALLEL_F_H
#define PARALLEL_F_H


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
void ghostrow_update(image *u);

void iso_diffusion_denoising_parallel(image *u, image *u_bar, float kappa, int iters);

#endif
