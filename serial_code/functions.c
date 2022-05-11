#include <stdlib.h>
#include <stdio.h>

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

void allocate_image(image *u, int m, int n) {
  (*u).image_data = (float**)malloc(m*sizeof(float*));
  for (int i=0; i<m; i++) {
    (*u).image_data[i] = (float*)malloc(n*sizeof(float));
  }
  (*u).m = m;
  (*u).n = n;

}

void deallocate_image(image *u) {
  for (int i=0; i<(*u).m; i++) {
    free((*u).image_data[i]);
  }
  free((*u).image_data);
}


void convert_jpeg_to_image(const unsigned char* image_chars, image *u) {
  for (int i=0; i<(*u).m; i++) {
    for (int j=0; j<(*u).n; j++) {
      (*u).image_data[i][j] = image_chars[i*(*u).n + j];
    }
  }
}

void convert_image_to_jpeg(const image *u, unsigned char* image_chars) {
  for (int i=0; i<(*u).m; i++) {
    for (int j=0; j<(*u).n; j++) {
      image_chars[i*(*u).n + j] = (unsigned char) (*u).image_data[i][j];
    }
  }
}

void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters) {
  int i, j;
  int m = (*u).m;
  int n = (*u).n;
  float cross_sum, four_times_val, difference;
  for (j=0; j<n; j++) {
    (*u_bar).image_data[0][j] = (*u).image_data[0][j];
    (*u_bar).image_data[m-1][j] = (*u).image_data[m-1][j];
  }
  for (i=0; i<m; i++) {
    (*u_bar).image_data[i][n-1] = (*u).image_data[i][n-1];
    (*u_bar).image_data[i][0] = (*u).image_data[i][0];
  }
  for (int iteration=0; iteration<iters; iteration++){
    for (i=1; i<m-1; i++) {
      for (j=1; j<n-1; j++) {
        cross_sum = (*u).image_data[i-1][j] + (*u).image_data[i][j-1] + (*u).image_data[i+1][j] + (*u).image_data[i][j+1];
        four_times_val = 4*(*u).image_data[i][j];
        difference = cross_sum - four_times_val;

        (*u_bar).image_data[i][j] = (*u).image_data[i][j] + kappa*difference;

      }
    }
    if (iteration<(iters-1)) {
      swap_images(u, u_bar, m, n);
    }
  }
}

void swap_images(image *u, image *u_bar, int m, int n) {
  float temp;
  for (int i=1; i<m-1; i++) {
    for (int j=1; j<n-1; j++) {
      temp = (*u).image_data[i][j];
      (*u).image_data[i][j] = (*u_bar).image_data[i][j];
      (*u_bar).image_data[i][j] = temp;
    }
  }
}
