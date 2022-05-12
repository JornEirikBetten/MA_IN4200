#include <stdlib.h>
#include <stdio.h>
#include "functions.h"


void allocate_image(image *u, int m, int n) {
  u->image_data = malloc(m*sizeof(u->image_data));
  u->image_data[0] = malloc(m*n*sizeof(u->image_data));
  for (int i=0; i<m; i++) {
    u->image_data[i] = &(u->image_data[0][n*i]);
  }
  u->m = m;
  u->n = n;

}

void deallocate_image(image *u){
  free(u->image_data[0]);
  free(u->image_data);

}


void convert_jpeg_to_image(const unsigned char* image_chars, image *u) {
  int m = u->m;
  int n = u->n;
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      u->image_data[i][j] = (float)image_chars[i*n + j];
    }
  }
}

void convert_image_to_jpeg(const image *u, unsigned char* image_chars) {
  int m = u->m;
  int n = u->n;
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      image_chars[i*n + j] = (unsigned char) u->image_data[i][j];
    }
  }
}

void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters) {
  int i, j;
  int m = u->m;
  int n = u->n;
  float cross_sum, four_times_val, difference;
  for (j=0; j<n; j++) {
    u_bar->image_data[0][j] = u->image_data[0][j];
    u_bar->image_data[m-1][j] = u->image_data[m-1][j];
  }
  for (i=0; i<m; i++) {
    u_bar->image_data[i][n-1] = u->image_data[i][n-1];
    u_bar->image_data[i][0] = u->image_data[i][0];
  }
  float **temp;
  /* Executing diffusion iters times */
  for (int iteration=0; iteration<iters; iteration++){
    for (i=1; i<m-1; i++) {
      for (j=1; j<n-1; j++) {
        cross_sum = u->image_data[i-1][j] + u->image_data[i][j-1] + u->image_data[i+1][j] + u->image_data[i][j+1];
        four_times_val = 4*u->image_data[i][j];
        difference = cross_sum - four_times_val;

        u_bar->image_data[i][j] = u->image_data[i][j] + kappa*difference;

      }
    }
    /* Swapping u and u_bar between each
       exectution of the diffusion except last.*/
    if (iteration<(iters-1)) {
      temp = u_bar->image_data;
      u_bar->image_data = u->image_data;
      u->image_data = temp;
    }
  }
}
