# Mandatory assignment 2


## Serial implementation
For both implementations you need simple-jpeg folder in ../IN4200_Oblig2_593865/ to run makefiles.
Compile:
  cd -> ../serial_code/
  make
  ./main kappa iters filename1 filename2
  EXAMPLE:
  ./main 0.2 100 mona_lisa_noisy.jpg mona_lisa_glossy.jpg

## Parallel implementation
Compile:
  cd -> ../parallel_code/
  make
  mpirun -np P ./parallel_main kappa iters filename1 filename2
  EXAMPLE:
  mpirun -np 4 ./parallel_main 0.2 1000 mona_lisa_noisy.jpg mona_lisa_glossy.jpg

Alternatively:
 Compile as above.
 run the job.script on FOX.


main_cart.c is an attempt at using a 2D virtual mapping topology that failed.

Latest test:
I had issues running the parallel implementation just now. It runs but doesnt finish. I hope this only is
because of some problem with my pc, or something. I hope my cleanup hasn't removed any essentials to the parallelized code. It worked before cleanup, and now I have to deliver.

Thank you for your time.
