CC = mpicc
CFLAGS = -O2
LDFLAGS = -L../simple-jpeg/ -lsimplejpeg

PROJ = main
OBJS = main.o
FUNC = functions.o

all: simple-jpeg $(PROJ)

functions: $(FUNC)
	$(CC) $(CFLAGS) $< -o
main: $(OBJS)
	$(CC) $(CFLAGS) $< functions.o -o $@ $(LDFLAGS)

simple-jpeg:
	$(MAKE) -C ../simple-jpeg

obj-clean:
	$(RM) *.o

exec-clean:
	$(RM) $(PROJ)

autosave-clean:
	$(RM) *~

clean:
	$(MAKE) obj-clean
	$(MAKE) exec-clean
	$(MAKE) autosave-clean
	$(MAKE) -C ../simple-jpeg clean
