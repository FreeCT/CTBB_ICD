CC=g++
CC_GPU=nvcc

ODIR=obj

IDIR_RECON=../include/
INCLUDES=-I$(IDIR_RECON)

LOCAL_INCLUDES=recon_structs.h

LIBS=-lm -lyaml-cpp -lctbb_read -lcudart

CFLAGS=-g -DHAVE_CONFIG_H -std=gnu++11 -I/usr/local/cuda/include
NVCCFLAGS=-g -DHAVE_CONFIG_H -std=c++11
CUDART = -L/usr/local/cuda-6.5/targets/x86_64-linux/lib/

_OBJ=main.o parse_config.o setup.o generate_system_matrix.o initialize_recon_volume.o icd_iteration.o rotate_slices.o penalties.o icd_iteration_gpu.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ));

all: ../icd

../icd: object_dir $(OBJ) 
	(cd obj && $(CC) -g -fopenmp $(CUDART) -o ../../icd *.o $(LIBS))

object_dir:
	mkdir -p obj

obj/main.o: main.cpp recon_structs.h	
	$(CC) $(CFLAGS) -c -o $@ $<

obj/parse_config.o: parse_config.cpp parse_config.h recon_structs.h
	$(CC) $(CFLAGS) -c -o $@ $<

obj/setup.o: setup.cpp setup.h recon_structs.h parse_config.h
	$(CC) $(CFLAGS) -c -o $@ $<

obj/generate_system_matrix.o: generate_system_matrix.cpp generate_system_matrix.h recon_structs.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -fopenmp -o $@ $<

obj/initialize_recon_volume.o: initialize_recon_volume.cpp initialize_recon_volume.h recon_structs.h
	$(CC) $(CFLAGS) -c -o $@ $<

obj/icd_iteration.o: icd_iteration.cpp icd_iteration.h recon_structs.h
	$(CC) $(CFLAGS) -c -fopenmp -o $@ $<

obj/icd_iteration_gpu.o: icd_iteration_gpu.cu icd_iteration.h recon_structs.h
	$(CC_GPU) $(NVCCFLAGS) -c -o $@ $< 

obj/rotate_slices.o: rotate_slices.cpp rotate_slices.h recon_structs.h interp.h
	$(CC) $(CFLAGS) -c -o $@ $<

obj/penalties.o: penalties.cpp penalties.h
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: all test clean object_dir

clean:
	rm obj/*.o; rm -rf ../*~; rm -f ../icd

