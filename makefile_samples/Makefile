CC = /usr/local/cuda-8.0/bin/nvcc

SRC = simulation.cu io.cu visualisation.cu main.cu #FLAMEGPU_kernals.cu functions.cu visualisation.cu

FLAGS = -g -G -gencode arch=compute_35,code=compute_35 --relocatable-device-code=true #--nvlink-options --verbose

LIBS = -lcudart -lglut -lGL -lGLU -lm -lcuda -lGLEW -lcudpp -lcudadevrt -lcufft

LIBDIRS = -L/home/juanjo/FLAMEGPU/cudpp-2.2-build/lib -L/usr/local/cuda-8.0/lib64 -L/home/juanjo/FLAMEGPU/glew/glew-2.1.0/lib

INC = -I/usr/local/cuda-8.0/targets/x86_64-linux/include -I/home/juanjo/FLAMEGPU/FlameGPU-1.4/FLAMEGPU-1.4.1/include -I/home/juanjo/FLAMEGPU/glew/glew-2.1.0/include/GL

#DEF = -D_CRT_SECURE_NO_DEPRECATE -D_MBCS -DVISUALISATION
DEF = -D_CRT_SECURE_NO_DEPRECATE -D_MBCS

all:
	$(CC) $(FLAGS) $(LIBDIRS) $(DEF) $(INC) $(LIBS) $(SRC) -o model

