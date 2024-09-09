CC=nvcc
SRC=cuda_kernel.cu
TARGET=main
FLAGS=-lcublas -arch=sm_89
INCLUDE_DIR=../headers

all : $(TARGET)

$(TARGET) : $(SRC) $(INCLUDE_DIR)/helpers.h $(INCLUDE_DIR)/cuda_kernels.cuh ./runner.cuh
	$(CC) $(SRC) -o $(TARGET) $(FLAGS)

.PHONY : clean
clean :
	rm $(TARGET)