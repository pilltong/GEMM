CC=nvcc
SRC=mma.cu
TARGET=main
FLAGS=-lcublas -arch=sm_89
INCLUDE_DIR=./headers

all : $(TARGET)

$(TARGET) : $(SRC) $(INCLUDE_DIR)/helpers.h $(INCLUDE_DIR)/cuda_kernels.cuh $(INCLUDE_DIR)/tensor_kernels.cuh
	$(CC) --ptxas-options=-v $(SRC) -o $(TARGET) $(FLAGS)

.PHONY : clean
clean :
	rm $(TARGET)