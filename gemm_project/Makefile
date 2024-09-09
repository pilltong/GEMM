CC=nvcc
SRC=mma.cu
TARGET=main
FLAGS=-lcublas -arch=sm_89

all : $(TARGET)

$(TARGET) : $(SRC) helper.h
	$(CC) --ptxas-options=-v $(SRC) -o $(TARGET) $(FLAGS)

.PHONY : clean
clean :
	rm $(TARGET)