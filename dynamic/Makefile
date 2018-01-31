VERSION = -arch=sm_35 -gencode arch=compute_35,code=sm_35
CFLAGS = ${VERSION} -dc -rdc=true
LDFLAGS = ${VERSION}

TARGET = hello
SRC = hello.cu
OBJ = $(SRC:.cu=.o)
DEPS = 

all:${TARGET}
	echo "Done!"

${TARGET}:${OBJ}
	nvcc -o $@ $^ $(LDFLAGS)

%.o:%.cu ${DEPS}
	nvcc -c -o $@ $< $(CFLAGS) 

clean:
	rm -f ${OBJ} ${TARGET}
