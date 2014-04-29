# You need to set the path to your OpenCL headers, or remove this and put them in a place
# where they will accessible
OPENCL=
CXXFLAGS =  -I../include -I${OPENCL} -O3 -g -Wall -Wno-reorder -fmessage-length=0 -std=c++11
OBJS = common.o main.o  opencl_cli_mod.o  rbm.o  rbm.shared.units.o
LIBS = -L/usr/lib64/nvidia -lgflags -lcurl -lOpenCL -L../lib/ -lvjson
TARGET =    ../bin/RBM

%.o: %.cpp
	gcc -c $(CXXFLAGS) $*.cpp -o $*.o
	gcc -MM $(CXXFLAGS) $*.cpp > $*.d
	@cp -f $*.d $*.d.tmp
	@sed -e 's/.*://' -e 's/\\$$//' < $*.d.tmp | fmt -1 | \
	sed -e 's/^ *//' -e 's/$$/:/' >> $*.d
	@rm -f $*.d.tmp

$(TARGET): kernels $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

kernels:
	../scripts/genKernels.py

all:	$(TARGET)

clean:
	rm -f $(OBJS) *.d
