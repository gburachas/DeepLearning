# You need to set the path to your OpenCL headers, or remove this and put them in a place
# where they will accessible
OPENCL=
CXXFLAGS =  -I../include -I${OPENCL} -O3 -g -Wall -Wno-reorder -fmessage-length=0 -std=c++11
OBJS = common.o deepbeliefnet.o  main.o  opencl_cli_mod.o rbm.o rbm.shared.units.o
LIBS = -lgflags -lcurl -lOpenCL -L../lib/ -lvjson
TARGET =    ../bin/DBN

%.o: %.cpp
	gcc -c $(CXXFLAGS) $*.cpp -o $*.o
	gcc -MM $(CXXFLAGS) $*.cpp > $*.d
	@cp -f $*.d $*.d.tmp
	@sed -e 's/.*://' -e 's/\\$$//' < $*.d.tmp | fmt -1 | \
	sed -e 's/^ *//' -e 's/$$/:/' >> $*.d
	@rm -f $*.d.tmp

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

kernels:
	../scripts/genKernels.py

clean:
	rm -f $(OBJS) *.d
