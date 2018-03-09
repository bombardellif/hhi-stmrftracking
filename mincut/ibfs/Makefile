CC      = g++
CFLAGS  = -O3 -Wall -std=c++11 -g
# LDFLAGS = -lm `pkg-config opencv --cflags --libs`
# LDFLAGS = -lm -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
LDFLAGS = -static-libgcc -static-libstdc++

OBJ = main.o ibfs.o
BIN = ibfs

ibfs: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rf $(BIN) $(OBJ)
