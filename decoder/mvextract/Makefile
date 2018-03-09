CC      = gcc
CFLAGS  = -Wall -std=c11 -g
# LDFLAGS = -lswscale -lavdevice -lavformat -lavcodec -lswresample -lavutil -lpthread -lbz2 -lz -lc -lrt
LDFLAGS = -lavutil -lavformat -lavcodec -lswresample -lz -lm -lpthread
FFMPEG_INC = -I/home/bombardelli/dev/FFmpeg/ffmpeg_build/include
FFMPEG_LIBS = -L/home/bombardelli/dev/FFmpeg/ffmpeg_build/lib

SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)
all: $(OBJS)

test: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(FFMPEG_LIBS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(FFMPEG_INC) -c $<

clean:
	rm -rf *.o
