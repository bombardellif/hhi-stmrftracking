#ifndef __CONCURRENTQUEUE_H__
#define __CONCURRENTQUEUE_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define QUEUE_CAPACITY 256

typedef struct QueueNode {
    int16_t (*data)[4];
    size_t size;
} QueueNode;

typedef struct ConcurrentQueue {
    _Atomic size_t head;
    _Atomic size_t tail;
    QueueNode buffer[QUEUE_CAPACITY];
} ConcurrentQueue;

bool concurrentqueue_push(ConcurrentQueue* self, int16_t (*data)[4], size_t size);
bool concurrentqueue_pop(ConcurrentQueue* self, int16_t (**out)[4], size_t* size);
void concurrentqueue_free(ConcurrentQueue* self);

#endif
