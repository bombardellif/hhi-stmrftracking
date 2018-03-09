#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>

#include "concurrentqueue.h"

static inline size_t inc(size_t a) {
    return (a+1) % QUEUE_CAPACITY;
}

bool concurrentqueue_push(ConcurrentQueue* self, int16_t (*data)[4], size_t size) {
    size_t curr_tail = atomic_load(&(self->tail));
    size_t next_tail = inc(curr_tail);
    // If queue is not full
    if (next_tail != atomic_load(&(self->head))) {
        self->buffer[curr_tail] = (QueueNode){.data = data, .size = size};
        // increment the tail
        atomic_store(&(self->tail), next_tail);
        return true;
    }
    return false;
}

bool concurrentqueue_pop(ConcurrentQueue* self, int16_t (**out)[4], size_t* size) {
    size_t curr_head = atomic_load(&(self->head));
    // if queue is not empty
    if (curr_head != atomic_load(&(self->tail))) {
        // Set output values
        *out = self->buffer[curr_head].data;
        *size = self->buffer[curr_head].size;
        // Increase the head towards the tail
        atomic_store(&(self->head), inc(curr_head));
        return true;
    }
    return false;
}

void concurrentqueue_free(ConcurrentQueue* self) {
    // Free operation is not thread safe. Exclusive access must be guaranteed
    for (size_t head = self->head, tail = self->tail;
    head != tail;
    head = inc(head)) {
        free(self->buffer[head].data);
    }
}
