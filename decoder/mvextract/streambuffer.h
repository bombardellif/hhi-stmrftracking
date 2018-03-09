#ifndef __STREAMBUFFER_H__
#define __STREAMBUFFER_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#include "concurrentqueue.h"

typedef struct StreamBuffer {
    char *url;
    ConcurrentQueue queue;
    pthread_t th_producer;
    pthread_mutex_t mutex_can_consume;
    pthread_cond_t cond_can_consume;
    pthread_mutex_t mutex_can_produce;
    pthread_cond_t cond_can_produce;
    _Atomic bool active;
    _Atomic bool eof;
} StreamBuffer;

StreamBuffer* streambuffer_new(const char url[]);
void streambuffer_set_eof(StreamBuffer* self);
void streambuffer_unset_active(StreamBuffer* self);
bool streambuffer_push(StreamBuffer* self, int16_t (*data)[4], size_t size);
bool streambuffer_pop(StreamBuffer* self, int16_t (**out)[4], size_t* size);
void streambuffer_free(StreamBuffer* self);

#endif
