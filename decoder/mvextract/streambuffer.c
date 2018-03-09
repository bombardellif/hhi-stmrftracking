#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <pthread.h>

#include "concurrentqueue.h"
#include "streambuffer.h"

StreamBuffer* streambuffer_new(const char url[]) {
    // Allocate memory initialing it with zeros
    StreamBuffer *self = (StreamBuffer*)calloc(1, sizeof(StreamBuffer));
    if (self) {
        size_t url_len = strlen(url) + 1;
        self->url = (char*)malloc(url_len * sizeof(char));
        if (!self->url)
            goto except;
        strncpy(self->url, url, url_len);
        self->active = true;
        if (pthread_mutex_init(&(self->mutex_can_consume), NULL))
            goto except;
        if (pthread_cond_init(&(self->cond_can_consume), NULL))
            goto except;
        if (pthread_mutex_init(&(self->mutex_can_produce), NULL))
            goto except;
        if (pthread_cond_init(&(self->cond_can_produce), NULL))
            goto except;
    }
    return self;
except:
    free(self->url);
    pthread_mutex_destroy(&(self->mutex_can_consume));
    pthread_cond_destroy(&(self->cond_can_consume));
    pthread_mutex_destroy(&(self->mutex_can_produce));
    pthread_cond_destroy(&(self->cond_can_produce));
    free(self);
    return NULL;
}

void streambuffer_set_eof(StreamBuffer* self) {
    atomic_store(&(self->eof), true);
    // Signal consumer thread to avoid blocking forever
    pthread_mutex_lock(&(self->mutex_can_consume));
    pthread_cond_signal(&(self->cond_can_consume));
    pthread_mutex_unlock(&(self->mutex_can_consume));
}

void streambuffer_unset_active(StreamBuffer* self) {
    atomic_store(&(self->active), false);
    // Signal producer thread to avoid blocking forever
    pthread_mutex_lock(&(self->mutex_can_produce));
    pthread_cond_signal(&(self->cond_can_produce));
    pthread_mutex_unlock(&(self->mutex_can_produce));
    // Signal consumer thread to avoid blocking forever
    pthread_mutex_lock(&(self->mutex_can_consume));
    pthread_cond_signal(&(self->cond_can_consume));
    pthread_mutex_unlock(&(self->mutex_can_consume));
}

bool streambuffer_push(StreamBuffer* self, int16_t (*data)[4], size_t size) {
    bool success = false;
    while (!success && atomic_load(&(self->active))) {
        success = concurrentqueue_push(&(self->queue), data, size);
        // If buffer is full, block and wait for a pop by other thread
        if (!success) {
            pthread_mutex_lock(&(self->mutex_can_produce));
            pthread_cond_wait(&(self->cond_can_produce),
                              &(self->mutex_can_produce));
            pthread_mutex_unlock(&(self->mutex_can_produce));
        } else {
            // Signalizes new data in the buffer
            pthread_mutex_lock(&(self->mutex_can_consume));
            pthread_cond_signal(&(self->cond_can_consume));
            pthread_mutex_unlock(&(self->mutex_can_consume));
        }
    }
    return success;
}

bool streambuffer_pop(StreamBuffer* self, int16_t (**out)[4], size_t* size) {
    bool success = false;
    while (!success && atomic_load(&(self->active))) {
        success = concurrentqueue_pop(&(self->queue), out, size);
        // if queue is empty and not end of file, block and wait for data
        if (!success) {
            if (atomic_load(&(self->eof)))
                break;
            pthread_mutex_lock(&(self->mutex_can_consume));
            pthread_cond_wait(&(self->cond_can_consume),
                              &(self->mutex_can_consume));
            pthread_mutex_unlock(&(self->mutex_can_consume));
        } else {
            // Signalizes available space in the buffer
            pthread_mutex_lock(&(self->mutex_can_produce));
            pthread_cond_signal(&(self->cond_can_produce));
            pthread_mutex_unlock(&(self->mutex_can_produce));
        }
    }
    return success;
}

void streambuffer_free(StreamBuffer* self) {
    // Free operation is not thread safe. Exclusive access must be guaranteed
    if (self) {
        concurrentqueue_free(&(self->queue));
        free(self->url);
        pthread_mutex_destroy(&(self->mutex_can_consume));
        pthread_cond_destroy(&(self->cond_can_consume));
        pthread_mutex_destroy(&(self->mutex_can_produce));
        pthread_cond_destroy(&(self->cond_can_produce));
        free(self);
    }
}
