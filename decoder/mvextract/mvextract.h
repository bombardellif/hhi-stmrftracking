#ifndef __MVEXTRACT_H__
#define __MVEXTRACT_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "streambuffer.h"

bool init(void);

StreamBuffer* read_videostream(const char url[]);

void stop(StreamBuffer* buff);

bool get_next_frame(StreamBuffer* buff, int16_t (**out)[4], size_t* size);

void destroy(StreamBuffer* buff);

#endif
