#ifndef __FAST_MVUTILS_H__
#define __FAST_MVUTILS_H__

#include <cstddef>
#include <cstdint>

double* c_fast_vectorized_pvm(int16_t (*vectors)[2], int shape[2]);

#endif
