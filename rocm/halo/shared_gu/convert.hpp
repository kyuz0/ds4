#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#define DB_HD __host__ __device__ __forceinline__
#else
#define DB_HD inline
#endif
namespace dense_bridge {
DB_HD _Float16 weight_half(const unsigned char* w,uint64_t i){
 const unsigned char*b=w+(i/32)*34;_Float16 s;__builtin_memcpy(&s,b,2);
 return (_Float16)((float)s*(float)((const int8_t*)(b+2))[i%32]);
}
DB_HD _Float16 activation_half(float x){return (_Float16)x;}
}
#undef DB_HD
