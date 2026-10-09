#pragma once
#include <cstdint>
namespace oa {
constexpr unsigned slot(unsigned wave,unsigned q){return wave+8*q;}
constexpr unsigned block(unsigned wave,unsigned q,unsigned step){return slot(wave,q)+32*step;}
constexpr unsigned element(unsigned lane,unsigned component){return (lane&15)*16+2*component+(lane>>4);}
// XOR rank by token: both WMMA-fragment stores and linear-output loads
// visit32 distinct dword banks, without padding the32KiB partial buffer.
constexpr unsigned scratch(unsigned element){unsigned t=element/16,r=element%16;return t*16+(r^t);}
constexpr uint64_t xrow(unsigned token,unsigned group){return uint64_t(token)*8+group;}
constexpr uint64_t output(unsigned token,unsigned group,unsigned rank){return uint64_t(token)*8192+group*1024+rank;}
}
