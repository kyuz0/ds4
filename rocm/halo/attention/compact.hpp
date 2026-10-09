#pragma once
#include "kv_access.hpp"
KVH_HD void public2048_convert_one(half*dst,const float*raw,const float*comp,
 uint64_t pair,uint32_t nraw,uint32_t cap,uint32_t start,uint32_t ncomp) {
 const uint64_t raw_pairs=(uint64_t)nraw*256;
 if(pair>=raw_pairs+(uint64_t)ncomp*256)return;
 const uint64_t source_pair=((start+pair/256)%cap)*256ull+pair%256;
 half2 h=pair<raw_pairs?kvh_convert_pair(raw,source_pair):kvh_convert_pair(comp,pair-raw_pairs);
 *reinterpret_cast<half2*>(dst+pair*2)=h;
}
