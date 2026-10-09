// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include "iq2_feed.hpp"
namespace iq2_rf {
struct PairDesc {unsigned expert,first0,first1,zero;};
constexpr unsigned kPairCap=192,kSingleCap=768;
extern "C" __global__ void halo_s9_fresh_descriptors(const unsigned* pairs,const unsigned* offsets,
 Desc* singles,unsigned* ns,PairDesc* paired,unsigned* np){
 if(threadIdx.x)return;
 unsigned os=0,op=0;
 for(unsigned e=0;e<256;e++){
  unsigned first=offsets[e],end=offsets[e+1],lo=first,hi=end;
  while(lo<hi){unsigned m=(lo+hi)/2;if(pairs[m]<12288)lo=m+1;else hi=m;}
  unsigned pending=0;bool have=false;
  for(unsigned part=0;part<2;part++){
   unsigned a=part?lo:first,b=part?end:lo;
   for(unsigned j=a;j<b;j+=64){
    unsigned count=b-j<64u?b-j:64u;
    if(count==64){
     if(!have){pending=j;have=true;}
     else{
      if(op>=kPairCap){__builtin_trap();return;}
      paired[op++]={e,pending,j,0};have=false;
     }
    }else{
     if(os>=kSingleCap){__builtin_trap();return;}
     singles[os++]={e,j,count,(j-a)&63u};
    }
   }
  }
  if(have){
   if(os>=kSingleCap){__builtin_trap();return;}
   singles[os++]={e,pending,64,0};
  }
 }
 *ns=os;*np=op;
}

}
