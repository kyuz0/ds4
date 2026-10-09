#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "../rocm/ds4_rocm_halo_shapes.h"
int main(void) {
    for (unsigned layer=0;layer<43;layer++) {
        assert(ds4_rocm_halo_scope_shape(2048,0,32897,layer,2048));
        assert(ds4_rocm_halo_scope_shape(4096,126976,131201,layer,4096));
    }
    assert(ds4_rocm_halo_uncached_a_shape(4096,126976,131201,4096,1024,8,128));
    assert(!ds4_rocm_halo_uncached_a_shape(4096,131072,131201,4096,1024,8,128));
    assert(!ds4_rocm_halo_uncached_a_shape(2048,0,65664,4096,1024,8,128));
    assert(!ds4_rocm_halo_uncached_a_shape(2048,1,65665,4096,1024,8,128));
    assert(!ds4_rocm_halo_scope_shape(0,0,0,0,0));
    assert(!ds4_rocm_halo_scope_shape(2048,1,65665,0,2048));
    assert(!ds4_rocm_halo_scope_shape(2048,0,2048,0,2048));
    assert(!ds4_rocm_halo_scope_shape(4096,131072,262144,0,4096));
    assert(!ds4_rocm_halo_scope_shape(2048,0,32897,43,2048));
    assert(!ds4_rocm_halo_scope_shape(2048,0,32897,0,4096));
    assert(ds4_rocm_halo_uncached_a_shape(2048,0,65665,4096,1024,8,128));
    assert(ds4_rocm_halo_uncached_a_shape(2048,2048,65665,4096,1024,8,128));
    assert(ds4_rocm_halo_uncached_a_shape(2048,4096,65665,4096,1024,8,128));
    assert(ds4_rocm_halo_uncached_a_shape(4096,0,65665,4096,1024,8,128));
    assert(ds4_rocm_halo_uncached_a_shape(2048,0,32897,4096,1024,8,128));

    for (uint32_t nc = 1024; nc <= 32768; nc += 512) {
        uint32_t pos = nc * 4 - 2048;
        assert(ds4_rocm_halo_indexer_shape(nc, 2048, pos, 64, 128, 4, 1));
        assert(!ds4_rocm_halo_indexer_shape(nc, 4096, pos, 64, 128, 4, 1));
        assert(!ds4_rocm_halo_indexer_shape(nc, 2048, pos+1, 64, 128, 4, 1));
        assert(!ds4_rocm_halo_indexer_shape(nc+1, 2048, pos, 64, 128, 4, 1));
    }
    assert(!ds4_rocm_halo_indexer_shape(512, 2048, 0, 64, 128, 4, 1));
    assert(!ds4_rocm_halo_indexer_shape(33280, 2048, 130048, 64, 128, 4, 1));
    assert(!ds4_rocm_halo_indexer_shape(1024, 1, 4095, 64, 128, 4, 1));
    assert(!ds4_rocm_halo_indexer_shape(1024, 2048, UINT32_MAX-2047, 64, 128, 4, 1));
    assert(!ds4_rocm_halo_indexer_shape(1024, 2048, 2048, 32, 128, 4, 1));
    assert(!ds4_rocm_halo_indexer_shape(1024, 2048, 2048, 64, 64, 4, 1));
    assert(!ds4_rocm_halo_indexer_shape(1024, 2048, 2048, 64, 128, 0, 1));
    assert(!ds4_rocm_halo_indexer_shape(1024, 2048, 2048, 64, 128, 4, 0));
    assert(ds4_rocm_halo_q8_shape(2048, 1024, 32768));
    assert(ds4_rocm_halo_q8_shape(4096, 1024, 32768));
    assert(ds4_rocm_halo_q8_shape(2048, 4096, 2048));
    assert(ds4_rocm_halo_q8_shape(4096, 4096, 2048));
    assert(ds4_rocm_halo_qa_shape(4096, 4096, 1024));
    assert(!ds4_rocm_halo_qa_shape(2048, 4096, 1024));
    const uint64_t tails[] = {0, 1, 127, 512, 2047, 2049, 4095, 4097, UINT32_MAX, UINT64_MAX};
    for (unsigned i = 0; i < sizeof(tails)/sizeof(tails[0]); ++i) {
        assert(!ds4_rocm_halo_q8_shape(tails[i], 1024, 32768));
        assert(!ds4_rocm_halo_qa_shape(tails[i], 4096, 1024));
    }
    assert(!ds4_rocm_halo_q8_shape(4096, 4096, 1024));
    assert(!ds4_rocm_halo_q8_shape(4096, 2048, 4096));
    assert(!ds4_rocm_halo_q8_shape(4096, UINT64_MAX, 32768));
    assert(ds4_rocm_halo_static_query_shape(2048, 0, 128, 1, 64, 512));
    assert(ds4_rocm_halo_static_query_shape(2048, 16, 128, 128, 64, 512));
    assert(ds4_rocm_halo_static_query_shape(2048, 512, 128, 4, 64, 512));
    assert(ds4_rocm_halo_static_query_shape(4096, 0, 128, 1, 64, 512));
    assert(ds4_rocm_halo_static_query_shape(4096, 32, 128, 128, 64, 512));
    assert(!ds4_rocm_halo_static_query_shape(4096, 1024, 128, 4, 64, 512));
    assert(!ds4_rocm_halo_static_query_shape(2048, 16, 128, 128, 32, 512));
    assert(!ds4_rocm_halo_static_query_shape(2049, 16, 128, 128, 64, 512));
    assert(!ds4_rocm_halo_static_query_shape(1, 0, 128, 1, 64, 512));
    assert(ds4_rocm_halo_hc_shape(2048, 16384, 24));
    assert(ds4_rocm_halo_hc_shape(4096, 16384, 24));
    assert(!ds4_rocm_halo_hc_shape(2047, 16384, 24));
    assert(!ds4_rocm_halo_hc_shape(1, 16384, 24));
    assert(!ds4_rocm_halo_hc_shape(4096, 4096, 24));
    assert(!ds4_rocm_halo_hc_shape(4096, 16384, 25));
    for (uint32_t rows=2048; rows<=4096; rows*=2) {
        for (uint32_t pos=rows; pos+rows<=131072; pos+=rows) {
            uint32_t end=pos+rows, nr=rows+128, cap=end+129, start=end-nr;
            assert(ds4_rocm_halo_direct_qk_shape(rows,pos,nr,cap,start,end/4,512,128,4,64,512,1));
            assert(ds4_rocm_halo_direct_qk_shape(rows,pos,nr,cap,start,end/128,0,128,128,64,512,0));
            assert(!ds4_rocm_halo_direct_qk_shape(rows,pos,nr,cap,start+1,end/4,512,128,4,64,512,1));
            assert(!ds4_rocm_halo_direct_qk_shape(rows,pos,nr,cap,start,end/4,256,128,4,64,512,1));
        }
    }
    assert(!ds4_rocm_halo_direct_qk_shape(1,2048,129,4096,1920,512,512,128,4,64,512,1));
    assert(ds4_rocm_halo_output_a_shape(2048,4096,1024,8));
    assert(ds4_rocm_halo_output_a_shape(4096,4096,1024,8));
    assert(!ds4_rocm_halo_output_a_shape(2047,4096,1024,8));
    assert(!ds4_rocm_halo_output_a_shape(4096,4096,1024,7));
    assert(ds4_rocm_halo_shared_down_shape(2048,2048,4096));
    assert(ds4_rocm_halo_shared_down_shape(4096,2048,4096));
    assert(!ds4_rocm_halo_shared_down_shape(1,2048,4096));
    assert(!ds4_rocm_halo_shared_down_shape(4096,4096,2048));
    puts("Halo production admission: PASS (CPU only)");
    return 0;
}
