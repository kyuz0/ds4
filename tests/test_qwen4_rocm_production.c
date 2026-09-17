/* Full-output production-shape expert oracle. Hashed row prototypes and
 * repeated inputs keep every output independently checkable without a huge
 * CPU GEMM; the row map avoids a periodic pattern aligned with WMMA tiles. */
#define main qwen_small_suite_main
#include "test_qwen4_kernels.c"
#undef main

enum { PE=2560, PF=640, PNE=512, PNS=10, PNO=11, PR=8, PV=17, PG=32 };
static unsigned production_row_key(unsigned row) {
    uint32_t x = (row + 1u) * UINT32_C(0x9e3779b9);
    x ^= x >> 16;
    x *= UINT32_C(0x85ebca6b);
    return (x ^ (x >> 13)) % PR;
}
static uint64_t production_weights(arena_t *a, unsigned type, unsigned rows,
                                   unsigned cols, double **shadow) {
    uint64_t proto=arena_tier(a,type,PNE*PR,cols,shadow);
    uint64_t rb=type==16 ? (uint64_t)(cols/256)*66 : type==10 ? (uint64_t)(cols/256)*84 :
                type==12 ? (uint64_t)(cols/256)*144 : (uint64_t)(cols/32)*17;
    uint64_t off=arena_alloc(a,(uint64_t)PNE*rows*rb);
    for (unsigned e=0;e<PNE;e++) for (unsigned r=0;r<rows;r++)
        memcpy(a->base+off+((uint64_t)e*rows+r)*rb,
               a->base+proto+((uint64_t)e*PR+production_row_key(r))*rb,rb);
    return off;
}
static float input_value(unsigned v,unsigned k) {
    return (float)((int)((v*23+k*17+k/7)%127)-63)/512.0f;
}
static void check_production(const char *name,const float *out,const double *reference,
                             const int32_t *selection,unsigned T,unsigned width) {
    double worst=0,scale=1e-6;uint64_t tested=0;
    for (unsigned t=0;t<T;t++) for (unsigned s=0;s<PNO;s++) for (unsigned r=0;r<width;r++) {
        float got=out[((uint64_t)t*PNO+s)*width+r];
        if (s==PNS) { require_ok(got==-1234.5f,"reserved output slot");continue; }
        unsigned e=selection[t*PNS+s],v=width==PF ? t%PV : (t*PNS+s)%PV;
        double ref=reference[((uint64_t)e*PV+v)*PR+production_row_key(r)];
        require_ok(isfinite(got),"finite full output");
        worst=fmax(worst,fabs(got-ref));scale=fmax(scale,fabs(ref));tested++;
    }
    for(unsigned g=0;g<PG;g++) require_ok(out[(uint64_t)T*PNO*width+g]==-1234.5f,"production output guard");
    printf("%s: %llu outputs, max_abs=%.9g scale=%.9g rel=%.9g\n",name,(unsigned long long)tested,worst,scale,worst/scale);
    require_ok(worst<=3e-5*scale,"unchanged expert tolerance");
}
int main(int argc,char **argv) {
    require_ok(argc==3,"usage: ds4-kernel-qwen-production Q2|Q4 T");
    unsigned type=!strcmp(argv[1],"Q2")?16:12,dtype=type==16?10:39;
    unsigned T=(unsigned)strtoul(argv[2],NULL,10),DF=dtype==10?768:PF;
    require_ok((!strcmp(argv[1],"Q2")||!strcmp(argv[1],"Q4")) && T>0 && T<=8193,"arguments");
    arena_t a={.size=UINT64_C(4)<<30};a.base=mmap(NULL,a.size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    require_ok(a.base!=MAP_FAILED,"production arena");
    double *gw,*uw,*dw;
    uint64_t go=production_weights(&a,type,PF,PE,&gw),uo=production_weights(&a,type,PF,PE,&uw),d=production_weights(&a,dtype,PE,DF,&dw);
    double *midref=calloc((uint64_t)PNE*PV*PR,sizeof(double)),*downref=calloc((uint64_t)PNE*PV*PR,sizeof(double));
    require_ok(midref && downref,"reference allocation");
    for(unsigned e=0;e<PNE;e++) for(unsigned v=0;v<PV;v++) for(unsigned r=0;r<PR;r++) {
        double g=0,u=0,z=0;
        for(unsigned k=0;k<PE;k++) { double x=(_Float16)input_value(v,k);
            g+=(double)(_Float16)(float)gw[((uint64_t)e*PR+r)*PE+k]*x;
            u+=(double)(_Float16)(float)uw[((uint64_t)e*PR+r)*PE+k]*x; }
        for(unsigned k=0;k<PF;k++) z+=(double)(_Float16)(float)dw[((uint64_t)e*PR+r)*DF+k]*(double)(_Float16)input_value(v,k);
        midref[((uint64_t)e*PV+v)*PR+r]=silu_d(g)*u;downref[((uint64_t)e*PV+v)*PR+r]=z;
    }
    free(gw);free(uw);free(dw);
    int32_t *sel=malloc((uint64_t)T*PNS*4),counts[PNE]={0};
    float *x=malloc((uint64_t)T*PE*4);require_ok(sel&&x,"inputs");
    for(unsigned t=0;t<T;t++) { for(unsigned k=0;k<PE;k++) x[(uint64_t)t*PE+k]=input_value(t%PV,k);
        for(unsigned s=0;s<PNS;s++){ unsigned e=s==0?0:s==PNS-1?511:1+(t*13+s*47)%509;sel[t*PNS+s]=e;counts[e]++; } }
    require_ok(ds4_gpu_init()&&ds4_gpu_set_model_map(a.base,a.size),"GPU/map initialization");
    ds4_gpu_tensor *gx=upload(x,(uint64_t)T*PE),*gs=ds4_gpu_tensor_alloc((uint64_t)T*PNS*4),*gl=ds4_gpu_tensor_alloc((uint64_t)PNE*(T+7)*4),*gc=ds4_gpu_tensor_alloc(PNE*4);
    uint64_t nm=(uint64_t)T*PNO*PF,np=(uint64_t)T*PNO*PE;
    ds4_gpu_tensor *gm=upload(NULL,nm+PG),*gp=upload(NULL,np+PG);
    require_ok(gs&&gl&&gc&&ds4_gpu_tensor_write(gs,0,sel,(uint64_t)T*PNS*4),"selection upload");
    require_ok(ds4_gpu_qwen4_moe_build_lists_tensor(gl,gc,gs,T,PNS,PNE,T+7),"production lists");
    int32_t actual[PNE];require_ok(ds4_gpu_tensor_read(gc,0,actual,sizeof(actual))&&!memcmp(actual,counts,sizeof(counts)),"all expert counts");
    require_ok(counts[0]==(int)T&&counts[511]==(int)T&&counts[510]==0,"hot/empty experts");
    require_ok(ds4_gpu_tensor_fill_f32(gm,-1234.5f,nm+PG),"mid guards");
    require_ok(ds4_gpu_qwen4_moe_mm_mid_tensor(gm,gx,gl,gc,a.base,a.size,go,uo,type,PNE,T,PNS,PNO,PE,PF,T+7)&&ds4_gpu_synchronize(),"production mid");
    float *mid=download(gm,nm+PG);check_production("gate/up",mid,midref,sel,T,PF);
    /* Independently controlled down inputs prevent upstream error cancellation. */
    for(unsigned t=0;t<T;t++) for(unsigned s=0;s<PNO;s++) for(unsigned k=0;k<PF;k++) mid[((uint64_t)t*PNO+s)*PF+k]=input_value((t*PNS+s)%PV,k);
    require_ok(ds4_gpu_tensor_write(gm,0,mid,nm*4)&&ds4_gpu_tensor_fill_f32(gp,-1234.5f,np+PG),"down inputs and guards");
    require_ok(ds4_gpu_qwen4_moe_mm_down_tensor(gp,gm,gl,gc,a.base,a.size,d,dtype,PNE,T,PNS,PNO,PF,PE,T+7)&&ds4_gpu_synchronize(),"production down");
    float *part=download(gp,np+PG);check_production("down",part,downref,sel,T,PE);
    free(part);free(mid);free(x);free(sel);free(midref);free(downref);
    ds4_gpu_tensor_free(gx);ds4_gpu_tensor_free(gs);ds4_gpu_tensor_free(gl);ds4_gpu_tensor_free(gc);ds4_gpu_tensor_free(gm);ds4_gpu_tensor_free(gp);
    ds4_gpu_cleanup();munmap(a.base,a.size);
    printf("PASS %s E=2560 F=640 experts=512 slots=10 T=%u down_stride=%u\n",argv[1],T,DF);
    return 0;
}
