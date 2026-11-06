#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
static int fd; static size_t bs; static off_t fsz; static int per;
static void *w(void *a){ unsigned seed=(unsigned)(uintptr_t)a*7919u+1; void *buf; posix_memalign(&buf,4096,bs);
 for(int i=0;i<per;i++){ off_t o=((off_t)(rand_r(&seed)%(unsigned)((fsz-bs)/4096)))*4096; if(pread(fd,buf,bs,o)!=(ssize_t)bs){perror("pread");exit(1);} } free(buf); return 0; }
int main(int c,char**v){ fd=open(v[1],O_RDONLY|O_DIRECT); fsz=lseek(fd,0,SEEK_END); bs=atol(v[2]); int th=atoi(v[3]); per=atoi(v[4]);
 struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0); pthread_t t[64]; for(int i=0;i<th;i++) pthread_create(&t[i],0,w,(void*)(uintptr_t)i); for(int i=0;i<th;i++) pthread_join(t[i],0);
 clock_gettime(CLOCK_MONOTONIC,&t1); double s=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9; printf("bs=%7zuKB threads=%2d  %.2f GB/s  avg_lat=%.2fms\n",bs/1024,th,(double)bs*th*per/s/1e9,s*1e3/per); return 0; }
