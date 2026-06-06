CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
# Target AMD Ryzen AI Max "Strix Halo" APUs (Zen 5 architecture)
NATIVE_CPU_FLAG ?= -march=znver5
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
DS4_C_INCS := ds4_gpu_env.inc ds4_gpu_startup.inc ds4_token_embedding.inc ds4_ngram_spec.inc

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas

HIPCC ?= $(shell command -v hipcc 2>/dev/null)
HIPCXXFLAGS ?= -O3 --offload-arch=native -std=c++17 -Wno-unused-parameter -Wno-unused-function
ROCM_PATH ?= /opt/rocm
ROCM_ARCH ?= gfx1151
ROCM_HIPCC ?= $(if $(HIPCC),$(HIPCC),$(ROCM_PATH)/bin/hipcc)
ROCM_CFLAGS ?= -O3 -ffast-math -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -L$(ROCM_PATH)/lib -lhipblas -lhipblaslt

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_distributed.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CORE_OBJS = ds4.o ds4_distributed.o ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o
METAL_LDLIBS := $(LDLIBS)
endif

NODE ?= node
PYTHON ?= python3
ROCM_API_SMOKE ?= /tmp/rocm_api_smoke

.PHONY: all help clean test pi-stateful-test cpu cuda cuda-spark cuda-generic cuda-regression rocm rocm-upstream check-rocm-exports rocm-api-smoke rocm-api-smoke-build

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make rocm         Build ROCm upstream-shaped binaries"
	@echo "  make check-rocm-exports  Check ds4_gpu.h vs ROCm exports"
	@echo "  make rocm-api-smoke      Build and run lightweight ROCm API smoke"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

DS4_LINK = $(CC) $(CFLAGS)
DS4_LINK_LIBS = $(METAL_LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make rocm                Build ROCm upstream-shaped binaries"
	@echo "  make rocm-upstream       Build ROCm upstream-shaped binaries"
	@echo "  make check-rocm-exports  Check ds4_gpu.h vs ROCm exports"
	@echo "  make rocm-api-smoke      Build and run lightweight ROCm API smoke"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make pi-stateful-test    Run Pi DS4 stateful provider regression tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

DS4_LINK = $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS = $(CUDA_LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

rocm rocm-upstream: ds4-rocm-upstream ds4-server-rocm-upstream ds4-bench-rocm-upstream ds4-eval-rocm-upstream ds4-agent-rocm-upstream
	@echo "ROCm upstream-shaped binaries built with ROCM_ARCH=$(ROCM_ARCH)"

check-rocm-exports: ds4_rocm.o
	$(PYTHON) tools/check_gpu_api_exports.py --backend ds4_rocm.o

rocm-api-smoke-build: ds4_rocm.o
	tools/run_rocm_api_smoke.sh --build-only --out $(ROCM_API_SMOKE)

rocm-api-smoke: ds4_rocm.o
	tools/run_rocm_api_smoke.sh --out $(ROCM_API_SMOKE)

ds4-mtp-oracle-bench-rocm-upstream: tools/mtp_oracle_microbench_gpuapi.o ds4_distributed.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-rocm-upstream: ds4_cli_gpuapi.o ds4_help.o linenoise.o ds4_distributed.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-server-rocm-upstream: ds4_server_gpuapi.o ds4_help.o ds4_kvstore.o rax.o ds4_distributed.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-bench-rocm-upstream: ds4_bench_gpuapi.o ds4_help.o ds4_distributed.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-eval-rocm-upstream: ds4_eval_gpuapi.o ds4_help.o ds4_distributed.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4-agent-rocm-upstream: ds4_agent_gpuapi.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_distributed.o ds4_gpuapi.o ds4_rocm.o
	$(ROCM_HIPCC) -o $@ $^ $(ROCM_LDLIBS)

ds4.o: ds4.c ds4.h ds4_distributed.h ds4_metal.h ds4_gpu.h $(DS4_C_INCS)
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_server.o: ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_distributed.h ds4_metal.h ds4_gpu.h $(DS4_C_INCS)
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_metal.h ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_gpuapi.o: ds4.c ds4.h ds4_distributed.h ds4_metal.h ds4_gpu.h $(DS4_C_INCS)
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4.c

ds4_cli_gpuapi.o: ds4_cli.c ds4.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_cli.c

ds4_server_gpuapi.o: ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_server.c

ds4_bench_gpuapi.o: ds4_bench.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_bench.c

ds4_eval_gpuapi.o: ds4_eval.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_eval.c

ds4_agent_gpuapi.o: ds4_agent.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -c -o $@ ds4_agent.c

tools/mtp_oracle_microbench_gpuapi.o: tools/mtp_oracle_microbench.c ds4.h
	$(CC) $(CFLAGS) -DDS4_USE_GPU_API -DDS4_USE_HIP -I. -c -o $@ tools/mtp_oracle_microbench.c

ds4_rocm.o: ds4_rocm.cu ds4_gpu.h ds4_iq2_tables_cuda.inc ds4_rocm.h rocm/ds4_rocm_runtime.cuh rocm/ds4_rocm_common.cuh rocm/ds4_rocm_embedding_launch.cuh rocm/ds4_rocm_q8.cuh rocm/ds4_rocm_norm_rope.cuh rocm/ds4_rocm_matmul.cuh rocm/ds4_rocm_fp8_kv.cuh rocm/ds4_rocm_fp8_kv_launch.cuh rocm/ds4_rocm_attention.cuh rocm/ds4_rocm_attention_launch.cuh rocm/ds4_rocm_hc.cuh rocm/ds4_rocm_output.cuh rocm/ds4_rocm_misc_launch.cuh rocm/ds4_rocm_indexer.cuh rocm/ds4_rocm_compressor.cuh rocm/ds4_rocm_shared_expert.cuh rocm/ds4_rocm_router.cuh rocm/ds4_rocm_moe.cuh rocm/ds4_rocm_moe_launch.cuh rocm/ds4_rocm_hc_output_launch.cuh rocm/ds4_rocm_hipblaslt.cuh
	$(ROCM_HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

ifneq ($(HIPCC),)
hip-rocwmma-smoke: tools/hip_rocwmma_smoke.cpp
	$(HIPCC) $(HIPCXXFLAGS) -o $@ $<

hip-q2-moe-wmma-bench: tools/hip_q2_moe_wmma_microbench.cpp
	$(HIPCC) $(HIPCXXFLAGS) -o $@ $<

hip-q8-wmma-bench: tools/hip_q8_wmma_microbench.cpp
	$(HIPCC) $(HIPCXXFLAGS) -o $@ $<
else
hip-rocwmma-smoke hip-q2-moe-wmma-bench hip-q8-wmma-bench:
	@echo "hipcc not found; cannot build $@" >&2
	@exit 1
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

pi-stateful-test:
	$(NODE) tests/pi_ds4_stateful_provider_test.mjs

test: ds4_test ds4-eval q4k-dot-test
	./ds4-eval --self-test-extractors
	./ds4_test

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-rocm-upstream ds4-server-rocm-upstream ds4-bench-rocm-upstream ds4-eval-rocm-upstream ds4-agent-rocm-upstream ds4-mtp-oracle-bench-rocm-upstream ds4_cpu ds4_native ds4_server_test ds4_test hip-rocwmma-smoke hip-q2-moe-wmma-bench hip-q8-wmma-bench tests/test_q4k_dot *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o tools/mtp_oracle_microbench_gpuapi.o
