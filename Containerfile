
# Build the locally modified DS4 source
FROM registry.fedoraproject.org/fedora:44 AS builder

RUN cat > /etc/yum.repos.d/rocm.repo <<'REPO'
[amdrocm-stable]
name=ROCm 10.0.0
baseurl=https://stable.repo.amd.com/rocm/core/packages/rhel10/x86_64
enabled=1
priority=50
gpgcheck=1
gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
REPO

RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
    make gcc gcc-c++ cmake libcurl-devel ninja-build \
    amdrocm-core-devel10.0-gfx1151 rdma-core-devel git-core \
    && dnf clean all

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib

WORKDIR /src/ds4
COPY . .

RUN make clean && \
    make -j4 strix-halo ROCM_ARCH=gfx1151 \
       CFLAGS="-O3 -ffast-math -march=native -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -fPIC" && \
    test -x ds4-server && test -x ds4 && test -x ds4-bench

FROM docker.io/kyuz0/strix-halo-ds4-toolbox:rocm-10.0

COPY --from=builder /src/ds4/ds4-server /usr/local/bin/ds4-server
COPY --from=builder /src/ds4/ds4 /usr/local/bin/ds4
COPY --from=builder /src/ds4/ds4-bench /usr/local/bin/ds4-bench

ENV DS4_ROCM_V41_SELECTED_READ_CHUNKS=8

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/ds4-server"]

CMD ["--rocm", "-m", "/models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf", "--ctx", "131072", "--ssd-streaming", "--ssd-streaming-cache-experts", "92GB", "--host", "0.0.0.0", "--port", "8000"]

