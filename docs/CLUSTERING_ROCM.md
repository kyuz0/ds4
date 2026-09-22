# DeepSeek V4.1: two-machine ROCm cluster

- Two ROCm/gfx1151 machines; tested with 128 GB RAM each.
- Same engine revision and `DeepSeek-V4.1-Flash-Q2.gguf` on both. Each keeps the full GGUF; each machine loads approximately 80.6 GiB of weights into RAM. Engram stays on disk.
- Exactly two machines: one coordinator and one worker. They share attention computation and split the experts equally.
- Cluster mode requires the assigned experts to fit in RAM. SSD expert streaming, DSpark and splitting by `--layers` are not supported.
- Both transports require a reachable TCP control address. Use a trusted network: peer traffic has no authentication or encryption.
- Build both peers with `make strix-halo ROCM_ARCH=gfx1151` after installing any required RoCE headers.
- Run from the engine build directory. Set these variables in **both** terminals; `MODEL` may differ between machines:

```bash
MODEL=/absolute/path/DeepSeek-V4.1-Flash-Q2.gguf
COORD=10.99.0.1       # Coordinator's address on the selected link
CTX=16384
```

## TCP over Ethernet or USB4

- Working Ethernet/IP connection; coordinator TCP port 9911 reachable from the worker. USB4 Ethernet (`thunderbolt_net`) works with the same commands: set `COORD` to the coordinator's USB IP address.
- No RDMA packages required.

```bash
# Coordinator
./ds4-server --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role coordinator --listen "$COORD" 9911 \
  --transport tcp --batched-session 1 --host 127.0.0.1 --port 8080

# Worker, in its own terminal
./ds4 --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role worker --coordinator "$COORD" 9911 \
  --transport tcp
```

## RoCE

- Both hosts need RoCE-capable Ethernet adapters, a working driver and an active Ethernet verbs port. Ordinary Ethernet alone is insufficient.
- Install the runtime/provider packages where inference runs; development headers are needed when building the engine. Package names: [Fedora](https://packages.fedoraproject.org/pkgs/rdma-core/), [Ubuntu](https://packages.ubuntu.com/source/jammy/rdma-core).

```bash
# Fedora, both inference/build environments
sudo dnf install libibverbs libibverbs-utils rdma-core-devel iproute

# Ubuntu/Debian alternative
sudo apt install rdma-core libibverbs1 ibverbs-providers ibverbs-utils libibverbs-dev iproute2
```

```bash
# Both hosts/environments: inspect local device, port and GIDs
ibv_devices
ibv_devinfo -v
rdma link
ulimit -l                      # Locked-memory allowance; at least 16 MiB for RoCE staging
DEV=rocep194s0                 # Replace with this host's verbs device
PORT=1
for file in /sys/class/infiniband/"$DEV"/ports/"$PORT"/gids/*; do
  idx=${file##*/}
  printf '%s  %s  %s  %s\n' "$idx" "$(cat "$file")" \
    "$(cat /sys/class/infiniband/"$DEV"/ports/"$PORT"/gid_attrs/types/"$idx")" \
    "$(cat /sys/class/infiniband/"$DEV"/ports/"$PORT"/gid_attrs/ndevs/"$idx")"
done
GID=1                         # Choose this host's nonzero RoCE v2 GID for the cabled NIC/IP
```

- `rdma link` comes from [Fedora iproute](https://packages.fedoraproject.org/pkgs/iproute/iproute/fedora-rawhide.html) or [Ubuntu iproute2](https://packages.ubuntu.com/jammy/all/iproute2/filelist).
- Device names and GID indexes may differ between hosts. `ibv_devinfo` must show an active port with Ethernet link layer.
- The selected device's `/dev/infiniband/uverbs*` must be accessible. Containers also need its device access, userspace provider and adequate memlock allowance; packages alone do not configure the host NIC.
- Set `COORD` to the coordinator's address on the RoCE Ethernet link.

```bash
# Coordinator
./ds4-server --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role coordinator --listen "$COORD" 9911 \
  --transport rdma --rdma-device "$DEV" --rdma-port "$PORT" --rdma-gid-index "$GID" \
  --batched-session 1 --host 127.0.0.1 --port 8080

# Worker
./ds4 --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role worker --coordinator "$COORD" 9911 \
  --transport rdma --rdma-device "$DEV" --rdma-port "$PORT" --rdma-gid-index "$GID"
```

- RoCE transfers use buffers in system RAM. GPU-direct transfers are not implemented; RCCL is not required.
- Explicit `tcp` or `rdma` fails if unavailable. `auto` negotiates configured RoCE, then TCP at connection setup; no mid-generation fallback.
- Decoder SWA Bounded Replay is off by default. To enable it, set `DS4_ENABLE_V41_DECODER_SWA_BOUNDED_REPLAY=1` on both coordinator and worker commands.

## Vision and first request

- Add `--vision /absolute/path/DeepSeek-V4.1-Flash-Vision.gguf` to **both** commands. Keep `--ctx` equal on both.
- The HTTP API runs only on the coordinator. Test port 8080 after startup; do not send HTTP to peer port 9911.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":64,"thinking":false}'
```


## Measured performance

- Two Framework Desktop systems, 128 GB each, 16-core Strix Halo / `gfx1151`; coordinator Ryzen AI Max+ 395, worker engineering sample `100-000001243-50_Y`.
- Model drives: coordinator SK hynix PC711 1 TB (PCIe 3.0 ×4, ext4); worker Kingston FURY Renegade 2 TB (`SFYRD2000G`, PCIe 4.0 ×4, btrfs). Engram stays disk-backed.
- Intel E810-C QSFP NICs, 100 Gb/s, MTU 9000; coordinator NIC at PCIe 3.0 ×4, worker at PCIe 4.0 ×4. Device logs and hardware send counters confirm RDMA payloads; TCP also carries RoCE connection setup.
- Linux `7.2.5-100.fc43.x86_64`, ROCm SDK `10.0.0-4` / HIP `7.15.26333`, TuneD `accelerator-performance`. Existing boot settings include `pci=realloc pcie_aspm=off` and the [GPU-visible memory settings](STRIX_HALO.md#gpu-visible-memory); their individual effects were not isolated.
- Native `ds4-bench`, Q2, 34,816 allocated context, 128 greedy output tokens per frontier, no DSpark or images. First row is a fresh prefix; second appends 8,192 tokens to the restored prefix. Startup excluded; one final run per cell, not a cold-cache measurement. Values are **prefill / decode tokens/s**.

| Context tokens | Appended tokens | TCP, 100 GbE | RoCE, 100 GbE |
|---:|---:|---:|---:|
| 8,192 | 8,192 | 395.61 / 16.59 | 391.56 / 17.33 |
| 16,384 | 8,192 | 382.74 / 16.32 | 380.32 / 17.13 |

- Full 129,280-logit vectors and 128-token continuations match controls and each other. No OOM. Scalar transport waiting uses active CPU polling; measured CPU cost is about half a core per host during decode.
- V4.1 CED uses about 8B active parameters/token in prefill and 16B in decode. Short appends can follow a different schedule. [Architecture](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/README.md?code=true).

### Reproduce the table

Use the transport/device setup above on both machines, then set:

```bash
MODEL=/absolute/path/DeepSeek-V4.1-Flash-Q2.gguf
COORD=10.99.0.1
TRANSPORT=rdma                         # tcp or rdma
DEV=rocep194s0                          # This host's active verbs device
GID=1                                  # This host's matching RoCE v2 GID
LINK=(--tensor-parallel --transport "$TRANSPORT")
case "$TRANSPORT" in
  rdma) LINK+=(--rdma-device "$DEV" --rdma-port 1 --rdma-gid-index "$GID") ;;
esac
```

```bash
# Coordinator, from the engine source directory: prepare the measured text.
python3 - <<'PYTHON'
from pathlib import Path
text = Path('tests/test-vectors/flash-vision-exp/prompts/long_memory_archive.txt').read_text()
Path('bench-prompt.txt').write_text((text + '\n') * 16)
PYTHON

tuned-adm active                        # Verify accelerator-performance during the workload
DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 ./ds4-bench --rocm -m "$MODEL" \
  --prompt-file bench-prompt.txt \
  --ctx-start 8192 --ctx-max 16384 --step-mul 1 --step-incr 8192 --ctx-alloc 34816 \
  --gen-tokens 128 --show-output --csv "tp-$TRANSPORT.csv" \
  --dump-frontier-logits-dir "frontiers-$TRANSPORT" \
  --role coordinator --listen "$COORD" 9911 "${LINK[@]}"

# Worker, start for each coordinator run.
./ds4 --rocm -m "$MODEL" --ctx 34816 \
  --role worker --coordinator "$COORD" 9911 "${LINK[@]}"
```

- The 2 GiB snapshot cap avoids replaying a prefix when restoring benchmark state; leave sufficient free RAM. Save CSV, frontier files, output, revision/build flags, model filename/size, profile and memory/swap counters.
