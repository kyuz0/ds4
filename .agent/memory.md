# Память проекта: ds4 (DwarfStar) — ROCm gfx1151 / Strix Halo

## Задача текущей сессии
Оптимизация производительности DeepSeek V4.1 Flash для **Strix Halo gfx1151 + ROCm 7.2**.
Ветка: `feat/rocm-deepseek41-halo-fix-kv-cache` (HEAD 7e0276f, 22 сен).
Базовый лог пользователя (ds4-server, Q2, `--ssd-streaming-cache-experts 96GB --ctx 500000`):
prefill 65 токенов = 17.5 с (3.71 t/s), decode ~9-9.5 t/s.

## Окружение
- **Dev ПК (этот)**: только разработка. Репозиторий `/home/neiron/work/ds-4.1-flash/ds4`.
- **Тестовый ПК (сборка/тесты)**: `sshpass -p 1234 ssh neiron@192.168.3.168` (bazzite, Ryzen AI 9 HX 370 / Radeon 890M / **gfx1150**, 92 GB).
  ROCm живёт в distrobox-контейнере `comfyui-rocm`: `HIP 7.2.53211`, hipBLASLt **1.2.2** (revision `dabb6df2b98`), rocBLAS ~5.x.
  Клон проекта на тестовом ПК: `~/work/ds-4.1-flash/ds4` (без .git). Моделей V4.1 нет — полный прогон там невозможен.
- **Целевая машина (пользователя)**: FEVM FAEX1 AI MAX+ 395, Radeon 8060S (gfx1151), 128 GB, RAID-0 3×Apacer AS2280Q4X 1TB, CachyOS.
  Замер fio (25.09): **реальный потолок ≈ 7.4-7.5 GiB/s (8.0-8.3 GB/s)** — заявленные 13.5 GB/s не подтверждаются (диски на 95-100 % util);
  4k random 134k IOPS / depth-1 latency 173 µs. Доступ у агента отсутствует — финальные замеры делает пользователь.
- Пароль тестового ПК и прочие детали: `/home/neiron/work/mcp-multimodal/SUMMARY.md`.

## Ключевые факты о коде (V4.1 ROCm SSD streaming)
- Форма V4.1 (`DS4_SHAPE_FLASH41`, `ds4.c:640`): 40 слоёв, 384 эксперта/слой (top-6 + 1 shared), n_embd 5120, n_ff_exp 2304.
  Всего routed-экспертов 15360; один эксперт (gate IQ2_XXS + up IQ2_XXS + down Q2_K) ≈ **9.49 MiB**; весь набор ≈ 152.7 GiB.
- Конфигурация кэша: `ds41_stream_cache_configure` (ds4.c:69225). `--ssd-streaming-cache-experts 96GB` →
  `7.12 GiB two-layer staging + 88.88 GiB dynamic (9588 experts)`. staging = 2 полных слоя × 384 эксперта, выделяется `cudaMalloc` при старте.
- Бюджет памяти (`ds41_memory_admit_for_host`, ds4.c:71290): `budget = host - reserve`, reserve = host/16 (min 8 GiB) + 2 GiB = 10 GiB.
  Практический потолок dynamic-кэша ≈ 10031 экспертов (~93 GiB) — на 4.6 % больше текущих 9588.
- Runtime-резерв свободной памяти при ленивом росте кэша: `DS4_ROCM_STREAM_FREE_RESERVE_GB` (по умолчанию = тот же 10 GiB).
- Decode: ~9.33 t/s при любом контексте. **НЕ storage-bound** (замер пользователя 25.09): hit rate кэша экспертов 98.1 % (4238468/4320720),
  с SSD ~0.43 GB/s при RAID 13.5 GB/s. Конфиг-ручки (кэш 94-96 GB, workers, chunk, reserve) не влияют. Утверждение docs/STRIX_HALO.md «Decode is storage-bound» — неверно.
  Симуляция per-layer LRU из профиля: cap 256 → 98.56 %, cap 384 → 99.56 % → политика почти оптимальна (потолок +1.5 пп).
  Оценка трафика весов на токен ≈ 8.8 GB (routed 2.39 + D2D-компактизация ~2.4 + shared ~1.5 + attention ~1.2 + Engram 0.74 + head ~0.54) → потолок ~26 t/s.
  Гипотеза подтверждена замером фаз (см. discoveries §10): конвейер GPU-bound (~84 ms/token GPU-работы при трафике ≈11.3 GB/token → ~141 GB/s эффективной полосы),
  host ~5 ms/token, miss-чтения на критическом пути (~15-20 ms/token), `rest` ~9-12 ms/token. Главный рычаг — убрать D2D-компактизацию и скрыть miss-сталы
  (split-путь, реализован под `DS4_ROCM_V41_MOE_SPLIT=1`).
- Prefill (layer-major): `ds41_stream_layer_*` (ds4.c:42310+) грузит **полный слой** (384 эксперта, ~3.82 GB) в staging;
  эксперты, уже лежащие в resident-кэше, копируются D2D (`cuda_stream_v41_prepare_layer`, runtime:3089). Первый prefill с холодным кэшем читает все 15360 экспертов ≈ 152.8 GB
  → наблюдаемые 17.66 с = **8.65 GB/s** — это уже на реальном пределе RAID (8.3 GB/s по fio), запаса нет.
- **Prefill коротких промптов идёт token-major через decode-путь**: `ds41_prefill_count` (ds4.c:42090) на ROCm возвращает 1 при `remaining < 256` →
  `layer_major=false` → `ds41_graph_step` на токен (в логе пользователя нет строки `V4.1 layer reuse`, `prepares=0`; `prefill chunk N/65` — по токену).
  Следствие: дополнения <256 токенов тёплого кэша идут по ~107 мс/токен (65 → ~7 с, 255 → ~27 с); sweep ≈ 7.2 с (читает только missing-эксперты слоя).
  Ручка (новая): `DS4_ROCM_V41_PREFILL_MIN_TOKENS` (2..4096, default 256; поведение по умолчанию не меняется).
- Сид кэша из prefill: `ds41_prefill_seed` (ds4.c:41897), включён по умолчанию (кроме `DS4_METAL_DISABLE_STREAMING_PREFILL_CACHE_SEED`).
  После каждого слоя берёт top-`budget/40` ≈ 239 экспертов по частоте в промпте и копирует их D2D из staging в resident-кэш.
- Эвикция resident-кэша: `cuda_stream_resident_evict_one` (runtime:1567) — глобальный LRU по clock; опционально
  `DS4_ROCM_STREAM_EVICT_PAST_LAYERS_FIRST` (по умолчанию off). Admission без фильтра.
- Read pool: `DS4_ROCM_STREAM_READ_WORKERS` (по умолчанию 16, максимум 24), O_DIRECT (отключается `DS4_ROCM_STREAM_NO_DIRECT`),
  staging 32 MiB/worker; чанк слоя 32 MiB (хардкод в `cuda_stream_layer_expert_cache_load`).
- Статистика кэша: `DS4_ROCM_STREAM_CACHE_STATS=1` (+ `DS4_ROCM_STREAM_CACHE_LAYER_STATS=1`) → печатает hits/misses/evictions при cleanup.
- Профайлер экспертов (`--expert-profile` / `DS4_EXPERT_PROFILE`, `--expert-hotlist` через `DS4_EXPERT_HOTLIST`):
  теперь **разрешён и для V4.1 ROCm** (гейт Metal-only снят; запись роутинга в `ds41_stream_selected_begin`).
  Пишет JSON (per-layer LRU hit-rate для caps 1..384) + hotlist-файл формата `layer expert hits weight`.
  Снятые данные пользователя (25.09) лежат на dev-ПК: `/home/neiron/work/ds-4.1-flash/tmp/{v41_profile.json,v41_hotlist.txt,profile.txt}`.
- Профайлер фаз decode: `DS4_ROCM_V41_DECODE_PROFILE=1` — строка на токен (`host[pre/moe/post]`, `moe_sub[read/plan/launch]`, `gpu`, `rest`);
  `DS4_ROCM_V41_DECODE_PROFILE_LAYER=1` — строка на слой. Перекрытие слоёв под streaming: `DS4_ROCM_V41_STREAM_QUEUE_LAYERS=1`
  (снимает per-layer `cudaDeviceSynchronize`; замерено: выигрыша нет — конвейер GPU-bound).
- Split-путь MoE (главный рычаг decode): `DS4_ROCM_V41_MOE_SPLIT` — **дефолт на gfx1151** (`ds4_rocm_gfx1151_flag`: unset = включён, `=0` = compact, `=1` = включить на других ROCm).
  Resident/missing split по указателям слотов вместо compact-таблицы. Harness с **реальными dims V4.1** и принудительно включёнными gfx1151-ядрами
  (`tmp/v41split_big.c`, тест-хук `DS4_ROCM_FORCE_GFX1151` на тестовом ПК): compact vs split **побитово идентичны** (`mid` 0/13824, `out` 0/5120, max=0),
  MoE hot 2.790 → **1.067 ms (2.6×)**. Замер пользователя: decode 9.38 → 11.71 t/s, prefill 18.19 → 16.31 с; при `"temperature":0` выводы совпадают.
- **KV-reuse при правке сообщения — root cause найден и исправлен (29.09)**: переиспользование части префикса возможно только через frontier-снимки
  (`ds41_gpu_graph.frontier_state`: кольцо 8 prefill + 8 decode слотов, состояние = `(pos<<1)|has_logits`; хинт `ds4_session_frontier_hint`, restore `ds41_frontier_restore`).
  Баг был в хинте: `if (pos > (uint32_t)best)` при сентинеле `best = -1` → `(uint32_t)(-1) = UINT32_MAX` → `best` не обновляется → хинт всегда -1 → evict + полный prefill.
  Фикс: `if ((int)pos > best) best = (int)pos;` (ds4.c:86663, в рабочем дереве). Соседний `ds41_frontier_restore` (ds4.c:40600) сравнивает корректно (`best >= 0 && pos <= best_pos`).
  Диагностика: `DS4_ROCM_V41_FRONTIER_DEBUG=1` (захваты, дамп кольца, выбор restore, решение пробы в сервере).
  Сервер в batched-режиме синкает префиксы по 2048 токенов (`server_session_sync`), поэтому захватов много; decode-снимки заполняются только в spec-ветке (для V4.1 неактуально).
  Механизм **fork-local**: внесён коммитом 7e0276f, в апстриме antirez/ds4 (main @ 0aaea5a) отсутствует.
- **Ловушка: сентинел -1 в unsigned-сравнениях**: в этом коде часто `int best = -1`; сравнивать с unsigned-позициями можно только как `(int)pos > best` или
  `best < 0 || pos > best_pos`. `pos > (uint32_t)best` при `best=-1` молча никогда не истинно (`UINT32_MAX`) — именно так возник баг frontier hint.
- **Тестовый ПК теперь с двумя GPU**: RX 7800M (gfx1101, индекс 0) + Radeon 890M (gfx1150, индекс 1). Бинарь под gfx1150 на индексе 0 падает
  с HIP-ассерцией `StatCO::getStatFunc` → все сборки/тесты запускать с явным выбором устройства.
  **ВАЖНО (изменилось 01.10): на тестовом ПК теперь iGPU = HIP-индекс 0** (`HIP_VISIBLE_DEVICES=1` даёт `no ROCm-capable device is detected`; RX 7800M в HIP не виден).
  Команды: `HIP_VISIBLE_DEVICES=0 make strix-halo ROCM_ARCH=gfx1150`; для kyuz0-тестов V4.1 на gfx1150 обязателен тест-хук `DS4_ROCM_FORCE_GFX1151=1`
  (иначе native-gfx1150 fallback-пути не совпадают с gfx1151-оракулами тестов).
- **Семплинг сервера (важно для A/B)**: env `DS4_SERVER_DEFAULT_TEMP` **не существует** (в репо нет); дефолт `DS4_DEFAULT_TEMPERATURE = 1.0f` (`ds4.h:59`, `docs/SERVER.md:46`),
  в thinking-режиме thinking-дефолты (`top_k=0`, `min_p=0.05`), **явные параметры запроса приоритетны** (`ds4_server.c:13971`), **seed случаен на каждый запрос**
  (`random_bytes` из `/dev/urandom`, `ds4_server.c:97/13916`) → один и тот же запрос даёт **разный текст на каждом вызове**. Для детерминизма — `"temperature": 0` **в теле запроса**
  (или `"seed": N` для воспроизводимого не-greedy). Диагностика: `--trace FILE` → печатает `temperature:`/`seed:` и raw request json (`ds4_server.c:11997+`).
  Именно это (а не split) объяснило «разный результат» в A/B SPLIT-0/1 от 26.09.
- Гонок в streaming-кэше нет ни в одном пути: план слоя (`cuda_stream_selected_load`, runtime:4648) начинается с `cuda_stream_selected_reuse_wait` (4683) —
  host ждёт reuse-event предыдущего слоя (записывается в **stream 0** после ядер, `mark_inflight`; в регионе `!q2k_path` — `ds4_rocm_moe_launch.cuh:2161`).
  Слои сериализованы → таблица/слоты не перезаписываются под работающими ядрами. Это же объясняет, почему `QUEUE_LAYERS` не даёт выигрыша.
- Профиль фаз замерен 25.09 (steady state): `total=116.0ms host=66.7 (pre 1.8, moe 64.8, post 0.1) moe_sub[read 43.3 plan 1.0 launch 19.9] gpu=40.5 rest=8.8`
  → GPU ~84 ms/token при трафике ≈ 11.3 GB/token (~141 GB/s эффективной полосы, пик ~230); compaction = 2.2 GB (19 %) — удаляется split-путём.
- Статические hotlist-таблицы: `ds4_streaming_hotlist.inc` (PRO, FLASH) и `_glm52.inc`; загрузчик
  `metal_graph_streaming_expert_hotlist_load_default` (ds4.c:22998) поддерживает варианты PRO/FLASH/GLM52,
  для **FLASH41 (V4.1) hotlist отсутствует** → не загружается (но работает файл через `DS4_ROCM_STREAMING_EXPERT_HOTLIST`).
- ROCm 7.2 vs ROCm 10.0 (тюнинг-пути отключены точным сравнением версий):
  - rocBLAS: `g_rocblas_f16_solution_set` требует версии `5.5.0.cd957402` или `5.6.0.8d1ae90e` (runtime:6422). На 7.2 → NONE.
  - hipBLASLt fixed-index prefill (2537/2539) и V4.1 Engram: требуют `version == 100401 && revision == "8d1ae90e"`
    (`rocm/ds4_rocm_hipblaslt.cuh:194`, `rocm/ds4_rocm_v41.cuh:940`). На 7.2 (1.2.2/dabb6df2b98) → выключено, работает fallback (WMMA-ядра).
  - Коммит `1ef9bba` (ветка origin/perf/rocm-gfx1151-ds4f-performance): warm 16K prefill 262.64 → 301.30 t/s на ROCm 10.0.
- KV/индексер V4.1: сжатый KV = 2.98 GiB при ctx 500000; индексер считается только в **4 слоях** (остальные переиспользуют выбор) — не главный расход.
- V4.1 НЕ поддерживает DSpark/спек-декод (`docs/MODELS.md:93`), поэтому DSpark-пути не важны.
- Engram: таблицы disk-only; проекция `engram_kv` (6144×25600 F16, 2 слоя: blk.1 и blk.14) — в decode rows=1 идёт через
  `ds4_gpu_matmul_f16_tensor` (fallback), в prefill rows>=32 через Lt-план/`engram_lds_token_reuse`.

## Полезные команды
- Сборка/тесты на тестовом ПК (внутри контейнера, без CPATH и -L — дефолты Makefile уже починены):
  `distrobox enter comfyui-rocm -- bash -lc 'cd /home/neiron/work/ds-4.1-flash/ds4 && HIP_VISIBLE_DEVICES=0 make strix-halo ROCM_ARCH=gfx1150 && HIP_VISIBLE_DEVICES=0 DS4_ROCM_FORCE_GFX1151=1 make test-rocm ROCM_ARCH=gfx1150'`
  (`HIP_VISIBLE_DEVICES=0`: на тестовом ПК iGPU = индекс 0; внутри distrobox HOME другой — используй абсолютный путь /home/neiron/work/ds-4.1-flash/ds4)
- Патч перегенерируется из pristine-чекаута: `git clone --depth 1 --branch feat/rocm-deepseek41-halo-fix-kv-cache git@github.com:hitman249/ds4.git /tmp/ds4-pristine`,
  скопировать изменённые файлы (8 шт.: ds4.c, ds4_help.c, ds4_server.c + 5 rocm/*.cuh) и `git -C /tmp/ds4-pristine diff > rocm72-perf.patch`.
  Проверка: `git -C /tmp/ds4-pristine reset --hard HEAD` → `git apply --check` + `patch -p1 --dry-run` → apply → sha256 всех 8 файлов совпадают с рабочей копией → снова `reset --hard` (pristine оставить чистым).
- Кросс-сборка под цель: `make strix-halo` (ROCM_ARCH=gfx1151) — компиляция проходит на gfx1150-ПК.
- Синк файлов: `sshpass -p 1234 rsync -az --exclude .git -e ssh <files> neiron@192.168.3.168:~/work/ds-4.1-flash/ds4/`
- Проверка «апстримный ли баг/фича»: `git log -S "<symbol>" -- ds4.c` (коммит-внедрение), `git show origin/main:ds4.c | rg <symbol>`,
  `curl -sSL https://raw.githubusercontent.com/antirez/ds4/main/ds4.c | rg <symbol>` (upstream = antirez/ds4; база форка — 0aaea5a).
- Быстрый compile-check правки ds4.c: `make ds4.o` (~45 с); `DS4_HAS_DEEPSEEK41_GPU` включён по умолчанию (ds4.c:51, `DS4_NO_GPU` не задан),
  так что ветки `#ifdef DS4_HAS_DEEPSEEK41_GPU` компилируются и на dev-ПК.
- На целевой машине пользователя (для замеров):
  `./ds4-server --rocm -m ...Q2.gguf --ssd-streaming --ssd-streaming-cache-experts 96GB --ctx 500000 --batched-session 1`
  плюс `DS4_ROCM_STREAM_CACHE_STATS=1` для hit/miss/evictions.
