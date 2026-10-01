# Открытия и эксперименты (проверено → результат → вывод)

## 1. Профиль узкого места V4.1 на Strix Halo (анализ лога + кода)
- Проверено: decode держит ~9-9.5 t/s при любом контексте; prefill 65 токенов = 17.5 с.
- Результат: decode — storage-bound: ~1.3-1.4 GB экспертов с SSD на токен (RAID 13.5 GB/s). Prefill первого запроса читает весь набор
  экспертов (40 слоёв × 384 × 9.49 MiB ≈ 152.7 GiB) layer-major стейджингом → при 13.5 GB/s это 11.3 с; наблюдаемые 17.5 с = ~8.7 GB/s эффективной полосы.
- Вывод: единственный рычаг decode — hit rate кэша экспертов (или его размер). Сайзинг: запрошено 96 GB → 88.88 GiB dynamic (9588 слотов),
  практический потолок по памяти ≈ 10031 слотов. Runtime-резерв свободной памяти 10 GiB (`DS4_ROCM_STREAM_FREE_RESERVE_GB`) может не дать кэшу
  вырасти до конца. Сид из prefill — top-239/слой по частоте в промпте; эвикция — глобальный LRU.
  **ОПРОВЕРГНУТО 25.09 (см. §6):** по реальным логам пользователя hit rate кэша = 98.1 %, с SSD читается ~0.43 GB/s — decode НЕ storage-bound.

## 6. РЕШАЮЩИЙ ЗАМЕР ПОЛЬЗОВАТЕЛЯ (Strix Halo, 25.09): hit rate 98.1 % — decode НЕ storage-bound
- Проверено (логи двух прогонов, кэш 94 и 95 GB): `stream cache stats cleanup`: selected calls=720120 slots=4320720 hits=4238468 misses=82252
  → **hit 98.1 %**; allocs=82252 (762.45 GiB) / evictions=72880 (675.58 GiB) за 1778 с → средний поток с SSD ≈ **0.43 GB/s**
  (при реальном потолке RAID ≈ 8.3 GB/s — см. §8 — это <6 % полосы).
  Per-layer hit по всем 40 слоям 97.7-98.4 %.
- Конфиг-ручки не влияют: кэш 94/95/96 GB, `FREE_RESERVE_GB=6`, `READ_WORKERS=24`, `LAYER_CHUNK_MB=64` → decode ~9.33-9.35 t/s, prefill ~17.7 с.
  Значит ни ёмкость кэша, ни I/O-параметры не лимитируют decode.
- Симуляция политики из профайла (`/home/neiron/work/ds-4.1-flash/tmp/v41_profile.json`, per-layer LRU, caps 1..384): cap 128 → 92.1 %,
  cap 256 → 98.56 %, cap 384 → 99.56 %. Рантайм имеет ~234 слота/слой → ~98.3 % (совпадает с наблюдением) → **политика почти оптимальна**, потолок +1.5 пп.
  Профиль: 40 слоёв, 523040 записей, 3138240 выборов; adjacent overlap 0.087-0.44 (слабая временная локальность, сильный перекос популярности).
- Оценка трафика весов на токен (из формы модели): routed 240×9.95 MB = 2.39 GB (из compact-кэша) + **D2D-компактизация ~2.4 GB**
  (`cuda_stream_selected_compact_mask` копирует все 6 экспертов в compact-буфер каждый слой) + shared Q8_0 ~1.5 GB + attention ~1.2 GB
  + Engram (2 слоя × 315 MB F16) 0.74 GB + head ~0.54 GB ≈ **8.8 GB/token** → при ~230 GB/s потолок ≈ 26 t/s. Наблюдаем 9.3 → запас 2-3× есть.
- Гипотеза (ждёт профиля фаз): decode лимитят host-side launch и синки. На слой: readback-синк (`ds4_gpu_tensor_read` 6 id) + `ds4_gpu_end_commands()`
  (= `cudaDeviceSynchronize`) при `--ssd-streaming` (`queue_layers=false` → drain каждый слой). Цепочка router(GPU) → readback/plan(host) → MoE(GPU)
  сериализует конвейер; 40 слоёв × (host-encode + 2 синка) может дать десятки мс/токен.
- Следствие: размер кэша и политику эвикции трогать бессмысленно; рычаги — (а) перекрыть host и GPU / убрать per-layer drain,
  (б) убрать D2D-компактизацию (slot-pointer путь), (в) эффективность MoE-ядер.

## 2. ROCm 7.2 отключает версионно-зашитые тюнинг-пути (проверено на тестовом ПК)
- Проверено: на тестовом ПК в контейнере hipBLASLt 1.2.2 (revision `dabb6df2b98`), rocBLAS не 5.5/5.6 → `g_rocblas_f16_solution_set = NONE`;
  `hipblasLtGetVersion` ≠ 100401 → fixed-index планы 2537/2539 и Engram-план отключены; работают WMMA/кастомные fallback-ядра.
- Вывод: на целевой машине с ROCm 7.2 prefill идёт по fallback-путям (на ROCm 10.0 тюнинг давал +15 % к prefill: 262 → 301 t/s).

## 3. Бенчмарк Lt-heuristic vs fallback на gfx1150 (ROCm 7.2, та же библиотека) — ключевой замер
Программа `/tmp/lt_bench.c` (вне репозитория) через `ds4_gpu_matmul_f16_tensor` / `ds4_gpu_dsv41_projection_rows`, best-of-3 со синком.

| Фигура (n_tok=2048) | WMMA fallback | Lt-heuristic | Выигрыш |
|---|---:|---:|---:|
| 16384→24 | 6.31 ms | 3.88 ms | 1.63× |
| 4096→1024 | 15.94 ms | 9.37 ms | 1.70× |
| 1024→8192 | 33.86 ms | 17.22 ms | 1.97× |
| Engram 6144→25600 (rows=2048), кастомное ядро `engram_lds_token_reuse<16>` | 1296.8 ms | 467.0 ms | **2.78×** |
| Engram, per-row generic (Lt-гейт выключен) | 7816.8 ms | 467.0 ms | 16.8× |

- Проверено: с `DS4_ROCM_HIPBLASLT_PREFILL_HEURISTIC=1 DS4_ROCM_F16_LT_PREFILL=1` Lt-план создаётся (нет сообщений «no algo») и считает.
  Суммы близки к fallback (различие — порядок накопления/F16-конвертация активаций; у Engram есть проверка точности конверсии).
- Вывод: opt-in Lt-heuristic реально быстрее fallback на той же библиотеке ROCm 7.2. На gfx1151 fallback-ядра специально тюнены, поэтому
  финальное решение — за A/B на целевой машине (флаг уже есть).

## 4. Ловушка rsync (урок)
- Проверено: `rsync -a rocm/ target/` копирует *содержимое* rocm/ в target (не в target/rocm/) → .cuh оказались в корне проекта на тестовом ПК,
  а `rocm/*.cuh` остались старыми; из-за этого один прогон бенчмарка не задействовал Lt-путь и дал ложный вывод «Lt медленнее».
- Вывод: синкать `rocm/` только в `.../ds4/rocm/` (или без trailing slash). Проверять `grep` ключевого маркера на удалённой стороне перед замерами.

## 5. Инфраструктура тестового ПК
- Проверено: `make strix-halo ROCM_ARCH=gfx1150` собирает все 5 бинарей; `make test-deepseek41-rocm` — 118 shape-тестов V4.1 ROCm (PASS);
  `make test-rocm` + `q4k-dot-test/mxfp4-dot-test` — зелёные; кросс-компиляция `--offload-arch=gfx1151` проходит.
- Вывод: цикл разработки рабочий; полный прогон модели возможен только у пользователя на Strix Halo.

## 7. Локальной валидации streaming decode MoE под ROCm нет (проверено 25.09)
- Проверено: `tests/test_metal_ssd_experts.c` (единственный тест, гоняющий `ds4_gpu_routed_moe_one_tensor` через streaming-кэш с эвикцией и сверкой
  streaming vs full-table) — macOS-only (`#include <mach/mach.h>` + `task_info`), на Linux не собирается.
  `tests/test_cuda_ssd_batch.c` линкуется с CUDA-объектами и требует символов, которых нет в ROCm-совместимости (`ds4_gpu_qwen4_*`, `ds4_gpu_dsv41_shared_start/join`,
  `ds4_gpu_matmul_f16_rms_fold_tensor`, ...) — под ROCm не линкуется.
- Вывод: для будущей правки MoE-пути (убрать D2D-компактизацию / перейти на slot-указатели) нужен новый Linux/ROCm-тест по образцу `test_metal_ssd_experts`:
  fake model map (mmap temp-файла с синтетическими экспертами) + сверка выхода streaming-пути с full-table.

## 8. fio RAID-0 на целевой машине (25.09): реальный потолок ~7.5 GiB/s, а не 13.5 GB/s
- Проверено (пользователь, fio 3.42, O_DIRECT):
  - seq 32 MiB, psync ×4 job (глубина фактически 1): **7400 MiB/s** (7760 MB/s), clat avg 16.4 ms; util md0 94.6 %, nvme 77.9/81.2/89.8 %.
  - seq 32 MiB, io_uring iodepth=8 ×4 job: **7466 MiB/s**, clat avg 125 ms; util md0 99.5 %, nvme 92.8-97.8 % → глубина не помогает.
  - randread 4k, iodepth=32 ×4 job: 134k IOPS, 525 MiB/s, clat avg 951 µs; util 99-100 %.
  - randread 4k, iodepth=1: 5626 IOPS, 22 MiB/s, clat avg 173 µs.
- Вывод: **реальный потолок RAID-0 ≈ 7.4-7.5 GiB/s (8.0-8.3 GB/s)**; заявленные 13.5 GB/s не подтверждаются (диски уже на 95-100 % util).
  4k random latency 173 µs (depth 1) / ~1 ms (depth 32) — слабо для NVMe (QLC/DRAM-less + md RAID0).
- Следствия:
  1. Prefill первого запроса (152.8 GB / 17.66 с = **8.65 GB/s**) уже **на пределе устройства** → тюнинг воркеров/чанков бесполезен;
     ускорить prefill можно только чтением меньшего объёма (тёплый кэш / меньший набор экспертов).
  2. Decode: 0.43 GB/s среднего потока с SSD — накопитель простаивает → decode не I/O-bound (подтверждено). Miss-чтения decode ≈ 48 MB/token
     (4.8 miss/token) → ~6 ms/token при полосе 8.3 GB/s; при latency-bound (3 региона по 3-4 MB на miss) может добавить больше — измерит фазовый профиль.

## 9. Prefill коротких промптов идёт token-major через decode-путь (найдено 25.09)
- Проверено по логам пользователя: в прогоне с промптом 65 токенов **нет строки `V4.1 layer reuse`** (`prepares=0`), `batch calls=0`, `seed calls=0` →
  sweep-загрузчик слоёв не использовался. Сервер печатает `prefill chunk N/65` на каждый токен → обработка по токену через `ds41_graph_step` (decode-путь).
- Причина в коде: `ds41_prefill_count` (ds4.c:42090): на ROCm `minimum = 256` → при `remaining < 256` возвращает 1 → `layer_major=false` → token-major.
  (Apple-ветка для тёплого кэша поднимает порог до 1024; у ROCm такой логики нет.)
- Следствие: любой append < 256 токенов идёт по ~107 мс/токен (тёплый кэш) → 65 токенов ≈ 7 с, 255 токенов ≈ 27 с;
  sweep для 65 токенов ≈ 7.2 с (тёплый: читает только missing-эксперты слоя, ~39 % × 3.82 GB/слой = 60 GB при 61 % resident).
  Точка безубыточности ≈ 64-70 токенов (тёплый) / ≈ 170 (холодный).
- Ручка (новая): `DS4_ROCM_V41_PREFILL_MIN_TOKENS` (2..4096, default 256) — понижает порог под streaming; проверено harness'ом на gfx1150:
  default rem<256 → 1; min=64: rem<64 → 1, rem=64..255 → сам count. Dispatch-тест `tests/test_deepseek41_prefill.c` под ROCm не собирается
  (нет `ds41_short_prefill_count`) → валидация harness'ом + A/B у пользователя (temp=0 для идентичности вывода).
- Замечание: sweep-путь для малых count юнит-тестами не покрыт (`tests/test_deepseek41_graph.c` гоняет `ds41_graph_prefill` для 4096/6144/8192/16384) —
  включать только A/B с проверкой вывода.
- **Замерено пользователем (25.09, vm.drop_caches=3 перед каждым прогоном):** sweep для холодного промпта 65 токенов = **24.16 с** (min=64 и min=32),
  token-major = **17.74-17.80 с** (min=128, min=256, дефолт) → **sweep медленнее на ~36 %** на холоде (читает полный набор ~152.8 GB при меньшей
  эффективности I/O). Вывод: ручку ниже ~128 не ставить для первого (холодного) запроса; выгода возможна только на тёплых дополнениях ≥128 токенов (не замерено).

## 10. Фазовый профиль decode (25.09) + split-путь MoE (реализован, проверен harness'ом)
- Профиль baseline (steady state, 116.0 ms/token): host 66.7 (pre 1.8, **moe 64.8**, post 0.1), moe_sub **read 43.3** / plan 1.0 / **launch 19.9**, gpu(drain) 40.5, rest 8.8.
  С `DS4_ROCM_V41_STREAM_QUEUE_LAYERS=1`: gpu 40.5 → **2.1**, но read 43.3 → **80.7** → итог тот же **115.7 ms** → конвейер GPU-bound, drain-синк не был оверхедом.
  В `launch` видны всплески до 55 ms на токенах с miss (join ждёт pread+upload) → miss-чтения стоят на критическом пути (~15-20 ms/token в среднем).
- Трафик весов на токен ≈ **11.3 GB**: attention+router 4.43 GB (55 % на слой!), routed 2.26, **compaction D2D 2.20**, shared 1.44, engram 0.59, head 0.35.
  При ~80 ms GPU это ~**141 GB/s** эффективной полосы (пик ~230) → запас ~1.6×; compaction = 19 % трафика и удаляется без потери точности.
- Реализовано (opt-in): `DS4_ROCM_V41_MOE_SPLIT=1` — в `routed_moe_launch` включается уже написанный, но отключённый (`split_selected = 0`)
  one-token resident/missing split: `cuda_stream_selected_apply_split` + ptrs-ядра gate/up/down (без compact-таблицы). Условие `split_supported` ослаблено
  до `(resident|missing) != 0` (одиночные проходы с mask=0 безопасны — ядро возвращается до записи). Reuse-event уже покрыт (`compact_selected = split_selected || ...`).
- Проверено harness'ом `/home/neiron/work/ds-4.1-flash/tmp/v41split.c` на gfx1150 (синтетические IQ2/Q2_K эксперты, 40 итераций hot/mixed/cold с эвикциями):
  **0 расхождений с compact-путём**; время MoE-вызова (план+запуск+drain, best-of-200): hot (все resident) compact 0.120 → split 0.065 ms (**1.85×**),
  cold (каждый вызов со свежими miss-экспертами) 0.181 → 0.067 ms (**2.7×** — resident-проход перекрывает чтения).
  Масштаб на реальную модель: ~0.5 ms/layer host (memcpy-энкьюи + join) + ~0.28 ms/layer D2D ≈ **~30 ms/token** + скрытие miss-сталов → 115 → ~75 ms (~13-14 t/s, +45-55 %).
- Ограничение локальной валидации: gfx1151-специфичные ядра (`moe_v41_gate_up_wave_ptrs_kernel`, `moe_v41_down_wave_ptrs_kernel`) на gfx1150 не запускаются
  (harness гоняет generic ptrs-ядра). Но те же ядра уже используются TP-путём (`rocm/ds4_rocm_v41.cuh:1415/1436`) с таблицей указателей и mask=0x3f.
  Финальная проверка — A/B у пользователя при temp=0 (идентичность вывода) + t/s.

## 26.09 — A/B split-пути у пользователя: скорость подтверждена, различие вывода объяснено семплингом
- Логи `/home/neiron/work/ds-4.1-flash/tmp/tests/DS4_ROCM_V41_MOE_SPLIT-{0,1}.log` (последовательно, 23:19 и 00:07):
  decode avg **9.38 → 11.71 t/s** (пиковые чанки до 11.9-13), prefill 65 токенов **18.19 → 16.31 с** (все чанки быстрее, разрыв растёт),
  первый decode-чанк 6.57 → 7.95 t/s. Выводы **разные**, но оба когерентны (оба — правильные ответы на вопрос про QuickJS/UTF-8).
- Расхождение начинается **с первых токенов** (reasoning: «We need answer in Russian likely.» vs «Russian.») → различие присутствует с самого начала,
  не редкая поздняя гонка.
- **Harness с реальными размерностями V4.1** (`/home/neiron/work/ds-4.1-flash/tmp/v41split_big.c`): 384/6/5120/2304/5120, IQ2_XXS gate/up + Q2_K down,
  gfx1151-ядра **принудительно** на gfx1150 (тест-хук `DS4_ROCM_FORCE_GFX1151` в `rocm/ds4_rocm_common.cuh`, только на тестовом ПК, вне патча):
  compact vs split **побитово идентичны** — `mid` diff 0/13824, `out` diff 0/5120, max=0, за 10 итераций, включая холодный кэш (все 6 missing — это и есть prefill-случай);
  MoE-вызов hot (все resident): compact **2.790 ms** → split **1.067 ms (2.6×)**.
- **Причина различия вывода — не split, а семплинг**: env `DS4_SERVER_DEFAULT_TEMP` **в репозитории нет** (grep пустой — пользователь использовал
  несуществующую переменную); дефолт сервера `DS4_DEFAULT_TEMPERATURE = 1.0f` (`ds4.h:59`, `docs/SERVER.md:46` «temperature 1, top-p 1, min-p 0.05»);
  в thinking-режиме применяются thinking-дефолты (`top_k=0`, `min_p=0.05`), **явные параметры запроса имеют приоритет** (`ds4_server.c:13971`);
  **seed случаен на каждый запрос** (`random_bytes` из `/dev/urandom`, `ds4_server.c:97/13916`) → один и тот же запрос даёт разный текст на каждом вызове.
  Комментарий авторов в коде: «...an explicit request value (e.g. temperature 0 from a benchmark harness) must win, or the same greedy request returns different text on every call».
- **Гонки нет и в compact-пути**: `cuda_stream_selected_load` (runtime:4648) начинается с `cuda_stream_selected_reuse_wait` (4683) — план каждого слоя
  ждёт reuse-event предыдущего слоя (host-side), а event пишется в **stream 0** после ядер (`mark_inflight`, в регионе `!q2k_path` — `ds4_rocm_moe_launch.cuh:2161`).
  Значит слои сериализованы, таблица/слоты не перезаписываются под работающими ядрами → гонки нет ни в split, ни в compact. Это же объясняет,
  почему `QUEUE_LAYERS` не дал выигрыша: план и так ждёт завершения предыдущего слоя.
- Диагностика для пользователя: `--trace FILE` у сервера печатает `temperature: %.3f ... seed: %llu` и **raw request json** (`ds4_server.c:11997+`);
  корректный A/B — `"temperature": 0` **в теле запроса** (или `"seed": N` для воспроизводимого не-greedy прогона).

## Дополнение (логи сессии пользователя): env-переменные, которых нет
- В командах пользователя (сессии 24-27.09, включая A/B SPLIT-0/1) фигурируют `DS4_SERVER_DEFAULT_TEMP=1.0|0.0` и `DS4_SERVER_DEFAULT_TOP_P=0.95`.
  **Ни одной из этих переменных в репозитории нет** (grep пустой); дефолты сервера — `DS4_DEFAULT_TEMPERATURE 1.0f`, `DS4_DEFAULT_TOP_P 1.0f`,
  `DS4_DEFAULT_MIN_P 0.05f` (`ds4.h:59-61`), а в thinking-режиме явные параметры запроса приоритетны (`ds4_server.c:13971`).
  То есть оба прогона A/B фактически шли при temperature 1.0 со **случайным seed** → разный текст обязателен.
- Раньше пользователь сравнивал только t/s, поэтому различие текстов заметил только сейчас.
- Harness повторён 2-й раз: снова 0/10 расхождений (compact vs split, real dims, gfx1151-ядра) → результат детерминирован.

## 27-28.09 — KV-reuse при правке сообщения: полный пересчёт и путь через frontier-снимки
- Симптом пользователя (лог 0928 03:54): после остановки генерации и правки сообщения сервер печатает
  `slot 0: request reuses nothing; evicting checkpoint (270704 tokens, idle 46 s, protected)` и
  `live kv cache miss live=270704 prompt=269814 common=269783 vision=match reason=token-mismatch`,
  затем делает полный prefill 269814 токенов (chunk 1/131). Общий префикс = 269783, т.е. расходится всего 31 токен.
- Точка отказа найдена в коде: `slot_probe_reuse_locked` (ds4_server.c:11593) — единственный источник истины для роутинга и исполнения.
  Последний тир переиспользования — DS41 frontier-снимки (ds4.c:86618 `ds4_session_frontier_hint`): если есть снимок с pos <= common,
  сервер делает REUSE_MEMORY_REWIND (ds4_server.c:13533) — rewind на снимок + re-prefill суффикса. В нашем случае хинт вернул -1.
- Механика снимков: `ds41_gpu_graph.frontier_state` — кольцо `DS41_FRONTIER_PREFILL_SLOTS(8) + DS41_FRONTIER_DECODE_SLOTS(8)`,
  состояние слота = `(pos << 1) | has_logits`. Публикация с release-семантикой (ds41_frontier_publish), копии тензоров энкьюятся раньше слова.
  Захват: `ds41_frontier_capture_prefill` — на каждом chunk layer-major prefill (ds4.c:76860) и на sync-frontier с логитами (76884);
  `ds41_frontier_capture_decode` — каждые 64 decode-токена, но вызывается только в spec-ветке (ds4_server.c:14029), т.е. в обычном decode кольцо decode не заполняется.
  Восстановление: `ds41_frontier_restore` (40570) — берёт новейший снимок <= desired (exact-match требует logits), затем чистит снимки выше best_pos.
- Важно: сервер в batched-режиме синкает префиксы по 2048 токенов (`server_session_sync`, ds4_server.c:12445), поэтому каждый sync-call
  захватывает и chunk-снимки, и sync-frontier; кольцо prefill хранит последние 8 захватов.
- Диагностика добавлена: `DS4_ROCM_V41_FRONTIER_DEBUG=1` — логирует захваты (slot/pos/logits), скан хинта с дампом кольца,
  выбор при restore и решение пробы в сервере (`reuse probe frontier common=... hint=...`). Собрано и прогнано на тестовом ПК (все тесты зелёные).
- Тестовый ПК: пользователь подключил вторую GPU — Radeon RX 7800M (gfx1101), теперь две: gfx1101 (индекс 0) + gfx1150 (индекс 1).
  ROCm по умолчанию берёт индекс 0 → бинарь, собранный под gfx1150, падает с HIP-ассерцией StatCO::getStatFunc.
  Решение: запускать с `HIP_VISIBLE_DEVICES=1` (проверено: ds4-kernel-v41 PASS 118, ds4_test --server OK, ds4-eval/agent_test OK).

## 29.09 — Root cause полного пересчёта KV: сентинел -1 в ds4_session_frontier_hint
- Проверено (логи пользователя, два прогона короткой беседы — прерывание генерации → «продолжай»):
  - без debug: `live kv cache miss live=21073 prompt=21040 common=21033 vision=match reason=token-mismatch` → `slot 0: ... evicting checkpoint` → полный prefill 21040 с нуля.
  - с `DS4_ROCM_V41_FRONTIER_DEBUG=1`: кольцо `[2]=20480 [3]=20480+logits [4]=21045 [5]=21045+logits [6]=16384 [7]=16384+logits [8]=21109+logits`,
    `frontier hint desired=21043 best=-1 ready=1` → `reuse probe frontier common=21043 live=21131 prompt=21050 hint=-1` → evict + prefill 21050 (214.6 с, прерван по ^C).
- Причина в коде (`ds4_session_frontier_hint`, ds4.c:86650): `int best = -1; ... if (pos > (uint32_t)best) best = (int)pos;` →
  `(uint32_t)(-1) = UINT32_MAX` → сравнение `pos > 4294967295` никогда не истинно → `best` не обновляется → хинт всегда -1.
- Результат: фикс `if ((int)pos > best) best = (int)pos;`; `make ds4.o` собрался чисто (44.6 с). Соседний `ds41_frontier_restore` (ds4.c:40600) сравнивает корректно
  (`best >= 0 && pos <= best_pos`); другие сентинелы `best=-1` в файле (45699, 68488, 68576) используют `best < 0 || ...` — корректны. Баг изолирован в хинте.
- Апстрим-проверка: `raw.githubusercontent.com/antirez/ds4/main/{ds4.c,ds4.h,ds4_server.c}` — 0 совпадений `ds41_frontier*`/`frontier_state`/`frontier_hint`;
  `git show origin/main:ds4.c` и `origin/feat/rocm-deepseek41-halo` — 0; `git log -S "ds4_session_frontier_hint"` — функция внесена коммитом 7e0276f
  (уже запушен в origin/feat/rocm-deepseek41-halo-fix-kv-cache); база форка — upstream 0aaea5a.
- Вывод: баг fork-local (регресс в 7e0276f), не апстримный → фиксим на ветке. Ожидаемое поведение после фикса: desired=21043 → best=20480 →
  `REUSE_MEMORY_REWIND` → replay суффикса 20480..21043 (563 токена) вместо evict+prefill 21050.
