# Текущее состояние (обновлено 29.09)

## Задача
Оптимизация производительности DeepSeek V4.1 Flash (Q2, SSD streaming) на Strix Halo gfx1151 / ROCm 7.2.
Базовый лог пользователя: prefill 65 токенов = 17.5-17.8 с, decode ~9.33 t/s (при hit rate кэша экспертов 98.1 %).

## Сделано (сессия 1, 24-25.09)
1. ROCm 7.2: все версионно-зашитые тюнинг-пути hipBLASLt/rocBLAS отключены (на тестовом ПК hipBLASLt 1.2.2) → prefill на fallback-ядрах.
2. Реализовано (opt-in, env):
   - `rocm/ds4_rocm_hipblaslt.cuh`: версионно-независимый Lt-heuristic план (`DS4_ROCM_HIPBLASLT_PREFILL_HEURISTIC=1`), состояние 2 в `g_hipblaslt_prefill_state`.
   - `rocm/ds4_rocm_matmul.cuh`: гейт Lt-prefill принимает heuristic; селектор `DS4_ROCM_F16_LT_PREFILL`.
   - `rocm/ds4_rocm_v41.cuh`: Engram Lt-план — heuristic fallback; тот же селектор.
   - `rocm/ds4_rocm_runtime.cuh`: ручка `DS4_ROCM_STREAM_LAYER_CHUNK_MB` (1..64, дефолт 32) + кламп числа job-ов.
   - `ds4.c`: профайлер экспертов разрешён на ROCm для DeepSeek41; запись роутинга в `ds41_stream_selected_begin`; сид resident-кэша из hotlist-файла
     на старте движка V4.1 (`DS4_ROCM_STREAMING_EXPERT_HOTLIST`, disable-флаг `DS4_ROCM_V41_DISABLE_STREAMING_SEED_BEFORE_PREFILL`).
   - `ds4_help.c` / `ds4_server.c`: `--expert-profile FILE` доступен и в ds4-server.
3. Замерено на gfx1150 (та же ROCm 7.2): Lt-heuristic 1.63×/1.70×/1.97× быстрее WMMA на фигурах prefill и 2.78× на Engram (467 ms против 1296.8 ms).

## Сделано (сессия 2, 25.09) — после замеров пользователя
4. Замеры пользователя (два прогона, кэш 94 и 95 GB + `DS4_ROCM_STREAM_CACHE_STATS/LAYER_STATS`): hit rate кэша экспертов **98.1 %** (4238468/4320720),
   с SSD ~0.43 GB/s → decode НЕ storage-bound. Конфиг-ручки (кэш 94/95/96 GB, `FREE_RESERVE_GB=6`, `READ_WORKERS=24`, `LAYER_CHUNK_MB=64`) не влияют
   (decode ~9.33-9.35 t/s, prefill ~17.7 с). См. discoveries.md §6.
5. Профайл экспертов снят (40 слоёв, 3138240 выборов, 523040 записей): симуляция per-layer LRU cap 256 → 98.56 %, cap 384 → 99.56 % → рантайм (~234 слота/слой)
   уже на кривой, политика почти оптимальна; ёмкость даёт максимум +1.5 пп. Файлы: `tmp/v41_profile.json`, `tmp/v41_hotlist.txt` (на dev-ПК).
6. Реализовано (в рабочей копии; патч `/home/neiron/work/ds-4.1-flash/rocm72-perf.patch`, 7 файлов +264/-33):
   - `DS4_ROCM_V41_DECODE_PROFILE=1` (+ `DS4_ROCM_V41_DECODE_PROFILE_LAYER=1`) — профилировщик фаз decode: одна строка на токен
     `host[pre/moe/post]`, `moe_sub[read/plan/launch]`, `gpu` (drain) и `rest`; per-layer строки при `_LAYER=1`.
   - `DS4_ROCM_V41_STREAM_QUEUE_LAYERS=1` — под `--ssd-streaming` разрешает `queue_layers` (нет `cudaDeviceSynchronize` на каждый слой; host-encode перекрывается с GPU).
     Безопасность: reuse-event (после routed MoE на default-stream) уже сериализует compact-кэш, слоты resident-кэша и slot_ids против MoE предыдущего слоя;
     drain на il==13 (перезапись Engram) и на последнем слое сохранён.
7. Проверено на тестовом ПК: сборка gfx1150 без warning, `make test-deepseek41-rocm` PASS 118 фигур, `make test-rocm` RC=0 (все сводки PASS), кросс-сборка gfx1151 чистая.
   `ds4.c` и rocm/*.cuh синкнуты (md5 совпадают); патч тоже на тестовом ПК.
8. Замеры RAID-0 пользователем (fio): seq 32 MiB ≈ 7400-7466 MiB/s (psync и io_uring depth 8 — одинаково), randread 4k 134k IOPS / 525 MiB/s,
   depth-1 latency 173 µs → **реальный потолок ≈ 7.5 GiB/s (8.3 GB/s), а не 13.5 GB/s**. Prefill первого запроса (152.8 GB / 17.66 с = 8.65 GB/s) уже
   на пределе устройства → тюнинг загрузчика бесполезен (см. discoveries §8).
9. Найдено: prefill коротких промптов идёт **token-major через decode-путь** (в логе нет `V4.1 layer reuse`, `prepares=0`; `prefill chunk N/65` — по токену):
   `ds41_prefill_count` на ROCm возвращает 1 при `remaining < 256` (ds4.c:42090). Append <256 токенов тёплого кэша ≈ 107 мс/токен (65 → ~7 с, 255 → ~27 с),
   тогда как sweep ≈ 7.2 с (читает только missing-эксперты). Добавлена ручка `DS4_ROCM_V41_PREFILL_MIN_TOKENS` (2..4096, default 256; поведение по умолчанию не меняется).
   Проверено harness'ом `/home/neiron/work/ds-4.1-flash/tmp/v41count.c` на gfx1150 (default/min=64/min=4096/min=1).

## Сделано (сессия 3, 25.09) — по результатам фазового профиля и fio
10. Профиль фаз decode (логи пользователя в `/home/neiron/work/ds-4.1-flash/tmp/tests/`): steady state **116.0 ms/token**, host 66.7 (pre 1.8, moe 64.8, post 0.1),
    moe_sub read 43.3 / plan 1.0 / launch 19.9, gpu(drain) 40.5, rest 8.8. С `QUEUE_LAYERS=1`: gpu 40.5 → 2.1, но read → 80.7 → **итог 115.7 ms** (перекрытие не даёт выигрыша — конвейер GPU-bound).
    Всплески `launch` до 55 ms на токенах с miss → miss-чтения на критическом пути (~15-20 ms/token).
11. Трафик весов на токен ≈ 11.3 GB (attention+router 4.43 GB, routed 2.26, compaction D2D 2.20, shared 1.44, engram 0.59, head 0.35) → ~141 GB/s эффективной полосы
    при пике ~230; **compaction = 19 % трафика**.
12. Реализовано: `DS4_ROCM_V41_MOE_SPLIT=1` — включён существовавший, но отключённый (`split_selected = 0`) one-token resident/missing split
    (`cuda_stream_selected_apply_split` + ptrs-ядра gate/up/down вместо compact-таблицы); условие `split_supported` ослаблено до `(resident|missing) != 0`.
13. Проверено harness'ом `/home/neiron/work/ds-4.1-flash/tmp/v41split.c` на gfx1150: **0 расхождений** с compact-путём за 40 итераций (hot/mixed/cold, эвикции);
    MoE-вызов (план+запуск+drain): hot 0.120 → 0.065 ms (**1.85×**), cold 0.181 → 0.067 ms (**2.7×**). Ожидаемо на цели ~30 ms/token + скрытие miss-сталов → ~13-14 t/s.
14. Замер пользователя по prefill-ручке (5 прогонов с drop_caches): sweep для холодных 65 токенов = **24.16 с**, token-major = **17.74-17.80 с** → sweep медленнее на холоде;
    ручку ниже 128 для первого запроса не ставить.
15. Тесты после всех правок (gfx1150): `make test-deepseek41-rocm` PASS 118, `make test-rocm` RC=0, кросс-сборка gfx1151 чистая; патч 584 строки (8 файлов +307/-38).
    Урок: `--offload-arch` при линковке harness'а должен совпадать с arch объектов (иначе HIP роняет symbol lookup); после `make test-rocm` (дефолт gfx1151)
    объекты надо пересобрать `-B` под нужный arch.

## Сделано (сессия 4, 26.09) — A/B пользователя и объяснение «разного результата»
16. Логи пользователя `tmp/tests/DS4_ROCM_V41_MOE_SPLIT-{0,1}.log`: decode avg **9.38 → 11.71 t/s** (пиковые чанки до ~13), prefill 65 токенов **18.19 → 16.31 с**,
    первый decode-чанк 6.57 → 7.95 t/s. Скорость подтверждена (t/s считается на токен, длина ответа на него не влияет).
17. Выводы разные, но оба когерентны; расхождение начинается **с первых токенов** (reasoning «...Russian likely.» vs «...Russian.») → различие не от поздней редкой гонки.
18. **Harness с реальными dims V4.1** `tmp/v41split_big.c` (384/6/5120/2304/5120, IQ2_XXS+Q2_K, gfx1151-ядра принудительно на gfx1150 через тест-хук `DS4_ROCM_FORCE_GFX1151` в `rocm/ds4_rocm_common.cuh`, только на тестовом ПК, вне патча):
    compact vs split **побитово идентичны** (`mid` 0/13824, `out` 0/5120, max=0) за 10 итераций, включая холодный кэш (все 6 missing = prefill-случай); MoE hot 2.790 → **1.067 ms (2.6×)**.
19. Причина различия вывода — **семплинг, не split**: env `DS4_SERVER_DEFAULT_TEMP` в репозитории нет; дефолт сервера temp=1.0 (`DS4_DEFAULT_TEMPERATURE`, `docs/SERVER.md:46`),
    в thinking-режиме thinking-дефолты (`min_p=0.05`), явные параметры запроса приоритетны (`ds4_server.c:13971`), **seed случаен на каждый запрос** (`/dev/urandom`).
    Гонки нет ни в одном пути: план слоя (`cuda_stream_selected_load`, runtime:4648) ждёт reuse-event предыдущего слоя (`reuse_wait`, 4683), event пишется в stream 0 после ядер.
    Диагностика: `--trace FILE` печатает `temperature:`/`seed:` и raw request json.

## Сделано (сессия 5, 27-28.09) — split дефолтом, диагностика KV-reuse, вторая GPU на тестовом ПК
20. Пользователь подтвердил A/B при temp=0: **вывод совпадает**, скорость растёт с `DS4_ROCM_V41_MOE_SPLIT=1` → split корректен end-to-end.
21. `DS4_ROCM_V41_MOE_SPLIT` теперь **дефолт на gfx1151** (через `ds4_rocm_gfx1151_flag`): unset = включён, `=0` = compact-путь, `=1` = включить на других ROCm.
22. Добавлена диагностика frontier-кольца (`DS4_ROCM_V41_FRONTIER_DEBUG=1`): захваты снимков, дамп кольца при скане хинта, выбор restore, решение пробы в сервере.
23. Тестовый ПК: пользователь подключил RX 7800M (gfx1101) — теперь две GPU (gfx1101 = индекс 0, gfx1150 = индекс 1);
    ROCm берёт индекс 0 → бинарь под gfx1150 падает с HIP-ассерцией. **Все тесты запускать с `HIP_VISIBLE_DEVICES=1`**
    (проверено: `ds4-kernel-v41` PASS 118, `ds4_test --server` OK, `ds4-eval`/`ds4_agent_test` OK).
24. Патч перегенерирован против pristine-чекаута ветки `7e0276f` (git clone hitman249/ds4): **678 строк, 8 файлов, +357/-40**;
    проверено `patch -p1` на чистом чекауте — результат побитово совпадает с рабочей копией (пересобран с фиксом hint 29.09 — см. п.31).

## Сделано (сессия 6, 29.09) — root-cause полного пересчёта KV найден и исправлен
25. Пользователь воспроизвёл полный пересчёт на **короткой** беседе с `DS4_ROCM_V41_FRONTIER_DEBUG=1`: после прерывания генерации и «продолжай»
    лог показал кольцо со снимками `[2]=20480`, `[3]=20480+logits` (≤ common=21043), но `frontier hint desired=21043 best=-1` → `hint=-1` →
    `evicting checkpoint` → полный prefill 21050 токенов (214.6 с, прерван). Без debug-флага симптом тот же (evict + prefill с нуля).
26. Root cause: `ds4_session_frontier_hint` (ds4.c:86650) — `if (pos > (uint32_t)best) best = (int)pos;` при сентинеле `best = -1`:
    `(uint32_t)(-1) = UINT32_MAX` → `pos > 4294967295` никогда не истинно → `best` навсегда `-1`, хинт всегда возвращает -1.
27. Фикс: `if ((int)pos > best) best = (int)pos;` (signed-сравнение). Сборка `make ds4.o` чистая (44.6 с; `DS4_HAS_DEEPSEEK41_GPU` включён по умолчанию — ds4.c:51, т.к. `DS4_NO_GPU` не задан).
28. Апстрим проверен — баг **не апстримный**: в antirez/ds4 main @ 0aaea5a (raw fetch ds4.c/ds4.h/ds4_server.c), а также в origin/main и origin/feat/rocm-deepseek41-halo
    нет ни одного символа frontier-кольца/хинта. Механизм внесён fork-коммитом `7e0276f` (a.dorokhin, 22.09), уже запушен в origin/feat/rocm-deepseek41-halo-fix-kv-cache → фиксим на ветке, upstream PR не нужен.
29. Ожидание после фикса: для desired=21043 хинт вернёт 20480 → `REUSE_MEMORY_REWIND` → rewind + replay суффикса 20480..21043 (563 токена) вместо evict + prefill 21050.
30. Соседний `ds41_frontier_restore` (ds4.c:40600) сравнивает корректно (`best >= 0 && pos <= best_pos`); другие сентинелы `best=-1` (45699, 68488, 68576) — `best < 0 || ...` — корректны. Баг изолирован в хинте.
31. Патч пересобран с фиксом и отладочной инструментацией: **685 строк, 8 файлов, +358/-41** (`/home/neiron/work/ds-4.1-flash/rocm72-perf.patch`).
    Проверено: `git apply --check` OK, `patch -p1 --dry-run` OK, после apply sha256 всех 8 файлов совпадают с рабочей копией; pristine возвращён в чистое состояние.

## В работе / следующий шаг
- Пользователь: проверить фикс на цели — короткая беседа, прерывание → «продолжай», с `DS4_ROCM_V41_FRONTIER_DEBUG=1`.
  Ожидание: `frontier hint desired=... best=<снимок>` (не -1), `reuse probe frontier ... hint=<pos>` → rewind+replay суффикса, **без** `evicting checkpoint` и полного prefill.
- Закоммитить фикс (`ds4.c:86663`) отдельно от отладочной инструментации (патч уже пересобран с фиксом).
- Затем: attention-рычаг (в профиле ~43.3 ms/token при ~4.4 GB → ~102 GB/s против ~146 у MoE-части).

## Открытые вопросы
- Остаются ли miss хинта после фикса (снимок старше common / кольцо перезаписано) и как часто нужны decode-снимки в обычном (не spec) decode.
- Точка безубыточности sweep-пути prefill на малых count на тёплом кэше (юнит-тестами не покрыт) — ждёт замера на цели.

## Изменённые файлы (патч: 8 файлов +358/-41, 685 строк)
- ds4.c (+~230: профайлер/hotlist/сид + фазовый профилировщик + queue_layers + порог prefill + диагностика frontier), ds4_help.c, ds4_server.c (+~15)
- ds4.c:86663 (frontier hint): фикс сентинела `(uint32_t)best` → `(int)pos > best` — в рабочем дереве и в патче (пересобран 29.09).
- rocm/ds4_rocm_hipblaslt.cuh (+43), rocm/ds4_rocm_matmul.cuh, rocm/ds4_rocm_moe_launch.cuh (split, теперь дефолт на gfx1151), rocm/ds4_rocm_runtime.cuh (+21), rocm/ds4_rocm_v41.cuh (+23)
- Патч: `/home/neiron/work/ds-4.1-flash/rocm72-perf.patch` (685 строк, 8 файлов, +358/-41; перегенерирован против pristine `git@github.com:hitman249/ds4.git` ветка `7e0276f`, включает фикс frontier hint; тест-хука `DS4_ROCM_FORCE_GFX1151` в нём нет)
- .agent/*.md (память), misc/rocm72-tuning.md (заметка, gitignored); harness'ы вне репо: tmp/v41count.c, tmp/v41split.c, tmp/v41split_big.c
- Тест-хук `DS4_ROCM_FORCE_GFX1151` живёт только в копии на тестовом ПК (нужен harness'у для gfx1151-ядер на gfx1150); при синке rocm/ его надо накладывать заново.

## Сделано (сессия 7, 01.10) — вливание kyuz0 и evolver в fix-kv-cache
Задача: влить последние изменения kyuz0/feat/rocm-deepseek41-halo и полезные изменения evolver/strix-halo-v41-dspark-pr.
Оба внешних репозитория добавлены как remote'ы: `kyuz0` (https://github.com/kyuz0/ds4.git), `evolver` (https://github.com/evolvemarketingitalia/ds4.git).
Общий предок обеих ветвей и нашей — cab7369 (origin/feat/rocm-deepseek41-halo tip).
- kyuz0 tip d32febf: 16 коммитов (DSpark resident на ROCm + конвертер support-GGUF + prefill-тюнинг: F16/Q8 projections, Gufo attention, fused prefill MoE, tiny HC/router).
- evolver tip 78b3de1: 13 коммитов (V4.1 DSpark поверх SSD-streaming — порт antirez/ds4 PR #1073, bf16-rounding fusion, DS4_SERVER_PREFILL_QUANTUM, split-детерминизм, KEEP_PAGES, конвертер FP8/markov-имена).
- Наш dcc586a: split экспертов (дефолт gfx1151) + ROCm 7.2 perf-патч + фикс frontier hint.

### kyuz0 merge (merge-probe-kyuz0, коммит 2d006e9)
- Конфликты только в ds4.c (3 хунка): free/reset graph (draft+frontier) и печать бюджета (ds41_engine_graph_bytes + наш hotlist-сид). Оба разрешены «оставить оба».
- Сборка на тестовом ПК (gfx1150, HIP_VISIBLE_DEVICES=1): `make strix-halo ROCM_ARCH=gfx1150` — чисто (до правок evolver-merge).

### evolver merge (merge-probe-evolver) — гибрид
Ключевой факт: у evolver и kyuz0 ДВЕ независимые реализации V4.1 DSpark drafting (одна resident-only у kyuz0, другая streaming-capable у evolver). Автомердж дал ДУБЛИ всех ds41_draft_* функций (киуз0-блок в середине файла, evolver-блок в конце) — удалил evolver-дубликаты (370 строк), оставив kyuz0-код как базу.
- Разрешение конфликтов: ds4.c (23) → ours (kyuz0), ds4_help.c (1) → ours, moe_launch.cuh (2) → ours (наша split-политика DS4_ROCM_V41_MOE_SPLIT, дефолт gfx1151) с сохранением evolver-фикса SPLIT_COMPACT_WAIT; v41.cuh (2) → kyuz0-пути первыми, evolver-rows-ядра остаются для непокрытых форм (row 7 / rows 7-8).
- Автомердж принёс: staging-ring + compute-stream uploads + KEEP_PAGES + compaction-wait (runtime.cuh), evolver-streaming-блок в ds41_moe_batch, PREFILL_QUANTUM + speculation в ds4_server.c, prefetch-подсистему, rows-ядра, конвертерные имена markov_head.embed/head + FP8 block size из scale shape.
- Исправлены дубликаты автомерджа: struct-члены (n_expert в summary/weights, draft в graph, ds41_dspark в engine), две декларации/реализации ds4_gpu_dsv41_markov_chain (оставлена kyuz0-версия с transposed_head), short-prefill (возврат kyuz0-поведения на ROCm: short_count=0 + guard).
- Включён opt-in DSpark-over-SSD: `DS4_ROCM_DSPARK_STREAMING=1` — снимает guard `g->streaming` в ds41_draft_init и допускает `--dspark` + `--ssd-streaming` в гейте движка; по умолчанию поведение kyuz0 (resident-only) не меняется.

### Проверки
- Сборка gfx1150 после каждого шага (итеративно устраняются ошибки автомерджа). Тесты (test-deepseek41-rocm / test-rocm) — после финальной сборки.
- E2E DSpark на тестовом ПК невозможен (моделей V4.1 нет) — финальный тест делает пользователь на цели.

### Изменённые ветки
- `merge-probe-kyuz0` = dcc586a + kyuz0 merge (2d006e9).
- `merge-probe-evolver` = merge-probe-kyuz0 + evolver merge (в работе).
- Цель: перенести оба merge-коммита в feat/rocm-deepseek41-halo-fix-kv-cache после зелёных тестов.

### Результаты проверок (01.10, продолжение)
- Сборка gfx1150 (HIP_VISIBLE_DEVICES=0): `make strix-halo ROCM_ARCH=gfx1150` — 0 errors, 0 warnings.
- `make test-deepseek41-rocm` c `HIP_VISIBLE_DEVICES=0 DS4_ROCM_FORCE_GFX1151=1` — **PASS 188** изолированных фигур (включая DSpark-ядра kyuz0: hc_mean/router/markov).
- Важно: без `DS4_ROCM_FORCE_GFX1151=1` тесты падают (native-gfx1150 fallback ≠ gfx1151-оракулы).
- На тестовом ПК iGPU теперь HIP-индекс 0 (RX 7800M в HIP не виден): `HIP_VISIBLE_DEVICES=1` → `no ROCm-capable device is detected`.
- Найденные и исправленные проблемы гибрида (evolver merge):
  1. Автомердж дал дубли ds41_draft_* (удалён evolver-блок 370 строк), дубли struct-членов (n_expert×2, draft×2, ds41_dspark×2), дубли ds4_gpu_dsv41_markov_chain/hc_mean (оставлены kyuz0-версии), дубли short-prefill (возврат kyuz0-поведения на ROCm).
  2. `attention-output` (one-row, rows=1): evolver-путь с fused bf16-округлением даёт другую арифметику, чем kyuz0-оракул теста → восстановлены kyuz0-ядра `v41_grouped_q8_f32_blocks4_kernel`/`v41_q8_f32_blocks4_kernel` + всегда `ds4_gpu_dsv41_quantize(low)`; fused-вариант остался только для rows 7..8.
  3. `ds41_dspark_streaming_enabled()` перенесён в безусловную секцию (используется гейтом вне `#ifdef DS4_HAS_DEEPSEEK41_GPU`).
