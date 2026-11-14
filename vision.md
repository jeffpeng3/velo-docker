# Vision：單一 request 的 image token 上限 16384 造成多圖請求失敗

狀態：open
影響範圍：veloGB10 上游 EXL3 server 的 vision 路徑（`VIS_MROPE_ROWS`）
本 repo 的 `docker-compose.yml` 目前沒有對應參數，只能靠 `--image-max-edge` 規避
v0.7.9 已確認常數未變，仍是 16384；下列行號已對到 v0.7.9 tag

## 現象

送多張圖片的 chat request 被拒：

```
error: bad request: 16808 image tokens exceed the per-request limit of 16384
```

- 圖片都不大，但張數多
- 純文字請求不受影響
- 模型：`jeffpeng3/Swift-1.5-Qwen3.8-Flash-Next-EXL3-3.05bpw-heretic`

## 根因

上限是上游的編譯期常數，統計的是**同一個 request 內所有圖片的 image token 總和**，不是單張上限。

| 位置 | 內容 |
|---|---|
| `src/exl3_serve.rs:4435` | `let n_img: usize = req.image_spans.iter().map(|s| s.num_tokens).sum();` |
| `src/exl3_serve.rs:4437` | `anyhow::ensure!(n_img <= VIS_MROPE_ROWS, "bad request: {n_img} image tokens exceed the per-request limit of {}")` |
| `src/exl3_forward.rs:2716` | `pub const VIS_MROPE_ROWS: usize = 16384;` |

`16384 = 16777216 px ÷ 1024 px/token`，也就是模型**單張圖**的 token 上限。上游把每個 slot 的 image-token RoPE 表直接開成單張圖的最大容量，於是單張圖的上限變成整個 request 的總量上限。

## 每張圖佔多少 image token

由 `src/vision_preproc.rs` 的 `smart_resize` 決定，32 px = 1 token，`min_pixels = 65536`、`max_pixels = 16777216`，再加上 `--image-max-edge`（預設 1024）限制長邊。

`min_pixels = 65536` 是關鍵：小圖會被主動放大到 256×256，所以不管圖多小，每張至少 64 個 token。

| 縮放後尺寸 | token/張 | 16384 可放張數 |
|---|---|---|
| 任何小於 256×256（被放大到 256×256） | 64 | 256 |
| 512×512 | 256 | 64 |
| 1024×576（1080p 16:9，長邊被壓到 1024） | 576 | 28 |
| 1024×1024（預設長邊上限） | 1024 | 16 |

`16808` 約等於 16 張滿邊圖，或 29 張 1080p 截圖。

## 為什麼需要這張表

- 文字 token 的位置只有一個數字，全模型共用一張 `cos_tab[max_pos][2*rdim]`，任何 lane、任何 request 都以位置索引讀同一張表。
- 圖片 token 的位置是 `(t, h, w)` 三元組，interleaved mrope 會依維度索引混用不同軸的值（`mrope_section = [11, 11, 10]`），值無法由單一位置索引推出，只能逐 token 算好存起來。
- 而且每個 slot 的圖片不同，所以表必須 per-slot。

實作（`src/exl3_forward.rs:2610-2618`）：

| buffer | 內容 |
|---|---|
| `rope_map[width][max_pos]` i32 | prompt 位置 `p` → `cos_tab[p]` 或 `mrope_tab` 某列 |
| `mrope_tab[width * VIS_MROPE_ROWS][2*rdim]` | slot 的 image-token cos/sin 列，request 進來時由 host 算好上傳 |
| `qm_cos` / `qm_sin[width * VIS_MROPE_ROWS][rdim]` | QSA indexer pool 過的 key 用同一份行 |

kernel 端只在 `xq_cos_gather_v` 做 `idx - max_pos` 的位移，所以列偏移由 host 寫進 map，改常數不需要動 kernel。

每列成本 = `(2*rdim + rdim + rdim) * 4 bytes`，`rdim = int(256 * 0.25) = 64`，即 1 KB/列。這張表在模型載入時就配置好，不管有沒有圖片都常駐。

## 目前指令的記憶體占用

`docker-compose.yml` 的參數：`--max-batch=2`、`--max-seq-len=524288`、`--kv-cache=q4k`、`--prefix-ckpt-mem-gb=4`、`--ple-ram=ssd`。

每 lane 的 KV + recurrent state（`fit_parts`，`src/exl3_serve.rs:6612`）：

```
per_tok = 12 層 full_attention × 2 × (2×256) × 0.5625   = 6912 B
        + 12 × 1 × 128 × 2 (indexer keys)                = 3072 B
        + 1 × 2 × 512 × 2 (MTP head)                     = 2048 B
        = 12032 B/token
per_lane = (12032 × 524289 + 116 MB state) × 1.15 ≈ 7.39 GB
```

整體 fit check：

| 項目 | 現在（16384 列） | 改成 131072 列 |
|---|---|---|
| weights 52.3 GB + vision tower 0.9 GB | 53.2 GB | 53.2 GB |
| n-gram table 32.6 GB | 在 SSD（`--ple-ram=ssd`） | 同 |
| 2 lanes × KV/state（`max_pos = 524289`） | 14.8 GB | 14.8 GB |
| `fixed`（scratch、graphs、`--prefix-ckpt-mem-gb=4`） | 19 GB | 19 GB |
| floor（`mem-watchdog-gb` 取 `max(5, 20)`） | 20 GB | 20 GB |
| mrope 表（2 lanes × 列數 × 1 KB） | 33.6 MB | 268.4 MB |
| **合計** | **107.0 GB** | **107.3 GB** |

改常數的常駐成本是 **+235 MB**（0.18% of 128 GB），開機時多一次 134 MB 的 host `vec![0f32; ...]`，上傳後釋放。

重點：`--max-seq-len=524288` 已經讓 KV 預先為 524289 個位置付掉 7.39 GB/lane，image token 從 16k 增加到 131k 不會再多配置 KV，只是用掉已付的位置（131072 × 12032 B ≈ 1.58 GB，已含在內）。

改常數後真正的成本是單筆 request 的 embedding splice：

| 131072 image tokens（hidden 2560） | 大小 |
|---|---|
| `image_embeds` host f32 | 1.34 GB |
| `rows` 再複製一份 host | 1.34 GB |
| `vis_rows_upload` 後 device f32 | 1.34 GB |

## 選項

### A. 調 `--image-max-edge`（不改程式，先做這個）

- `--image-max-edge=512`：每張 ≤ 256 token，現有 16384 可放 64 張
- `--image-max-edge=256`：每張 ≤ 64 token（等於 `min_pixels` 下限），可放 256 張
- 代價：OCR、細字、小元件辨識會退化，下限卡在 64 token/張

### B. 把 `VIS_MROPE_ROWS` 改大（改上游常數，需重編）

- `131072`：單 request 上限 131k image token
- 常駐 +235 MB（本指令），安全性沒問題：map 是 i32，`max_pos < 2^20` 的位元打包限制不受影響
- 只解除第一道閘，還會撞到 `--max-seq-len`（要 ≥ image token + 文字）、embedding splice 的 4 GB 級暫存，以及 vision tower 的執行時間

### C. 按 request 動態配置（最省記憶體，工程量大）

用 `n_img` 決定列數，不再固定預留整張表。要改 buffer 生命週期與 CUDA graph 參數的綁定，風險高於 A、B。

## 建議

1. 先在 compose 加 `--image-max-edge=512`，確認多圖場景可用
2. 若真的需要一次塞 100k 等級的 image token，走 B，並同步檢查 `--max-seq-len` 與單筆 request 的記憶體峰值
3. C 留給上游，不在本 repo 處理

## 驗收

- 送 N 張 1024×1024 圖片（N 依選項推得的上限）不再出現 `image tokens exceed the per-request limit`
- 純文字請求的回應與改動前一致（`xq_cos_gather_v` 在 `rplen = rdelta = 0` 時必須走 `cos_tab[p]`）
- 開機 fit check 的合計與上表相符（誤差在 `fixed` 的量測噪聲內）

## 上游參考位置

- `src/exl3_serve.rs:4435-4438` 上限檢查
- `src/exl3_forward.rs:2716` `VIS_MROPE_ROWS`
- `src/exl3_forward.rs:2610-2618` buffer 配置說明
- `src/exl3_forward.rs:9002`、`9047-9048` 表實際配置
- `src/exl3_forward.rs:25234`、`25251-25255` 每 request 上傳
- `src/vision_preproc.rs:44-48`、`54-76` `smart_resize` 參數
- `kernels/exl3_bench.cu:4850` `xq_cos_gather_v`
- `src/exl3_serve.rs:6612` `fit_parts` 記憶體估算
