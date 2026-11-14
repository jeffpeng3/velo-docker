# velo-docker

把上游 `sf-stav/veloGB10` 的預編譯 release 重包成 Docker image，推送到 GHCR。首版只支援單機，不支援 TP。

## 用法

```bash
docker run -d --name velo --gpus all -p 9000:9000 \
  -v ~/models:/models:ro \
  ghcr.io/<owner>/velo-docker:latest \
  --server --model-dir /models/<model> --port 9000 --max-batch 1
```

說明：

- image 內是上游 tarball 原樣解包，`WORKDIR /opt/velogb10`，`ENTRYPOINT ["./gb10_inference"]`，直接把 server 參數接在後面
- 模型不進 image，用 `-v` 掛載
- 一定要加 `--gpus all`，`libcuda.so.1` 由 host driver 提供
- `--runtime nvidia` 只選 runtime，沒有 device request 就不會注入 driver libs，容器會噴 `libcuda.so.1: cannot open shared object file`。Compose 用 `gpus: all`，不要用舊寫法 `runtime: nvidia`
- 測 GPU 要覆蓋 entrypoint：`docker run --rm --gpus all --entrypoint nvidia-smi <image>`。直接接 `nvidia-smi` 會被當成 `gb10_inference` 的參數
- 只開 `9000`，TP 的 `29500` 首版不支援
- v0.7.9 起 `--kv-cache q4k` 已在上游，不需要自建 fork image。容器內不要用 `--update`，image 也沒有 `curl`；開機版號檢查用 `--no-update-check` 關掉
- 自訂 chat template 用 `--chat-template <path>`，不用再覆寫 HF cache 裡的 blob
- `--model-dir` 用 HF repo id 時必須寫成 `--model-dir <id>` 兩個參數，`--model-dir=<id>` 不會觸發解析。`--hf-cache` 要指到含 `models--*` 的 `hub` 目錄
- `--gpu-wait=hybrid` 在 v0.7.9 的 EXL3 server 尚未生效，`velogb10_gpu_wait_hybrid_waits_total` 會是 0；留著是為了上游接上後自動生效。每次升版用 hybrid 開一次，counter 大於 0 才需要重新量測

## tag

- `v0.7.x`：對應上游 release tag
- `latest`：上游最新版
- `v0.7.8-q4k`：舊的自建 q4k image，v0.7.9 起已由上游取代，只留作 rollback

## CI

`.github/workflows/sync-and-push.yml` 每天 UTC 02:00 查上游最新版，GHCR 已有就跳過，沒有就下載、驗 SHA、建 `linux/arm64`、推送。舊版回補用 Actions 頁的 Run workflow 手填 `version`。

## 本地建置

```bash
scripts/fetch-velo.sh v0.7.4 ./.fetch
rm -rf ./velogb10 && mv ./.fetch/velogb10 ./velogb10
docker build -t velo-docker:v0.7.4 .
docker run --rm --gpus all velo-docker:v0.7.4 --help | head -20
```

## 容量

v0.7.5+ 的 EMB1 tarball 把 kernels 嵌進 binary，沒有外部 PTX，binary 會比舊版大；v0.7.9 的 binary 是 420MB，壓縮後 tarball 81MB。v0.7.4 以前是 binary 約 30MB 加 PTX 約 60MB。CUDA runtime 依賴約 900MB，其中 `libcublasLt` 一個就佔 600MB。這是正確執行的下限，砍不掉。base 沒用整套 `nvidia/cuda:runtime`，只拷 `cudart`、`cublas`、`cublasLt`、`curand`、`nvrtc`，省下約 1GB 用不到的 `cufft`、`cusolver`、`cusparse`。
