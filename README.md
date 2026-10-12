# velo-docker

把上游 `sf-stav/veloGB10` 的預編譯 release 重包成 Docker image，推送到 GHCR。首版只支援單機，不支援 TP。

## 用法

```bash
docker run -d --name velo --gpus all -p 9000:9000 \
  -v ~/models:/models:ro \
  ghcr.io/<owner>/velo-docker:latest \
  --server --model-dir /models/<model> --port 9000 --max-batch 1
```

说明：

- image 內是上游 tarball 原樣解包，`WORKDIR /opt/velogb10`，`ENTRYPOINT ["./gb10_inference"]`，直接把 server 參數接在後面
- 模型不進 image，用 `-v` 掛載
- 一定要加 `--gpus all`，`libcuda.so.1` 由 host driver 提供
- 只開 `9000`，TP 的 `29500` 首版不支援

## tag

- `v0.7.x`：對應上游 release tag
- `latest`：上游最新版

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

binary 約 30MB，PTX 約 60MB，CUDA runtime 依賴約 900MB，其中 `libcublasLt` 一個就佔 600MB。這是正確執行的下限，砍不掉。實測 image 約 1.16GB。base 沒用整套 `nvidia/cuda:runtime`，只拷 `cudart`、`cublas`、`cublasLt`、`curand`、`nvrtc`，省下約 1GB 用不到的 `cufft`、`cusolver`、`cusparse`。
