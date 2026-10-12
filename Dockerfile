# velo-docker：把上游預編譯 tarball 重包成最小 runtime image。
# 不從原始碼重編。CUDA toolkit 不裝整套，只從 nvidia/cuda runtime 拷需要的 6 個 .so。
# TP 首版不支援，runtime 只留單機 server 需要的依賴。
ARG CUDA_TAG=13.0.2-runtime-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG} AS cuda-libs

FROM ubuntu:24.04
LABEL org.opencontainers.image.source="https://github.com/sf-stav/veloGB10"
LABEL org.opencontainers.image.description="veloGB10 single-node inference, repacked from upstream prebuilt tarball"

# libcuda.so.1 由 NVIDIA Container Toolkit 在 runtime 掛載，不包進 image。
# 其餘 ldd 實測需要的：cudart/cublas/cublasLt/curand/nvrtc 由 cuda-libs stage 拷貝，
# ibverbs/nl/lzma/stdc++ 由 apt 提供。
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    libibverbs1 \
    liblzma5 \
    libnl-3-200 \
    libnl-route-3-200 \
    libstdc++6 \
  && rm -rf /var/lib/apt/lists/*

# cp -a 保留 symlink。用 wildcard COPY 會把 link 和本體各存一份，浪費約 900MB。
# 用 --mount 讀取 cuda stage，不留中間層。
# 注意 glob 要分開寫：libcublas.so* 配對不到 libcublasLt.so，libnvrtc.so* 配對不到 libnvrtc-builtins。
RUN --mount=from=cuda-libs,source=/usr/local/cuda/lib64,target=/tmp/cudalib \
  mkdir -p /usr/local/cuda/lib64 \
  && cp -a /tmp/cudalib/libcudart.so* /tmp/cudalib/libcublas.so.* \
    /tmp/cudalib/libcublasLt.so* /tmp/cudalib/libcurand.so* \
    /tmp/cudalib/libnvrtc.so.* /tmp/cudalib/libnvrtc-builtins.so* \
    /usr/local/cuda/lib64/ \
  && ls -la /usr/local/cuda/lib64/
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# 上游 tarball 解開後的 velogb10/ 目錄原樣拷入。
# binary 用相對路徑讀 src/ptx，WORKDIR 必須是這個目錄。
WORKDIR /opt/velogb10
COPY velogb10/ ./

EXPOSE 9000
ENTRYPOINT ["./gb10_inference"]
CMD ["--help"]
