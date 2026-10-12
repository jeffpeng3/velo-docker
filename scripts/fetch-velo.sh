#!/usr/bin/env bash
# 下載上游 release tarball，驗 SHA，解出 velogb10/ 目錄給 Docker build context 用。
# 用法：scripts/fetch-velo.sh <version> [dest-dir]
# 範例：scripts/fetch-velo.sh v0.7.4 ./.fetch
set -euo pipefail

VERSION="${1:?usage: fetch-velo.sh <version> [dest-dir], e.g. v0.7.4}"
DEST="${2:-./.fetch}"
UPSTREAM_REPO="${UPSTREAM_REPO:-sf-stav/veloGB10}"
BASE_URL="https://github.com/${UPSTREAM_REPO}/releases/download/${VERSION}"

mkdir -p "${DEST}"
cd "${DEST}"

echo "== download ${VERSION} from ${UPSTREAM_REPO}"
curl -fsSL -o SHA256SUMS.txt "${BASE_URL}/SHA256SUMS.txt"
curl -fsSL -o "velogb10-${VERSION}-gb10-sm121.tar.gz" "${BASE_URL}/velogb10-${VERSION}-gb10-sm121.tar.gz"
TARBALL="$(ls velogb10-*.tar.gz 2>/dev/null | head -1)"

echo "== verify sha256"
python3 - "${TARBALL}" SHA256SUMS.txt <<'EOF'
import hashlib, sys
tarball, sums = sys.argv[1], sys.argv[2]
digest = hashlib.sha256(open(tarball, 'rb').read()).hexdigest()
expected = open(sums).read()
if digest not in expected:
    print(f"MISMATCH: {tarball} sha256={digest}")
    print(expected)
    sys.exit(1)
print(f"SHA_OK {tarball} {digest}")
EOF

echo "== extract"
rm -rf velogb10
tar xzf "${TARBALL}"
test -x velogb10/gb10_inference
PTX_COUNT="$(ls velogb10/src/ptx/*.ptx | wc -l)"
echo "ptx_count=${PTX_COUNT}"
test "${PTX_COUNT}" -ge 19
file velogb10/gb10_inference | grep -q aarch64
echo "OK dest=${DEST}/velogb10 version=${VERSION}"
