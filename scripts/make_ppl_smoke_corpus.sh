#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/oscar_kv_ppl_smoke_corpus.txt}"
REPEATS="${REPEATS:-128}"
SMOKE_CORPUS_SOURCES="${SMOKE_CORPUS_SOURCES:-$ROOT_DIR/fixtures/ppl_smoke_seed.txt}"

IFS=',' read -ra sources <<< "$SMOKE_CORPUS_SOURCES"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

for _ in $(seq 1 "$REPEATS"); do
  for src in "${sources[@]}"; do
    if [[ "$src" != /* ]]; then
      src="$ROOT_DIR/$src"
    fi
    if [[ -f "$src" ]]; then
      {
        printf '\n\n===== %s =====\n\n' "${src#$ROOT_DIR/}"
        sed 's/[[:cntrl:]]/ /g' "$src"
      } >> "$OUT"
    else
      echo "missing smoke corpus source: $src" >&2
      exit 1
    fi
  done
done

bytes="$(wc -c < "$OUT" | tr -d ' ')"
hash="$(sha256sum "$OUT" | awk '{print $1}')"
echo "wrote $OUT"
echo "bytes=$bytes"
echo "sha256=$hash"
