#!/bin/bash
# Rewrite bbtautau datacards so process decay names match inference-devel HH model.
# Replaces hbbhtauau -> hbbhtt so the model parses two decays (hbb, htt) from SM_HIGG_DECAYS.
#
# Usage:
#   rewrite_datacards_for_dhi.sh [ -o OUT ] INPUT
#   rewrite_datacards_for_dhi.sh --in-place INPUT [ INPUT ... ]
#   rewrite_datacards_for_dhi.sh INPUT [ INPUT ... ]
#
# - Without -o or --in-place: writes each INPUT to <dirname>/<basename>_dhi.txt
# - With -o OUT: writes single INPUT to OUT
# - With --in-place: overwrites each INPUT (creates INPUT.bak)
set -e

OUT=""
IN_PLACE=false
FILES=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -o)
      OUT="$2"
      shift 2
      ;;
    --in-place)
      IN_PLACE=true
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      FILES+=("$1")
      shift
      ;;
  esac
done

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "Usage: $0 [ -o OUT | --in-place ] INPUT [ INPUT ... ]" >&2
  echo "  Replaces hbbhtauau -> hbbhtt in datacard(s) for inference-devel HH model." >&2
  exit 1
fi

if [[ -n "$OUT" && ${#FILES[@]} -gt 1 ]]; then
  echo "-o can only be used with a single INPUT" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "No such file: $f" >&2
    exit 1
  fi
done

for f in "${FILES[@]}"; do
  if [[ -n "$OUT" ]]; then
    sed 's/hbbhtauau/hbbhtt/g' "$f" > "$OUT"
    echo "Wrote $OUT"
  elif $IN_PLACE; then
    sed -i.bak 's/hbbhtauau/hbbhtt/g' "$f"
    echo "Updated $f (backup $f.bak)"
  else
    dir=$(dirname "$f")
    base=$(basename "$f" .txt)
    out="$dir/${base}_dhi.txt"
    sed 's/hbbhtauau/hbbhtt/g' "$f" > "$out"
    echo "Wrote $out"
  fi
done
