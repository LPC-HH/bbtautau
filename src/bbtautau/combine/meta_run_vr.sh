#!/bin/bash
# shellcheck disable=SC2086

# Run the unblinded background-only VR fits (run_unblinded_bbtt.sh) over all bmin_* dirs of a
# VR card tag (CreateDatacard.py --region vr, cards vr<ch>pass.txt / vr<ch>fail.txt).
# The SR keeps using meta_run_blinded.sh.
#
# Usage: meta_run_vr.sh [individual|combined|both] --cards-dir DIR [--cmds "..."] [--only bmin_10] [-y]
#   individual: one fit per channel found (labels vrhh, vrhm, vrhe)
#   combined:   all channels found in one fit (label vrall)
#   both (default): individual + combined

SCRIPT_PATH="/home/users/lumori/bbtautau/src/bbtautau/combine/run_unblinded_bbtt.sh"
PREFIX="vr"
DEFAULT_CMDS="--bfit --dfit --gofdata"
cmds="$DEFAULT_CMDS"
CARDS_BASE_DIR=""
only_dirs=""
skip_confirm=0
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cards-dir)
            CARDS_BASE_DIR="${2%/}"
            shift 2
            ;;
        --only)
            only_dirs="$2"
            shift 2
            ;;
        --cmds)
            cmds="$2"
            shift 2
            ;;
        -y|--yes)
            skip_confirm=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [individual|combined|both] --cards-dir DIR [--cmds CMDS] [--only bmin_10,bmin_12] [-y]"
            echo "  --cmds: passed to run_unblinded_bbtt.sh (default: '$DEFAULT_CMDS'; add --goftoys for toys)"
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option $1"
            exit 1
            ;;
        *)
            if [ -z "$MODE" ]; then
                MODE="$1"
            else
                echo "ERROR: Unexpected argument $1"
                exit 1
            fi
            shift
            ;;
    esac
done
MODE="${MODE:-both}"

if [ -z "$CARDS_BASE_DIR" ] || [ ! -d "$CARDS_BASE_DIR" ]; then
    echo "ERROR: give an existing --cards-dir (VR card tag with bmin_* subdirs)"
    exit 1
fi
case "$MODE" in
    individual|combined|both) ;;
    *) echo "ERROR: Invalid mode '$MODE'"; exit 1 ;;
esac

card_dirs=()
for dir in "$CARDS_BASE_DIR"/*/; do
    if compgen -G "${dir}${PREFIX}*pass.txt" > /dev/null; then
        base=$(basename "$dir")
        if [ -n "$only_dirs" ] && [[ ",$only_dirs," != *",$base,"* ]]; then
            continue
        fi
        card_dirs+=("$dir")
    fi
done
if [ ${#card_dirs[@]} -eq 0 ]; then
    echo "No VR card directories (${PREFIX}*pass.txt) found in $CARDS_BASE_DIR"
    exit 1
fi

echo "VR fits: mode=$MODE cmds='$cmds'"
for dir in "${card_dirs[@]}"; do echo "  - $(basename "$dir")"; done

if [ "$skip_confirm" -eq 0 ]; then
    read -p "Do you want to proceed? (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

nfail=0
run_one() {
    local card_dir="$1"
    shift
    echo "=================================================="
    echo "$card_dir: run_unblinded_bbtt.sh $*"
    echo "=================================================="
    if (cd "$card_dir" && bash "$SCRIPT_PATH" --prefix "$PREFIX" --workspace "$@" $cmds); then
        echo "OK: $card_dir $*"
    else
        echo "FAILED: $card_dir $*"
        nfail=$((nfail + 1))
    fi
}

for card_dir in "${card_dirs[@]}"; do
    if [ "$MODE" != "combined" ]; then
        for pcard in "$card_dir"/"${PREFIX}"*pass.txt; do
            ch=$(basename "$pcard" .txt)
            ch=${ch#"$PREFIX"}
            ch=${ch%pass}
            run_one "$card_dir" --channel "$ch"
        done
    fi
    if [ "$MODE" != "individual" ]; then
        run_one "$card_dir"
    fi
done

echo "All VR fits done ($nfail failed)"
[ "$nfail" -eq 0 ]
