#!/bin/bash
# shellcheck disable=SC2086

####################################################################################################
# F-test for the TF order in the validation region (VR): unblinded, background-only (r = 0).
# VR counterpart of run_ftest_bbtt.sh (SR, blinded); adapted from HH4b's run_ftest_hh4b_unblinded.sh.
#
# 1) Makes VR datacards (CreateDatacard.py --region vr) for TF orders 0..3
# 2) Per channel and order: workspace, b-only fit and saturated GoF on data (run_unblinded_bbtt.sh -wbg)
# 3) Runs fit diagnostics and saves shapes (-d|--dfit)
# 4) Generates toys from the b-only fit of order --order (-t|--goftoys)
# 5) Fits orders --order and --order + 1 to the same toys and gets their GoF statistics (-f|--ffits)
#
# The VR is signal-agnostic: one model per bmin with channels vr<ch>pass / vr<ch>fail, so there is
# one F-test per channel, all in the same card dir (outputs labelled vr<ch>). Unlike the SR there
# are no channel masks and no frozen bins: the fit sees data in all mass bins.
#
# Example (all channels, order 0 vs 1):
#   ./run_ftest_vr_bbtt.sh --cardstag <TAG>-vr --templatestag <TAG> --bmin 10 -tf --order 0
#
# Author: Raghav Kansal, Haoyang "Billy" Li
####################################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_UNBLINDED="${SCRIPT_DIR}/run_unblinded_bbtt.sh"

goftoys=0
ffits=0
dfit=0
seed=42
numtoys=100
order=0
year="all"
bmin=10
CHANNELS=("hh" "he" "hm")
prefix="vr"

options=$(getopt -o "tfdo:s:" --long "cardstag:,templatestag:,goftoys,ffits,dfit,order:,numtoys:,seed:,year:,bmin:,channel:" -- "$@") || exit 1
eval set -- "$options"

while true; do
    case "$1" in
        -t|--goftoys)
            goftoys=1
            ;;
        -f|--ffits)
            ffits=1
            ;;
        -d|--dfit)
            dfit=1
            ;;
        --cardstag)
            shift
            cards_tag=$1
            ;;
        --templatestag)
            shift
            templates_tag=$1
            ;;
        -o|--order)
            shift
            order=$1
            ;;
        -s|--seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --year)
            shift
            year=$1
            ;;
        --bmin)
            shift
            bmin=$1
            ;;
        --channel)
            shift
            case "$1" in
                hh|he|hm) CHANNELS=("$1") ;;
                *) echo "ERROR: Invalid channel '$1'. Use 'hh', 'he', or 'hm'"; exit 1 ;;
            esac
            ;;
        --)
            shift
            break;;
    esac
    shift
done

if [ -z "${cards_tag:-}" ] || [ -z "${templates_tag:-}" ]; then
    echo "ERROR: --cardstag and --templatestag are required"
    exit 1
fi

# Run from combine directory so relative paths (cards_dir) resolve correctly
cd "${SCRIPT_DIR}" || exit

echo "Arguments: cardstag=$cards_tag templatestag=$templates_tag dfit=$dfit goftoys=$goftoys \
ffits=$ffits order=$order seed=$seed numtoys=$numtoys year=$year bmin=$bmin channels=${CHANNELS[*]}"

analysis_name="bmin_${bmin}"
templates_dir="$(cd "${SCRIPT_DIR}/../postprocessing/templates" && pwd)/${templates_tag}"
cards_dir="${SCRIPT_DIR}/cards/f_tests/${cards_tag}/"
mkdir -p "${cards_dir}"
echo "Saving datacards to ${cards_dir}"
echo "Templates: ${templates_dir}"

# background-only everywhere: r fixed to 0 (as in run_unblinded_bbtt.sh)
bonly="--setParameters r=0 --freezeParameters r"

####################################################################################################
# Making cards (all channels at once, one model per order)
####################################################################################################

for ord in {0..3}
do
    model_name="nTF_${ord}"
    bin_dir="${cards_dir}${model_name}/${analysis_name}"

    if [ ! -f "${bin_dir}/${prefix}hhpass.txt" ]; then
        echo "Making VR datacards for $model_name (${analysis_name})"
        python3 -u "${SCRIPT_DIR}/../postprocessing/CreateDatacard.py" --region vr \
            --templates-dir "${templates_dir}" --model-name "${model_name}" --nTF "${ord}" \
            --cards-dir "${cards_dir}" --year "${year}" --bmin "${bmin}" || exit 1
    fi

    if [ ! -d "${bin_dir}" ]; then
        echo "ERROR: Card directory ${bin_dir} not found after CreateDatacard"
        exit 1
    fi
done

####################################################################################################
# F-test per channel
####################################################################################################

for ch in "${CHANNELS[@]}"; do
    label="${prefix}${ch}"
    wsm_snapshot=higgsCombine${label}Snapshot.MultiDimFit.mH125

    echo "========== VR F-test for ${label} (${analysis_name}) =========="

    for ord in {0..3}; do
        cd "${cards_dir}nTF_${ord}/${analysis_name}" || exit

        # workspace, b-only fit, GoF on data, unless already there
        if [ ! -f "./higgsCombine${label}Data.GoodnessOfFit.mH125.root" ] || [ ! -f "./${wsm_snapshot}.root" ]; then
            echo "Making workspace, b-only fit, GoF on data for ${label} order ${ord}"
            bash "${RUN_UNBLINDED}" -wbg --channel "${ch}" --prefix "${prefix}"
        fi

        if [ $dfit = 1 ]; then
            bash "${RUN_UNBLINDED}" -d --channel "${ch}" --prefix "${prefix}"
        fi

        cd "${SCRIPT_DIR}" || exit
    done

    toys_name="${label}${order}"
    toys_file="${cards_dir}nTF_${order}/${analysis_name}/higgsCombineToys${toys_name}.GenerateOnly.mH125.${seed}.root"

    if [ $goftoys = 1 ]; then
        cd "${cards_dir}nTF_${order}/${analysis_name}" || exit
        mkdir -p outs
        ulimit -s unlimited
        echo "Toys for ${label} order ${order}"
        combine -M GenerateOnly -m 125 -d ${wsm_snapshot}.root \
            --snapshotName MultiDimFit --bypassFrequentistFit $bonly \
            -n "Toys${toys_name}" -t "$numtoys" --saveToys -s "$seed" -v 1 2>&1 | tee "outs/${label}gentoys${order}.txt"
        cd "${SCRIPT_DIR}" || exit
    fi

    if [ $ffits = 1 ]; then
        for ord in $order $((order+1)); do
            cd "${cards_dir}nTF_${ord}/${analysis_name}" || exit
            mkdir -p outs
            ulimit -s unlimited
            echo "GoF fits on order-${order} toys for ${label} order ${ord}"
            combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 $bonly \
                -n "Toys${toys_name}" -v 1 -s "$seed" -t "$numtoys" --toysFile "${toys_file}" \
                2>&1 | tee "outs/${label}GoF_toys${order}.txt"
            cd "${SCRIPT_DIR}" || exit
        done
    fi

    echo "========== Done ${label} =========="
done
