#!/bin/bash
# shellcheck disable=SC2086

####################################################################################################
# 1) Makes datacards and workspaces for different orders of polynomials
# 2) Runs background-only fit (Higgs mass window blinded) for lowest order polynomial and GoF test (saturated model) on data
# 3) Runs fit diagnostics and saves shapes (-d|--dfit)
# 4) Generates toys and gets test statistics for each (-t|--goftoys)
# 5) Fits +1 order models to all 100 toys and gets test statistics (-f|--ffits)
#
# Uses --sig-region ggf|vbf|all (consistent with run_blinded_bbtt.sh)
# Use --all-sr-ch to run f-test for each (signal-region x channel) combination
#   --all-sr-ch --sig-region all  -> 6 runs (ggf/vbf x hh/he/hm)
#   --all-sr-ch --sig-region ggf  -> 3 runs (ggf x hh/he/hm)
# Use --channel CH to run for a single channel only
#
# Example (all 6 sr x channel):
#   ./run_ftest_bbtt.sh --cardstag 0215_sm+bsm_ggf+vbf --templatestag 0215_sm+bsm_ggf+vbf -tf --all-sr-ch --sig-region all
#
# Author: Raghav Kansal, Haoyang "Billy" Li
####################################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BLINDED="${SCRIPT_DIR}/run_blinded_bbtt.sh"

goftoys=0
ffits=0
dfit=0
limits=0
seed=42
numtoys=100
order=0
year="2022EE"
sig_region="ggf"
bmin=10
channel=""
all_sr_ch=0

options=$(getopt -o "tfdlo:s:y" --long "cardstag:,templatestag:,goftoys,ffits,dfit,limits,order:,numtoys:,seed:,year:,sig-region:,bmin:,channel:,all-sr-ch" -- "$@")
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
        -l|--limits)
            limits=1
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
        --seed)
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
        --sig-region)
            shift
            sig_region=$1
            ;;
        --bmin)
            shift
            bmin=$1
            ;;
        --channel)
            shift
            channel=$1
            ;;
        --all-sr-ch)
            all_sr_ch=1
            ;;
        --)
            shift
            break;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
    shift
done

# Validate sig-region and set SIG_REGIONS + siglabel (must match run_blinded_bbtt.sh)
case "$sig_region" in
    ggf)
        SIG_REGIONS=("ggfbbtt")
        siglabel="ggf"
        ;;
    vbf)
        SIG_REGIONS=("vbfbbtt")
        siglabel="vbf"
        ;;
    all)
        SIG_REGIONS=("ggfbbtt" "vbfbbtt")
        siglabel="allsigs"
        ;;
    *)
        echo "ERROR: Invalid sig-region '$sig_region'. Use 'ggf', 'vbf', or 'all'"
        exit 1
        ;;
esac

CHANNELS=("hh" "he" "hm")
CMS_PARAMS_LABEL="CMS_bbtautau_boosted"
analysis_name="bmin_${bmin}"

# --all-sr-ch runs per (sig-region, channel). Use --sig-region ggf|vbf to restrict.
if [ $all_sr_ch = 1 ]; then
    case "$sig_region" in
        ggf) SR_CH_LIST="ggf:hh ggf:he ggf:hm" ;;
        vbf) SR_CH_LIST="vbf:hh vbf:he vbf:hm" ;;
        all) SR_CH_LIST="ggf:hh ggf:he ggf:hm vbf:hh vbf:he vbf:hm" ;;
        *) echo "ERROR: --all-sr-ch requires --sig-region ggf, vbf, or all"; exit 1 ;;
    esac
elif [ -n "$channel" ]; then
    # Single channel mode - validate channel
    case "$channel" in
        hh|he|hm) ;;
        *)
            echo "ERROR: Invalid channel '$channel'. Use 'hh', 'he', or 'hm'"
            exit 1
            ;;
    esac
    SR_CH_LIST="${siglabel}:${channel}"
else
    SR_CH_LIST=""
fi

# Require cards_tag and templates_tag
if [ -z "${cards_tag:-}" ] || [ -z "${templates_tag:-}" ]; then
    echo "ERROR: --cardstag and --templatestag are required"
    exit 1
fi

# Run from combine directory so relative paths (cards_dir, CreateDatacard) resolve correctly
cd "${SCRIPT_DIR}" || exit

echo "Arguments: cardstag=$cards_tag templatestag=$templates_tag dfit=$dfit \
goftoys=$goftoys ffits=$ffits order=$order seed=$seed numtoys=$numtoys year=$year \
sig-region=$sig_region siglabel=$siglabel bmin=$bmin channel=$channel all-sr-ch=$all_sr_ch"


####################################################################################################
# Set up paths and fit args
####################################################################################################

templates_dir="$(cd "${SCRIPT_DIR}/../postprocessing/templates" && pwd)/${templates_tag}"
cards_dir="cards/f_tests/${cards_tag}/"
mkdir -p "${cards_dir}"
echo "Saving datacards to ${cards_dir}"
echo "Templates: ${templates_dir}"

####################################################################################################
# Making cards (always in main bmin_10 dir; per-channel uses subdirs with symlinks)
####################################################################################################

for ord in {0..3}
do
    model_name="nTF_${ord}"
    bin_dir="${cards_dir}${model_name}/${analysis_name}"

    # create datacards if they don't already exist (check first sr for this sig-region)
    first_sr="${SIG_REGIONS[0]}"
    card_check="${bin_dir}/${first_sr}hhpass.txt"
    if [ ! -f "${card_check}" ]; then
        echo "Making Datacard for $model_name (${analysis_name})"
        python3 -u "${SCRIPT_DIR}/../postprocessing/CreateDatacard.py" --templates-dir "${templates_dir}" \
            --model-name "${model_name}" --nTF "${ord}" --cards-dir "${cards_dir}" --year "${year}" \
            --bmin "${bmin}" -y --regions pass
    fi

    if [ ! -d "${bin_dir}" ]; then
        echo "ERROR: Card directory ${bin_dir} not found after CreateDatacard"
        exit 1
    fi
done

####################################################################################################
# Run f-test: either per (siglabel, channel) or combined
####################################################################################################

run_ftest_for_combo() {
    local sl="$1"
    local ch="$2"
    local sr_flag="$3"

    local combo_label="${sl}_${ch}"
    local wsm_snap=higgsCombine${sl}${ch}Snapshot.MultiDimFit.mH125

    local sr_name=""
    case "$sl" in
        ggf) sr_name="ggfbbtt" ;;
        vbf) sr_name="vbfbbtt" ;;
        *) echo "ERROR: bad siglabel $sl"; exit 1 ;;
    esac

    local maskargs="mask_${sr_name}${ch}fail=1,mask_${sr_name}${ch}failBlinded=0,mask_${sr_name}${ch}pass=1,mask_${sr_name}${ch}passBlinded=0"
    local setparams=""
    local freeparams=""
    for bin in {5..8}; do
        setparams+="${CMS_PARAMS_LABEL}_tf_dataResidual_${sr_name}${ch}Bin${bin}=0,"
        freeparams+="${CMS_PARAMS_LABEL}_tf_dataResidual_${sr_name}${ch}Bin${bin},"
    done
    setparams=${setparams%,}
    freeparams=${freeparams%,}

    echo "========== F-test for ${combo_label} (sig-region=${sr_flag}, channel=${ch}) =========="

    for ord in {0..3}; do
        model_name="nTF_${ord}"
        base_bin="${cards_dir}${model_name}/${analysis_name}"
        combo_bin="${base_bin}_${sl}_${ch}"

        mkdir -p "${combo_bin}"
        cd "${combo_bin}" || exit

        for f in fail failMCBlinded pass passMCBlinded; do
            ln -sf "../${analysis_name}/${sr_name}${ch}${f}.txt" "${sr_name}${ch}${f}.txt" 2>/dev/null || true
        done
        # Symlink ROOT shape file (datacards reference HHModel_*.root)
        ln -sf "../${analysis_name}/HHModel_${analysis_name}_${sr_name}.root" "HHModel_${analysis_name}_${sr_name}.root" 2>/dev/null || true

        if [ ! -f "${sr_name}${ch}pass.txt" ]; then
            echo "ERROR: Card ${sr_name}${ch}pass.txt not found in ${base_bin}"
            exit 1
        fi
        if [ ! -f "HHModel_${analysis_name}_${sr_name}.root" ]; then
            echo "ERROR: Shape file HHModel_${analysis_name}_${sr_name}.root not found in ${base_bin}"
            exit 1
        fi

        outsdir="./outs"
        mkdir -p "${outsdir}"

        # Run -wbg if GoF file missing OR snapshot corrupted (e.g. only has 'toys', missing workspace 'w')
        snapshot_ok=0
        if [ -f "./higgsCombineData.GoodnessOfFit.mH125.root" ] && [ -f "./${wsm_snap}.root" ]; then
            if python3 -c "
import sys
try:
    import uproot
    f = uproot.open('${wsm_snap}.root')
    keys = [k.split(';')[0] for k in f.keys()]
    # Snapshot file must have workspace 'w', not just 'toys'
    if 'w' in keys:
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
" 2>/dev/null; then
                snapshot_ok=1
            fi
        fi
        if [ $snapshot_ok = 0 ]; then
            echo "Making workspace, b-only fit, GoF on data for ${combo_label} order ${ord}"
            bash "${RUN_BLINDED}" -wbg --sig-region "${sr_flag}" --channel "${ch}"
        fi

        if [ $dfit = 1 ]; then
            bash "${RUN_BLINDED}" -d --sig-region "${sr_flag}" --channel "${ch}"
        fi

        if [ $limits = 1 ]; then
            bash "${RUN_BLINDED}" -l --sig-region "${sr_flag}" --channel "${ch}"
        fi

        cd "${SCRIPT_DIR}" || exit
    done

    model_name="nTF_${order}"
    base_bin="${cards_dir}${model_name}/${analysis_name}"
    combo_bin="${base_bin}_${sl}_${ch}"
    toys_name=$order

    cd "${combo_bin}" || exit
    toys_file="$(pwd)/higgsCombineToys${toys_name}.GenerateOnly.mH125.$seed.root"
    outsdir="./outs"
    cd "${SCRIPT_DIR}" || exit

    if [ $goftoys = 1 ]; then
        cd "${combo_bin}" || exit
        ulimit -s unlimited
        echo "Toys for ${combo_label} order ${order}"
        combine -M GenerateOnly -m 125 -d ${wsm_snap}.root \
            --snapshotName MultiDimFit --bypassFrequentistFit \
            --setParameters "${maskargs},${setparams},r=0" \
            --freezeParameters "${freeparams},r" \
            -n "Toys${toys_name}" -t "$numtoys" --saveToys -s "$seed" -v 9 2>&1 | tee "$outsdir/gentoys.txt"
        cd "${SCRIPT_DIR}" || exit
    fi

    if [ $ffits = 1 ]; then
        for ord in $order $((order+1)); do
            model_name="nTF_${ord}"
            combo_bin="${cards_dir}${model_name}/${analysis_name}_${sl}_${ch}"
            cd "${combo_bin}" || exit
            outsdir="./outs"
            ulimit -s unlimited
            echo "GoF fits for ${combo_label} order ${ord}"
            combine -M GoodnessOfFit -d ${wsm_snap}.root --algo saturated -m 125 \
                --setParameters "${maskargs},${setparams},r=0" \
                --freezeParameters "${freeparams},r" \
                -n "Toys${toys_name}" -v 9 -s "$seed" -t "$numtoys" --toysFile "${toys_file}" 2>&1 | tee "$outsdir/GoF_toys${toys_name}.txt"
            cd "${SCRIPT_DIR}" || exit
        done
    fi

    echo "========== Done ${combo_label} =========="
}

run_ftest_combined() {
    wsm_snapshot=higgsCombine${siglabel}Snapshot.MultiDimFit.mH125

    maskunblindedargs=""
    for sr in "${SIG_REGIONS[@]}"; do
        for ch in "${CHANNELS[@]}"; do
            maskunblindedargs+="mask_${sr}${ch}fail=1,mask_${sr}${ch}failBlinded=0,mask_${sr}${ch}pass=1,mask_${sr}${ch}passBlinded=0,"
        done
    done
    maskunblindedargs=${maskunblindedargs%,}

    setparamsblinded=""
    freezeparamsblinded=""
    for sr in "${SIG_REGIONS[@]}"; do
        for ch in "${CHANNELS[@]}"; do
            for bin in {5..8}; do
                setparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_${sr}${ch}Bin${bin}=0,"
                freezeparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_${sr}${ch}Bin${bin},"
            done
        done
    done
    setparamsblinded=${setparamsblinded%,}
    freezeparamsblinded=${freezeparamsblinded%,}

    for ord in {0..3}; do
        model_name="nTF_${ord}"
        bin_dir="${cards_dir}${model_name}/${analysis_name}"
        cd "${bin_dir}" || exit
        echo "Working in ${bin_dir}"
        outsdir="./outs"
        mkdir -p "${outsdir}"

        if [ ! -f "./higgsCombineData.GoodnessOfFit.mH125.root" ]; then
            echo "Making workspace, doing b-only fit and gof on data"
            bash "${RUN_BLINDED}" -wbg --sig-region "${sig_region}"
        fi

        if [ $dfit = 1 ]; then
            bash "${RUN_BLINDED}" -d --sig-region "${sig_region}"
        fi

        if [ $limits = 1 ]; then
            bash "${RUN_BLINDED}" -l --sig-region "${sig_region}"
        fi

        cd "${SCRIPT_DIR}" || exit
    done

    model_name="nTF_${order}"
    bin_dir="${cards_dir}${model_name}/${analysis_name}"
    toys_name=$order
    cd "${bin_dir}" || exit
    toys_file="$(pwd)/higgsCombineToys${toys_name}.GenerateOnly.mH125.$seed.root"
    outsdir="./outs"
    cd "${SCRIPT_DIR}" || exit

    if [ $goftoys = 1 ]; then
        cd "${bin_dir}" || exit
        ulimit -s unlimited
        echo "Toys for $order order fit (siglabel=${siglabel})"
        combine -M GenerateOnly -m 125 -d ${wsm_snapshot}.root \
            --snapshotName MultiDimFit --bypassFrequentistFit \
            --setParameters "${maskunblindedargs},${setparamsblinded},r=0" \
            --freezeParameters "${freezeparamsblinded},r" \
            -n "Toys${toys_name}" -t "$numtoys" --saveToys -s "$seed" -v 9 2>&1 | tee "$outsdir/gentoys.txt"
        cd "${SCRIPT_DIR}" || exit
    fi

    if [ $ffits = 1 ]; then
        for ord in $order $((order+1)); do
            model_name="nTF_${ord}"
            bin_dir="${cards_dir}${model_name}/${analysis_name}"
            cd "${bin_dir}" || exit
            outsdir="./outs"
            ulimit -s unlimited
            echo "Fits for $model_name"
            combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
                --setParameters "${maskunblindedargs},${setparamsblinded},r=0" \
                --freezeParameters "${freezeparamsblinded},r" \
                -n "Toys${toys_name}" -v 9 -s "$seed" -t "$numtoys" --toysFile "${toys_file}" 2>&1 | tee "$outsdir/GoF_toys${toys_name}.txt"
            cd "${SCRIPT_DIR}" || exit
        done
    fi
}

# Main dispatch
if [ -n "$SR_CH_LIST" ]; then
    for combo in $SR_CH_LIST; do
        sl="${combo%%:*}"
        ch="${combo##*:}"
        sr_flag="$sl"
        run_ftest_for_combo "$sl" "$ch" "$sr_flag"
    done
else
    run_ftest_combined
fi
