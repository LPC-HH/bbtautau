#!/bin/bash
# shellcheck disable=SC2086,SC2206

####################################################################################################
# Unblinded background-only fits, for the validation region (VR) cards of
# CreateDatacard.py --region vr (channels vr<ch>pass + vr<ch>fail, no MCBlinded copies).
# Adapted from HH4b's run_unblinded_hh4b.sh; the SR stays on run_blinded_bbtt.sh.
#
# No channel masks, no frozen window bins; the signal process is in the cards but r is fixed
# to 0 everywhere (background-only). No limits / significance.
#
# 1) Combine cards and make a workspace (--workspace / -w)
# 2) Background-only fit, saves a snapshot (--bfit / -b)
# 3) Fit diagnostics (b-only, --skipSBFit) + nuisance pulls + pre/post-fit shapes (--dfit / -d)
# 4) GoF (saturated) on data (--gofdata / -g)
# 5) GoF on toys, generated from the b-only snapshot (--goftoys / -t), off unless asked
#
# Run in a card dir (bmin_<b>/). --channel <ch> fits one channel (label vr<ch>), otherwise all
# channels whose vr<ch>pass.txt exists are combined (label vrall).
#
# Usage: run_unblinded_bbtt.sh [-wbdgt] [--channel hh] [--prefix vr] [--numtoys 100] [--seed 42]
####################################################################################################

workspace=0
bfit=0
dfit=0
gofdata=0
goftoys=0
seed=42
numtoys=100
mintol=0.1  # --cminDefaultMinimizerTolerance
verbose=1
prefix="vr"
CHANNELS=("hh" "hm" "he")
channellabel=""

options=$(getopt -o "wbdgt" --long "channel:,prefix:,workspace,bfit,dfit,gofdata,goftoys,seed:,numtoys:,mintol:,verbose:" -- "$@") || exit 1
eval set -- "$options"

while true; do
    case "$1" in
        --channel)
            shift
            CHANNELS=($1)
            channellabel=$1
            ;;
        --prefix)
            shift
            prefix=$1
            ;;
        -w|--workspace)
            workspace=1
            ;;
        -b|--bfit)
            bfit=1
            ;;
        -d|--dfit)
            dfit=1
            ;;
        -g|--gofdata)
            gofdata=1
            ;;
        -t|--goftoys)
            goftoys=1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --mintol)
            shift
            mintol=$1
            ;;
        --verbose)
            shift
            verbose=$1
            ;;
        --)
            shift
            break;;
    esac
    shift
done

dataset=data_obs
CMS_PARAMS_LABEL="CMS_bbtautau_boosted"
cards_dir="."
label="${prefix}${channellabel:-all}"
ws=${cards_dir}/combined_${label}
wsm_snapshot=higgsCombine${label}Snapshot.MultiDimFit.mH125
outsdir=${cards_dir}/outs
mkdir -p $outsdir

echo "Arguments: workspace=$workspace bfit=$bfit dfit=$dfit gofdata=$gofdata goftoys=$goftoys \
seed=$seed numtoys=$numtoys mintol=$mintol verbose=$verbose"
echo "Model: $(pwd)"
echo "Label: $label"

ccargs=""
found=()
for channel in "${CHANNELS[@]}"; do
    pcard="${prefix}${channel}pass.txt"
    fcard="${prefix}${channel}fail.txt"
    if [ -f "$pcard" ] && [ -f "$fcard" ]; then
        ccargs+="${prefix}${channel}pass=${cards_dir}/${pcard} ${prefix}${channel}fail=${cards_dir}/${fcard} "
        found+=("$channel")
    elif [ -n "$channellabel" ]; then
        echo "ERROR: $pcard / $fcard not found in $(pwd)"
        exit 1
    fi
done
if [ ${#found[@]} -eq 0 ]; then
    echo "ERROR: no ${prefix}<ch>pass.txt cards in $(pwd)"
    exit 1
fi
echo "Channels: ${found[*]}"

# background-only everywhere: r fixed to 0
bonly="--setParameters r=0 --freezeParameters r"

# need to run this for large # of nuisances
# https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735
ulimit -s unlimited

if [ $workspace = 1 ]; then
    echo "Combining cards $ccargs"
    combineCards.py $ccargs > $ws.txt

    echo "Running text2workspace"
    text2workspace.py $ws.txt -o $ws.root 2>&1 | tee $outsdir/${label}text2workspace.txt
elif [ ! -f "$ws.root" ]; then
    echo "Workspace doesn't exist! Use the -w|--workspace option to make workspace first"
    exit 1
fi


if [ $bfit = 1 ]; then
    echo "Unblinded background-only fit (r = 0)"
    combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${ws}.root -v $verbose \
    --cminDefaultMinimizerStrategy 1 --cminDefaultMinimizerTolerance "$mintol" --X-rtd MINIMIZER_MaxCalls=400000 \
    $bonly -n ${label}Snapshot 2>&1 | tee $outsdir/${label}MultiDimFit.txt
fi


if [ $dfit = 1 ]; then
    echo "Fit Diagnostics (unblinded, background-only)"
    combine -M FitDiagnostics -m 125 -d ${ws}.root $bonly --skipSBFit \
    --cminDefaultMinimizerStrategy 1 --cminDefaultMinimizerTolerance "$mintol" --X-rtd MINIMIZER_MaxCalls=400000 \
    -n ${label} --ignoreCovWarning -v $verbose 2>&1 | tee $outsdir/${label}FitDiagnostics.txt

    # diffNuisances needs a floating --poi (r is frozen here): use the first channel's TF par0
    # (only used to sort the printed table by correlation)
    pullpoi=$(grep -ho "${CMS_PARAMS_LABEL}_tf_dataResidual_${prefix}${found[0]}pass_[A-Za-z0-9]*_par0" \
        "${prefix}${found[0]}pass.txt" | head -1)
    python3 $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py fitDiagnostics${label}.root \
    --skipFitS --poi "$pullpoi" -g ${label}_nuisance_pulls.root --all --regex='^(?!.*mcstat)' \
    --vtol=0.3 --stol=0.1 --vtol2=2.0 --stol2=0.5 2>&1 | tee $outsdir/${label}nuisance_pulls.txt

    echo "Fit Shapes (prefit + postfit)"
    PostFitShapesFromWorkspace --dataset "$dataset" -w ${ws}.root --output ${label}FitShapes.root \
    -m 125 -f fitDiagnostics${label}.root:fit_b --postfit --print 2>&1 | tee $outsdir/${label}FitShapes.txt
fi


if [ $gofdata = 1 ]; then
    if [ ! -f "${wsm_snapshot}.root" ]; then
        echo "Background-only fit snapshot doesn't exist! Use -b|--bfit first."
        exit 1
    fi
    echo "GoF on data"
    combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
    --snapshotName MultiDimFit $bonly \
    -n ${label}Data -v $verbose 2>&1 | tee $outsdir/${label}GoF_data.txt
fi


if [ $goftoys = 1 ]; then
    if [ ! -f "${wsm_snapshot}.root" ]; then
        echo "Background-only fit snapshot doesn't exist! Use -b|--bfit first."
        exit 1
    fi
    echo "GoF on toys (generated from the b-only post-fit snapshot)"
    combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
    --snapshotName MultiDimFit --bypassFrequentistFit --toysFrequentist --saveToys $bonly \
    -n ${label}Toys -v $verbose -s "$seed" -t "$numtoys" 2>&1 | tee "$outsdir/${label}GoF_toys${seed}.txt"
fi
