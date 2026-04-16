#!/bin/bash
# shellcheck disable=SC2086

syst="full"
inj=""
param="kl"
unblinded="False"
xsec="False"
card_base="combined_"
while getopts ":p:is:uxd:" opt; do
  case $opt in
    p)
      param=$OPTARG
      ;;
    i)
      inj="<i"
      ;;
    s)
      syst=$OPTARG
      ;;
    u)
      unblinded="True"
      ;;
    x)
      xsec="True"
      ;;
    d)
      card_base=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ "$syst" == "full" ]]; then
    frozen=""
elif [[ "$syst" == "bkgd" ]]; then
    frozen="--frozen-parameters allConstrainedNuisances"
elif [[ "$syst" == "stat" ]]; then
    frozen="--frozen-parameters allConstrainedNuisances,var{CMS_bbbb_hadronic_tf_dataResidual.*}"
else
    echo "Invalid syst argument"
    exit 1
fi

if [[ "$param" == "kl" ]]; then
    parameters="kl,-15,20,36"
    show="kt,CV,C2V"
elif [[ "$param" == "C2V" ]]; then
    parameters="C2V,0,2,21"
    show="kl,kt,CV"
else
    echo "Invalid param argument"
    exit 1
fi

xsecbr=""
if [[ "$xsec" == "True" ]]; then
   xsecbr="--xsec fb --frozen-groups signal_norm_xsbr --br bbtt"
fi

card_dir=./
datacards="${card_dir}/${card_base}.txt${inj}"
model=hh_model_run23.model_boosted_run23
campaign="62 fb$^{-1}$ (13.6 TeV)"

export DHI_CMS_POSTFIX="Supplementary"
law run PlotUpperLimits \
    --version dev  \
    --datacards "$datacards" \
    --hh-model "$model" \
    --remove-output 0,a,y \
    --show-parameters "$show" \
    --campaign "$campaign" \
    --use-snapshot False \
    --file-types pdf,png,root,c $xsecbr \
    --pois r \
    --scan-parameters "$parameters" \
    --y-log \
    --unblinded "$unblinded" \
    --save-ranges $frozen