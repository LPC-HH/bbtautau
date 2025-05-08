#!/bin/bash
# shellcheck disable=SC2086,SC2043,SC2206

####################################################################################################
# Script for making templates
# Author: Raghav Kansal
####################################################################################################


####################################################################################################
# Options
# --tag: Tag for the templates and plots
# --year: Year to run on - by default runs on all years
####################################################################################################

years=("2022" "2022EE" "2023" "2023BPix")
channels=("hh" "he" "hm")

MAIN_DIR="../../.."
data_dir_2022="/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal"
data_dir_otheryears="/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"
TAG=""

options=$(getopt -o "" --long "year:,tag:,channel:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        --year)
            shift
            years=($1)
            ;;
        --tag)
            shift
            TAG=$1
            ;;
        --channel)
            shift
            channels=($1)
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

if [[ -z $TAG ]]; then
  echo "Tag required using the --tag option. Exiting"
  exit 1
fi

echo "TAG: $TAG"

for year in "${years[@]}"
do
    # this needs a more permanent solution
    if [[ $year == "2022" ]]; then
        data_dir=$data_dir_2022
        # continue
    else
        data_dir=$data_dir_otheryears
    fi

    echo $data_dir

    echo "Templates for $year"
    for channel in "${channels[@]}"
    do
        echo "    Templates for $channel"
        python -u postprocessing.py --year $year --channel $channel --data-dir $data_dir --plot-dir "${MAIN_DIR}/plots/Templates/$TAG" --template-dir "templates/$TAG" --templates
    done
done
