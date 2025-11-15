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
# --use_bdt: Flag to enable use of BDT in template creation
# --bmin: Minimum background yield value(s) - supports multiple values (e.g., --bmin 1 5 10)
# --no-sensitivity-dir: Disable the --sensitivity-dir argument (default: enabled)
####################################################################################################

years=("2022" "2022EE" "2023" "2023BPix")
channels=("hh" "he" "hm")
bmin_values=(1)  # Default to single value, can be overridden with --bmin

MAIN_DIR="/home/users/lumori/bbtautau"
SCRIPT_DIR="${MAIN_DIR}/src/bbtautau/postprocessing"
data_dir_2022="/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal"
data_dir_otheryears="/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr24Fix_v12_private_signal"
SENSITIVITY_DIR="${MAIN_DIR}/plots/SensitivityStudy/2025-07-31/"
TAG=""
USE_BDT=0
USE_SENSITIVITY_DIR=1  # Flag to control --sensitivity-dir argument (default: on)

# Function to display help
show_help() {
    echo "Usage: $0 --tag TAG [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --tag TAG          Tag for the templates and plots"
    echo ""
    echo "Optional arguments:"
    echo "  --year YEAR        Year to run on (default: all years)"
    echo "  --channel CHANNEL  Channel to run on (default: all channels)"
    echo "  --use_bdt          Enable BDT usage in template creation"
    echo "  --no-sensitivity-dir  Disable the --sensitivity-dir argument (default: enabled)"
    echo "  --bmin VALUES      Space-separated list of minimum background yield values"
    echo "                     Examples: --bmin 1"
    echo "                              --bmin 1 5 10"
    echo "                              --bmin 1 2 5 8 10 15 20"
    echo ""
    echo "Examples:"
    echo "  $0 --tag my_analysis --bmin 1 5 10"
    echo "  $0 --tag my_analysis --year 2022 --channel hh --use_bdt --bmin 1 5 8"
}

# Parse arguments manually to handle multiple bmin values
while [[ $# -gt 0 ]]; do
    case "$1" in
        --year)
            shift
            years=($1)
            shift
            ;;
        --tag)
            shift
            TAG=$1
            shift
            ;;
        --bmin)
            shift
            # Parse multiple bmin values separated by spaces
            bmin_values=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                bmin_values+=($1)
                shift
            done
            ;;
        --channel)
            shift
            channels=($1)
            shift
            ;;
        --use_bdt)
            USE_BDT=1
            shift
            ;;
        --no-sensitivity-dir)
            USE_SENSITIVITY_DIR=0
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [[ -z $TAG ]]; then
  echo "Tag required using the --tag option. Exiting"
  exit 1
fi

# Validate that bmin_values is not empty
if [[ ${#bmin_values[@]} -eq 0 ]]; then
  echo "No bmin values provided. Using default value of 1"
  bmin_values=(1)
fi

echo "TAG: $TAG"
echo "BMIN VALUES: ${bmin_values[*]}"
echo "YEARS: ${years[*]}"
echo "CHANNELS: ${channels[*]}"
echo "USE_BDT: $USE_BDT"
echo "USE_SENSITIVITY_DIR: $USE_SENSITIVITY_DIR"

for year in "${years[@]}"
do
    # this needs a more permanent solution
    if [[ $year == "2022" ]]; then
        data_dir=$data_dir_2022
    else
        data_dir=$data_dir_otheryears
    fi

    echo $data_dir

    echo "Templates for $year"
    for channel in "${channels[@]}"
    do
        echo "    Templates for $channel with bmin values: ${bmin_values[*]}"
        # Build base command
        base_cmd="python -u ${SCRIPT_DIR}/postprocessing.py --year $year --channel $channel --data-dir $data_dir --plot-dir \"${MAIN_DIR}/plots/Templates/$TAG\" --template-dir \"${MAIN_DIR}/src/bbtautau/postprocessing/templates/$TAG\" --templates"

        # Add --use_bdt if enabled
        if [[ $USE_BDT -eq 1 ]]; then
            base_cmd="$base_cmd --use_bdt --model 29July25_loweta_lowreg"
        fi

        # Add --sensitivity-dir if enabled
        if [[ $USE_SENSITIVITY_DIR -eq 1 ]]; then
            base_cmd="$base_cmd --sensitivity-dir \"$SENSITIVITY_DIR\""
        fi

        # Add bmin values

        #TODO: This block  needs to be tested, could not be correct

        base_cmd=("$base_cmd" --bmin "${bmin_values[@]}")
        # Execute the command
        # eval $base_cmd
        "${base_cmd[@]}"

    done
done
