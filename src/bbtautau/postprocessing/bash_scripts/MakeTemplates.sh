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
# --use-part: Use ParT tagger instead of BDT (maps to --use_ParT in python)
# --do-vbf: Include VBF signal regions
# --bmin: Minimum background yield value(s) - supports multiple values (e.g., --bmin 1 5 10)
# --no-sensitivity-dir: Disable the --sensitivity-dir argument (default: enabled)
# --test-mode: Run in test mode (reduced data size)
# --tt-pres: Apply tt preselection
####################################################################################################

years=("2022" "2022EE" "2023" "2023BPix")
channels=("hh" "hm" "he")
bmin_values=(5 10 12)  # Can be overridden with --bmin

MAIN_DIR="/home/users/lumori/bbtautau"
SCRIPT_DIR="${MAIN_DIR}/src/bbtautau/postprocessing"
DATA_DIR="/ceph/cms/store/user/lumori/bbtautau/skimmer/25Sep23AddVars_v12_private_signal"
SENSITIVITY_DIR="${MAIN_DIR}/plots/SensitivityStudy/2025-12-27/" #"${MAIN_DIR}/plots/SensitivityStudy/2025-12-12/" is bmin = 10, "${MAIN_DIR}/plots/SensitivityStudy/2025-12-15/" is bmin = 5
TAG=""
USE_PART=0
DO_VBF=0
USE_SENSITIVITY_DIR=1  # Flag to control --sensitivity-dir argument (default: on)
TEST_MODE=0
TT_PRES=0
GGF_MODEL="19oct25_ak4away_ggfbbtt"
VBF_MODEL="19oct25_ak4away_vbfbbtt"

# Function to display help
show_help() {
    echo "Usage: $0 --tag TAG [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --tag TAG              Tag for the templates and plots"
    echo ""
    echo "Optional arguments:"
    echo "  --year YEAR            Year to run on (default: all years)"
    echo "  --channel CHANNEL      Channel to run on (default: all channels)"
    echo "  --use-part             Use ParT tagger instead of BDT"
    echo "  --do-vbf               Include VBF signal regions"
    echo "  --no-sensitivity-dir   Disable the --sensitivity-dir argument (default: enabled)"
    echo "  --test-mode            Run in test mode (reduced data size)"
    echo "  --tt-pres              Apply tt preselection"
    echo "  --ggf-model MODEL      GGF model name (default: 19oct25_ak4away_ggfbbtt)"
    echo "  --vbf-model MODEL      VBF model name (default: 19oct25_ak4away_vbfbbtt)"
    echo "  --bmin VALUES          Space-separated list of minimum background yield values"
    echo "                         Examples: --bmin 1"
    echo "                                  --bmin 1 5 10"
    echo "                                  --bmin 1 2 5 8 10 15 20"
    echo ""
    echo "Examples:"
    echo "  $0 --tag my_analysis --bmin 1 5 10"
    echo "  $0 --tag my_analysis --year 2022 --channel hh --use-part --bmin 1 5 8"
    echo "  $0 --tag my_analysis --do-vbf --bmin 10"
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
        --use-part)
            USE_PART=1
            shift
            ;;
        --do-vbf)
            DO_VBF=1
            shift
            ;;
        --no-sensitivity-dir)
            USE_SENSITIVITY_DIR=0
            shift
            ;;
        --test-mode)
            TEST_MODE=1
            shift
            ;;
        --tt-pres)
            TT_PRES=1
            shift
            ;;
        --ggf-model)
            shift
            GGF_MODEL=$1
            shift
            ;;
        --vbf-model)
            shift
            VBF_MODEL=$1
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
echo "USE_PART: $USE_PART"
echo "DO_VBF: $DO_VBF"
echo "USE_SENSITIVITY_DIR: $USE_SENSITIVITY_DIR"
echo "TEST_MODE: $TEST_MODE"
echo "TT_PRES: $TT_PRES"
echo "GGF_MODEL: $GGF_MODEL"
echo "VBF_MODEL: $VBF_MODEL"

for year in "${years[@]}"
do
    echo "Data dir: $DATA_DIR"

    echo "Templates for $year"
    for channel in "${channels[@]}"
    do
        echo "    Templates for $channel with bmin values: ${bmin_values[*]}"

        # Build command as array to handle spaces and special characters properly
        cmd=(
            python -u "${SCRIPT_DIR}/postprocessing.py"
            --year "$year"
            --channel "$channel"
            --data-dir "$DATA_DIR"
            --plot-dir "${MAIN_DIR}/plots/Templates/$TAG"
            --template-dir "${MAIN_DIR}/src/bbtautau/postprocessing/templates/$TAG"
            --templates
            --ggf-modelname "$GGF_MODEL"
            --vbf-modelname "$VBF_MODEL"
        )

        # Add --use_ParT if enabled (use ParT instead of BDT)
        if [[ $USE_PART -eq 1 ]]; then
            cmd+=(--use_ParT)
        fi

        # Add --do-vbf if enabled
        if [[ $DO_VBF -eq 1 ]]; then
            cmd+=(--do-vbf)
        fi

        # Add --sensitivity-dir if enabled
        if [[ $USE_SENSITIVITY_DIR -eq 1 ]]; then
            cmd+=(--sensitivity-dir "$SENSITIVITY_DIR")
        fi

        # Add --test-mode if enabled
        if [[ $TEST_MODE -eq 1 ]]; then
            cmd+=(--test-mode)
        fi

        # Add --tt-pres if enabled
        if [[ $TT_PRES -eq 1 ]]; then
            cmd+=(--tt-pres)
        fi

        # Add bmin values (passed as multiple arguments)
        cmd+=(--bmin "${bmin_values[@]}")

        # Print command for debugging
        echo "    Running: ${cmd[*]}"

        # Execute the command
        "${cmd[@]}"

    done
done
