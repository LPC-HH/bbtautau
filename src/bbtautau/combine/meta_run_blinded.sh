#!/bin/bash
# shellcheck disable=SC2086

# Script to run blinded analysis across multiple card directories
# Usage: ./meta_run_blinded.sh [combined|individual|both] [--cmds <commands>]

# Configuration
CARDS_BASE_DIR="/home/users/lumori/bbtautau/src/bbtautau/cards/26Jan6-vbf"
SCRIPT_PATH="/home/users/lumori/bbtautau/src/bbtautau/combine/run_blinded_bbtt.sh"
CHANNELS=("hh" "hm" "he")

# Default values
DEFAULT_CMDS="--bfit --limits --dfit"
cmds="$DEFAULT_CMDS"
do_vbf=0

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cmds)
            case "$2" in
                "dfit")
                    cmds="--dfit"
                    ;;
                *)
                    cmds="$2"
                    ;;
            esac
            shift 2
            ;;
        --do-vbf)
            do_vbf=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [MODE] [--cmds COMMANDS] [--do-vbf]"
            echo "  MODE: combined (default), individual, or both"
            echo "  --cmds: commands to pass to analysis script (default: '$DEFAULT_CMDS')"
            echo "          Use 'dfit' as shorthand for '--dfit'"
            echo "  --do-vbf: include VBF regions"
            echo ""
            echo "Signal region behavior:"
            echo "  Without --do-vbf:"
            echo "    - combined: 1 fit (ggf only)"
            echo "    - individual: 3 fits (hh, hm, he for ggf)"
            echo "    - both: 1 combined + 3 individual = 4 fits"
            echo "  With --do-vbf:"
            echo "    - combined: 1 fit (allsigs = ggf+vbf, 6 regions)"
            echo "    - individual: 6 fits (3 channels × 2 sig regions: ggf, vbf)"
            echo "    - both: 1 combined + 6 individual = 7 fits"
            echo ""
            echo "Output file naming: {siglabel}{channellabel}AsymptoticLimits.txt"
            echo "  e.g., ggfAsymptoticLimits.txt, vbfhhAsymptoticLimits.txt, allsigsAsymptoticLimits.txt"
            echo ""
            echo "Examples:"
            echo "  $0                           # combined ggf (1 fit)"
            echo "  $0 individual                # 3 individual ggf fits"
            echo "  $0 --do-vbf                  # combined allsigs (1 fit)"
            echo "  $0 individual --do-vbf       # 6 individual fits (ggf+vbf)"
            echo "  $0 both --do-vbf             # 1 combined + 6 individual = 7 fits"
            echo "  $0 --cmds dfit               # combined ggf with --dfit"
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            # This is the positional MODE argument
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

# Set default mode if not provided
MODE="${MODE:-combined}"

# Function to run analysis for a given directory and mode
run_analysis() {
    local card_dir="$1"
    local run_mode="$2"
    local channel="$3"
    local run_cmds="$4"
    local sig_region="$5"  # ggf, vbf, or ggfvbf

    echo "=================================================="
    echo "Running analysis in: $card_dir"
    echo "Mode: $run_mode"
    if [ -n "$channel" ]; then
        echo "Channel: $channel"
    fi
    echo "Signal region: $sig_region"
    echo "=================================================="

    cd "$card_dir" || {
        echo "ERROR: Cannot enter directory $card_dir"
        return 1
    }

    if [ "$run_mode" = "combined" ]; then
        echo "Running combined channel analysis with sig-region=$sig_region..."
        $SCRIPT_PATH --workspace --sig-region "$sig_region" $run_cmds

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Combined analysis ($sig_region) completed successfully"
        else
            echo "✗ Combined analysis ($sig_region) failed"
        fi

    elif [ "$run_mode" = "individual" ] && [ -n "$channel" ]; then
        echo "Running individual analysis for channel=$channel, sig-region=$sig_region..."
        $SCRIPT_PATH --channel "$channel" --workspace --sig-region "$sig_region" $run_cmds

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Individual analysis ($channel, $sig_region) completed successfully"
        else
            echo "✗ Individual analysis ($channel, $sig_region) failed"
        fi
    fi

    echo ""
}

# Function to check if directory contains required datacard files
check_directory() {
    local dir="$1"

    # Check for at least some expected datacard files
    if [ -f "$dir/ggfbbtthhpass.txt" ] || [ -f "$dir/ggfbbtthepass.txt" ] || [ -f "$dir/ggfbbtthmpass.txt" ]; then
        return 0
    else
        return 1
    fi
}

# Main execution
echo "Card Analysis Runner"
echo "==================="
echo "Base directory: $CARDS_BASE_DIR"
echo "Script path: $SCRIPT_PATH"
echo "Mode: $MODE"
if [ "$do_vbf" = "1" ]; then
    echo "Signal regions: ggf, vbf"
else
    echo "Signal regions: ggf"
fi
echo ""

# Check if base directory exists
if [ ! -d "$CARDS_BASE_DIR" ]; then
    echo "ERROR: Cards base directory does not exist: $CARDS_BASE_DIR"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Analysis script does not exist: $SCRIPT_PATH"
    exit 1
fi

# Find all card directories
card_dirs=()
for dir in "$CARDS_BASE_DIR"/*/; do
    if [ -d "$dir" ] && check_directory "$dir"; then
        card_dirs+=("$dir")
    fi
done

if [ ${#card_dirs[@]} -eq 0 ]; then
    echo "No valid card directories found in $CARDS_BASE_DIR"
    exit 1
fi

echo "Found ${#card_dirs[@]} card directories:"
for dir in "${card_dirs[@]}"; do
    echo "  - $(basename "$dir")"
done
echo ""

# Ask for confirmation
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run analysis based on mode
case "$MODE" in
    "combined")
        echo "Running COMBINED analysis only..."
        for card_dir in "${card_dirs[@]}"; do
            if [ "$do_vbf" = "1" ]; then
                run_analysis "$card_dir" "combined" "" "$cmds" "all"
            else
                run_analysis "$card_dir" "combined" "" "$cmds" "ggf"
            fi
        done
        ;;

    "individual")
        echo "Running INDIVIDUAL channel analyses only..."
        if [ "$do_vbf" = "1" ]; then
            SIG_REGIONS_TO_RUN=("ggf" "vbf")
        else
            SIG_REGIONS_TO_RUN=("ggf")
        fi
        for card_dir in "${card_dirs[@]}"; do
            for sig_region in "${SIG_REGIONS_TO_RUN[@]}"; do
                for channel in "${CHANNELS[@]}"; do
                    run_analysis "$card_dir" "individual" "$channel" "$cmds" "$sig_region"
                done
            done
        done
        ;;

    "both")
        echo "Running BOTH combined and individual analyses..."
        for card_dir in "${card_dirs[@]}"; do
            # Combined: use "all" (allsigs) if --do-vbf, otherwise ggf
            if [ "$do_vbf" = "1" ]; then
                run_analysis "$card_dir" "combined" "" "$cmds" "all"
            else
                run_analysis "$card_dir" "combined" "" "$cmds" "ggf"
            fi

            # Individual: loop over ggf+vbf if --do-vbf, otherwise just ggf
            if [ "$do_vbf" = "1" ]; then
                for sig_region in "ggf" "vbf"; do
                    for channel in "${CHANNELS[@]}"; do
                        run_analysis "$card_dir" "individual" "$channel" "$cmds" "$sig_region"
                    done
                done
            else
                for channel in "${CHANNELS[@]}"; do
                    run_analysis "$card_dir" "individual" "$channel" "$cmds" "ggf"
                done
            fi
        done
        ;;

    *)
        echo "ERROR: Invalid mode '$MODE'. Use 'combined', 'individual', or 'both'"
        echo "Usage: $0 [combined|individual|both]"
        exit 1
        ;;
esac

echo "=================================================="
echo "All analyses completed!"
echo "=================================================="
