#!/usr/bin/env bash
set -e

OUTPUT_FILE="log.log.log"
PID_FILE=".log.pid"

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -f    Follow-only: only tail live logs, no historical"
    echo "  -k    Kill running tail process"
    echo "  -h    Show this help"
    echo ""
    echo "Default: Aggregate historical logs then start live tail"
}

kill_existing_tail() {
    if [ -f "$PID_FILE" ]; then
        if kill "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Killed old process"
        else
            echo "Couldn't kill it. 🤷"
        fi
        rm -f "$PID_FILE"
    fi
}

old_logs() {
    echo "Dumping logs..."
    # Remove existing file
    rm -f "$OUTPUT_FILE"

    # Find all files in /var/log by ctime (creation/change time), oldest first
    # Use find with -printf to get ctime and filename, then sort
    find /var/log -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | while read -r _ filepath; do
        # Check file type
        filetype=$(file -b "$filepath" 2>/dev/null || echo "unknown")

        if [[ "$filepath" == *.gz ]]; then
            # Compressed file - use zcat
            printf "🪵"
            if command -v zcat >/dev/null 2>&1; then
                zcat "$filepath" 2>/dev/null >> "$OUTPUT_FILE" || true
            fi
        elif [[ "$filetype" == *"ASCII text"* ]] || [[ "$filetype" == *"Unicode text, UTF-8"* ]]; then
            # Text file - use cat
            printf "🪵"
            cat "$filepath" 2>/dev/null >> "$OUTPUT_FILE" || true
        fi
    done
    echo ""  # Add newline after dots
}

live_logs() {
    echo "Following logs..."

    # Find recent log files and show what we'll tail
    find /var/log -name '*.log' -type f -mmin -60 2>/dev/null | tail -n10 | while IFS= read -r logfile; do
        echo "Will tail: $logfile"
    done

    # Start tailing recent log files in the background (start from end with -n0)
    find /var/log -name '*.log' -type f -mmin -60 2>/dev/null | tail -n10 | xargs -r tail -n0 -F >> "$OUTPUT_FILE" 2>/dev/null &

    # Save the PID
    echo $! > "$PID_FILE"

    echo "PID: $(cat "$PID_FILE")"
    echo "To cancel: make log-stop"
}

# Parse command line options
FOLLOW_ONLY=false
KILL_MODE=false

while getopts "fkh" opt; do
    case $opt in
        f)
            FOLLOW_ONLY=true
            ;;
        k)
            KILL_MODE=true
            ;;
        h)
            show_usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_usage
            exit 1
            ;;
    esac
done

# Execute based on mode
kill_existing_tail
if [ "$KILL_MODE" = true ]; then
    exit 0
fi

if [ "$FOLLOW_ONLY" = false ]; then
    echo "Creating $OUTPUT_FILE..."
    old_logs
fi

live_logs
echo "🪵🪵🪵: $OUTPUT_FILE"

