#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
    echo "A command is required (for example: khoroos ui)." >&2
    exit 2
fi

if [ -n "${HF_TOKEN_FILE:-}" ]; then
    if [ ! -r "$HF_TOKEN_FILE" ]; then
        echo "HF_TOKEN_FILE is not readable: $HF_TOKEN_FILE" >&2
        exit 1
    fi
    HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN_FILE is empty: $HF_TOKEN_FILE" >&2
        exit 1
    fi
    export HF_TOKEN
fi

prefetch="${KHOROOS_PREFETCH_MODELS:-auto}"
case "$prefetch" in
    auto|AUTO)
        case "${1##*/}:${2:-}" in
            khoroos:ui|khoroos:analyze) prefetch=1 ;;
            *) prefetch=0 ;;
        esac
        for argument in "$@"; do
            if [ "$argument" = "--help" ]; then
                prefetch=0
            fi
        done
        ;;
esac

case "$prefetch" in
    1|true|TRUE|yes|YES|on|ON)
        echo "Prefetching Khoroos model checkpoints..."
        khoroos models download
        ;;
    0|false|FALSE|no|NO|off|OFF)
        ;;
    *)
        echo "KHOROOS_PREFETCH_MODELS must be auto or a boolean value." >&2
        exit 2
        ;;
esac

exec "$@"
