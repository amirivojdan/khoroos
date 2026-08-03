#!/bin/sh
set -eu

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

case "${KHOROOS_PREFETCH_MODELS:-1}" in
    1|true|TRUE|yes|YES|on|ON)
        echo "Prefetching Khoroos model checkpoints..."
        khoroos models download
        ;;
    0|false|FALSE|no|NO|off|OFF)
        ;;
    *)
        echo "KHOROOS_PREFETCH_MODELS must be a boolean value." >&2
        exit 2
        ;;
esac

exec "$@"
