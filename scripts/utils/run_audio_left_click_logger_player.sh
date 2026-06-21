#!/usr/bin/env bash
set -euo pipefail

UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$UTILS_DIR/../.." && pwd)"
LUA_SCRIPT="${LUA_SCRIPT:-$UTILS_DIR/lua/audio_click_logger.lua}"

DEFAULT_AUDIO_DIR="${1:-${DIR_AUDIO:-${DIR_SBS:-$REPO_ROOT/work/audio}}}"
DEFAULT_CSV_PATH="${2:-${CSV_PATH:-$REPO_ROOT/audio_click_annotations.csv}}"

read -r -p "Audio folder [${DEFAULT_AUDIO_DIR}]: " AUDIO_DIR_INPUT
AUDIO_DIR="${AUDIO_DIR_INPUT:-$DEFAULT_AUDIO_DIR}"

read -r -p "CSV output path [${DEFAULT_CSV_PATH}]: " CSV_PATH_INPUT
CSV_PATH="${CSV_PATH_INPUT:-$DEFAULT_CSV_PATH}"

if ! command -v mpv >/dev/null 2>&1; then
  echo "[ERR] mpv not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$AUDIO_DIR" ]]; then
  echo "[ERR] folder not found: $AUDIO_DIR" >&2
  exit 1
fi

if [[ ! -f "$LUA_SCRIPT" ]]; then
  echo "[ERR] Lua script not found: $LUA_SCRIPT" >&2
  exit 1
fi

AUDIO_DIR="$(cd "$AUDIO_DIR" && pwd)"

mapfile -d '' -t files < <(
  find "$AUDIO_DIR" -type f \
    \( -iname '*.wav' -o -iname '*.wave' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.ogg' -o -iname '*.opus' -o -iname '*.aiff' -o -iname '*.aif' -o -iname '*.wma' \) \
    -print0 | sort -z
)

if (( ${#files[@]} == 0 )); then
  echo "[ERR] no audio files found in: $AUDIO_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$CSV_PATH")"

echo "[INFO] Audio folder: $AUDIO_DIR"
echo "[INFO] CSV output file: $CSV_PATH"
echo "[INFO] Files in playlist: ${#files[@]}"
echo "[INFO] Left click marks a file as selected in the CSV"

mpv \
  --keep-open=always \
  --idle=yes \
  --force-window=yes \
  --audio-display=no \
  --script="$LUA_SCRIPT" \
  --script-opts="audio_click_logger-csv_path=$CSV_PATH,audio_click_logger-show_osd=yes,audio_click_logger-dedupe=yes" \
  -- "${files[@]}"
