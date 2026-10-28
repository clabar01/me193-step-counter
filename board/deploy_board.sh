#!/usr/bin/env bash
# Deploy the board app (board/) to the Uno Q and restart it.
#
#   board/deploy_board.sh             # Python side + models (after retraining)
#   board/deploy_board.sh --sketch    # also copies the sketch
#   BOARD=arduino@10.247.137.47 board/deploy_board.sh   # if AirFour.local doesn't resolve
#
# Either way 'app start' recompiles and reflashes the MCU, so power-cycle the
# board afterwards (see CLAUDE.md).
#
# Training scripts copy their exported models into board/python/models/
# (training/shared.py copy_to_app); this checks each copy matches models/.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  "" | --sketch) ;;
  -h | --help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown option: $1 (use --sketch or --help)" >&2; exit 2 ;;
esac

BOARD="${BOARD:-arduino@AirFour.local}"
APP="~/ArduinoApps/me193_steps"

for app_copy in board/python/models/*; do
  f="models/$(basename "$app_copy")"
  [[ -f "$f" ]] || continue
  if ! cmp -s "$f" "$app_copy"; then
    echo "$app_copy differs from $f: rerun the training script that exports it" >&2
    exit 1
  fi
done

files=(board/app.yaml board/python)
if [[ "${1:-}" == "--sketch" ]]; then
  files+=(board/sketch)
fi

echo "copying ${files[*]} to $BOARD:ArduinoApps/me193_steps/"
scp -q -r "${files[@]}" "$BOARD:ArduinoApps/me193_steps/"
echo "restarting me193_steps (stop + start; 'app restart' can fail with 'app is running')"
ssh "$BOARD" "timeout -k 5 120 arduino-app-cli app stop $APP; timeout -k 5 420 arduino-app-cli app start $APP"
# 'app start' reflashes the sketch even when it hasn't changed (seen 2026-10-26)
echo
echo ">>> The MCU was reflashed: POWER-CYCLE THE BOARD before trusting it (unplug ~5 s, plug back in)."
