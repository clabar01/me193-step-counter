"""Copy finished trial recordings off the board into data/raw/.

A trial is finished when both <name>.csv and <name>.json exist (the recorder
writes the .json, then renames <name>.csv.part to <name>.csv), so a trial that
is still recording, or stopped and waiting for its count, is never copied.
Files already in data/raw/ are never overwritten; a same-named file with a
different size is reported instead.

    python board/pull_recordings.py
    BOARD=arduino@192.168.1.173 python board/pull_recordings.py      # if AirFour.local doesn't resolve
    python board/pull_recordings.py --dry-run
    python board/pull_recordings.py --src data/sim --dest /tmp/pulled  # local folder instead of the board (testing)

Uses ssh/scp (no password on the board, see the README).
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REMOTE_DIR = "ArduinoApps/me193_steps/recordings"   # relative to the board user's home


def remote_listing(board, remote_dir):
    """{name: size} of the files in the board's recordings folder."""
    cmd = f"cd {shlex.quote(remote_dir)} 2>/dev/null && find . -maxdepth 1 -type f -printf '%s %f\\n'"
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=10", board, cmd],
                         capture_output=True, text=True)
    if out.returncode not in (0, 1):
        sys.exit(f"ssh {board} failed: {out.stderr.strip() or out.returncode}")
    files = {}
    for line in out.stdout.splitlines():
        size, _, name = line.partition(" ")
        if name:
            files[name] = int(size)
    return files


def local_listing(src):
    return {p.name: p.stat().st_size for p in Path(src).iterdir() if p.is_file()}


def main():
    ap = argparse.ArgumentParser(description="Copy finished trials from the board into data/raw/.")
    ap.add_argument("--board", default=os.environ.get("BOARD", "arduino@AirFour.local"))
    ap.add_argument("--remote-dir", default=REMOTE_DIR, help=f"folder on the board (default ~/{REMOTE_DIR})")
    ap.add_argument("--src", type=Path, help="copy from this local folder instead of the board")
    ap.add_argument("--dest", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be copied")
    args = ap.parse_args()

    files = local_listing(args.src) if args.src else remote_listing(args.board, args.remote_dir)
    where = args.src if args.src else f"{args.board}:~/{args.remote_dir}"
    trials = sorted(n[:-5] for n in files if n.endswith(".json") and f"{n[:-5]}.csv" in files)
    unfinished = sorted(n for n in files if n.endswith(".csv.part"))

    args.dest.mkdir(parents=True, exist_ok=True)
    wanted, conflicts, have = [], [], 0
    for name in trials:
        for fname in (f"{name}.csv", f"{name}.json"):
            local = args.dest / fname
            if not local.exists():
                wanted.append(fname)
            elif local.stat().st_size != files[fname]:
                conflicts.append(fname)
            else:
                have += 1

    print(f"{where}: {len(trials)} finished trials, {len(unfinished)} unfinished")
    for name in unfinished:
        print(f"  not copied (still recording or waiting for its count): {name}")
    for fname in conflicts:
        print(f"  CONFLICT, not copied: {fname} is already in {args.dest} with a different size")
    if not wanted:
        print(f"nothing new ({have // 2} trials already in {args.dest})")
        return
    print(f"copying {len(wanted)} files to {args.dest}{' (dry run)' if args.dry_run else ''}:")
    for fname in wanted:
        print(f"  {fname}")
    if args.dry_run:
        return
    if args.src:
        for fname in wanted:
            shutil.copy2(args.src / fname, args.dest / fname)
    else:
        remote = [f"{args.board}:{args.remote_dir}/{fname}" for fname in wanted]
        if subprocess.run(["scp", "-q", "-p", *remote, str(args.dest)]).returncode != 0:
            sys.exit("scp failed")
    print("done")


if __name__ == "__main__":
    main()
