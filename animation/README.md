# 3D avatar asset

`avatar.glb` is the dashboard's 3D character (made with Meshy), with all of its
animations in one file. Its texture was shrunk from 4096 to 1024 px so the file
is small enough for git (23.3 MB -> 3.8 MB):

```bash
python animation/shrink_glb.py animation/avatar.glb   # keeps avatar.original.glb (gitignored)
```

Run that again after re-exporting from Meshy.

The dashboard picks clips by name (`CLIPS` in the 3D AVATAR section of
`dashboard/dashboard.html`):

| Activity | Clip | Notes |
| --- | --- | --- |
| standing | `Listening_Gesture` (9.3 s) | idle, with one arm gesture |
| walking | `Walking` (1.07 s) | one stride per loop |
| running | `Running` (0.67 s) | one stride per loop |
| biking | `Unicycle_Jump_Dismount`, 0-0.62 s only | the riding part, played back and forth (= one pedal turn); a unicycle wheel is drawn under it |
| biking on your back | `bicycle_crunch` (1.13 s) | one pedal turn per loop; exported 1.4 m below the floor, lifted in code |

If a re-export renames a clip, update `CLIPS`. If a clip loops more than one
stride or pedal turn, set its `cycles`, so the animation speed still matches
the measured cadence.

# Explainer animation

`explainer.html` is a 7-scene animated explainer (2:50), drawn in plain
JavaScript on a 1920x1080 canvas from real data in `data/story.json`:

| # | Scene | Real data shown |
| --- | --- | --- |
| 1 | The hardware | (diagram) |
| 2 | What the network reads | trial A_09's 8 input channels |
| 3 | Inside the step detector | the final model's feature maps and output on 4 s of A_09 |
| 4 | Teaching it to count | A_09's output through the warm start and the count training; the warm-start loss |
| 5 | Learning from mistakes | per-layer gradient sizes, 6 weights over training, the training loss, A_09 predictions early to final |
| 6 | From steps to distance | B_03's step features and the MLP's step lengths through training vs the tape-measured 30 m |
| 7 | Live on the board | replay of A_09's counted steps on a dashboard mock |

`data/story.json` comes from `record_training.py`, a same-seed re-run of the
real trainings with snapshots (it checks the re-run step detector equals the
committed model, and writes nothing in `models/`):

```bash
python animation/record_training.py      # rewrites animation/data/story.json
```

To watch or record it, serve the repo (a double-clicked file can't load the JSON):

```bash
python dashboard/serve.py                # then open http://localhost:8000/animation/explainer.html
```

Keys: space play/pause, left/right scene, F fullscreen, R (or the "Play all for
recording" button) restarts from scene 1 full screen with the controls and
cursor hidden; Esc leaves. `?t=95&paused` opens at 1:35, paused.
