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
