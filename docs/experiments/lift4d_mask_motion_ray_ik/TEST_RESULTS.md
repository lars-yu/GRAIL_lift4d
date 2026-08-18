Remote Grail environment:

- `py_compile` passed for the optimizer, formal runner, comparison renderer,
  and top-view renderer.
- Focused formal runner/static-priority tests after the continuity change:
  `16 passed`.
- Full unittest discovery after the continuity and renderer fixes:
  `58 passed`.

Real retry6 acceptance metrics:

| Check | Result |
| --- | ---: |
| Lift4D supervised frames | 121 / 121 |
| Motion onset | frame 105 |
| Contact hand | right, raw 2D mask distance |
| Contact-frame distance | 0.0262805 m |
| Moving frames under 5 cm | 100% |
| Maximum adjacent distance change | 0.00221541 m |
| Static optimized Z std | 2.38419e-7 m |
| Body RMSE increase | 2.30837 px |
| Hand RMSE increase | -0.545679 px |
| Human-mask IoU delta | +0.000540361 |
| Object contact gradient | 0 |
| Four-frame periodic jumps | 0 |

Stage-loss restoration was also verified from `optimization_metrics.json`:

- Stage A: `0.000244268857 -> 0.000244268857`
- Stage B: `33.8629265 -> 19.4475689`
- Stage C: `16.3003254 -> 14.3555822`

All three final losses are no greater than their corresponding initial losses.

All formal videos were opened with OpenCV and verified as nonblank `121`-frame
MP4s at 30 fps. The three-column comparison is `3840x720`; the rigid and top
views are `1280x720`.

All ten acceptance gates in `optimization_metrics.json` are `true`.
