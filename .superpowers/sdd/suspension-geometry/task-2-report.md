# Task 2 report: rigid corner kinematics and actuation

## Delivered interfaces

- `CornerState` is a frozen dataclass with `valid`, `reason`, `jounce`, `wheel_center`, `rotation`, `upper_outboard`, `lower_outboard`, `tie_outboard`, `residual`, `condition`, and `solution`. `solution[:3]` is wheel-centre translation from nominal in metres; `solution[3:]` is the incremental upright rotation vector in radians. `rotation` is the matrix applied to nominal upright point offsets about the nominal wheel centre.
- `solve_corner(corner, jounce, seed=None, max_nfev=100, tolerance=1e-9)` closes four wishbone arm lengths, the tie-link length, and the requested wheel-centre z position with a bounded six-variable least-squares solve. Residual and conditioning diagnostics are in metres and a dimensionless, length-scaled condition number.
- `solve_corner_sweep(corner, jounces, max_nfev=100, tolerance=1e-9)` starts at zero, continues independently in positive and negative jounce, and returns states in caller order.
- `attachment_position(corner, state, attachment)` transforms chassis, upright, lower-arm, upper-arm, and rocker attachments from their reference positions.
- `actuation_state(corner, state)` returns `valid`, `reason`, `spring_length`, `spring_compression`, `spring_stroke_margin`, `damper_length`, `damper_compression`, `damper_stroke_margin`, `rocker_angle`, and `rod_residual`. Compression is positive when length decreases from nominal. Each stroke margin is a two-element metre array `[remaining_compression, remaining_extension]`, computed as `[length - min_length, max_length - length]`.
- `arb_angles(geometry, states)` returns one entry per configured axle with `left_angle`, `right_angle`, `twist`, `valid`, and `reason`. The common shaft direction is from the right pivot to the left pivot. Configured lever axes must be parallel or antiparallel to it; local angles are projected onto that shaft direction and `twist = left_angle - right_angle`.

Invalid corner reasons include `nonfinite_jounce`, `jounce_out_of_range`, `invalid_seed`, `solver_error`, `no_convergence`, `singular_constraints`, and `discontinuous_branch`. Actuation reasons include `invalid_corner`, `attachment_invalid`, `rocker_closure`, `rocker_toggle`, `rocker_limit`, and `stroke_limit`. ARB geometry outside the common-shaft model returns `bar_axis_invalid`; an unreachable drop link returns `link_closure`, a toggle returns `link_toggle`, and a configured angular limit returns `angle_limit`. Invalid physical outputs are NaN while finite solver diagnostics and the best-fit solution are retained where available.

## Verification evidence

TDD RED: the first focused run failed with 12 expected missing-API assertions (exit 1). The implemented focused suite then passed 16 tests (exit 0). The full suite passed 38 tests (exit 0), including the Task 1 parser tests after the synthetic ARB fixture correction.

Commands and observed results:

```text
$uvTask2Cache = Join-Path $env:TEMP 'vdtools-uv-cache-task2'; uv --cache-dir $uvTask2Cache run --extra dev python -m pytest tests/suspension_geometry/test_kinematics.py tests/suspension_geometry/test_actuation.py -q
exit 0 — 16 passed in 2.08 s

$uvTask2Cache = Join-Path $env:TEMP 'vdtools-uv-cache-task2'; uv --cache-dir $uvTask2Cache run --extra dev python -m pytest -q
exit 0 — 38 passed in 2.45 s

uv run --extra dev python -m pytest -q
exit 0 — 38 passed in 10.17 s (independent parent run)
```

An independent link-length probe over front-corner jounces from -0.06 m to +0.06 m gave a maximum rigid-constraint residual of `2.22e-16 m`. The normalized condition number ranged from `95.87` to `105.83`. A deliberately impossible in-range request at +0.60 m returned `no_convergence` with a `0.17543 m` maximum residual and a finite condition estimate; neighboring 0 m and +0.02 m samples remained valid. Out-of-range and NaN requests returned `jounce_out_of_range` and `nonfinite_jounce` without affecting neighboring sweep states.

The synthetic rocker closes its pushrod to at most `5.55e-17 m` residual over the tested sweep. At +0.04 m bump it has `-0.29248 rad` rocker rotation and `+0.03931 m` spring compression. Its normalized rod/rocker tangent alignment was 0.70 at +0.04 m and 0.55 at +0.06 m (zero is toggle; one is full first-order distance sensitivity), confirming that the tested bump travel is away from a toggle. The rocker geometry remains actuator-valid through +0.06 m and -0.08 m; +0.08 m is rejected by its configured rocker-angle limit. Its declared jounce range therefore exceeds usable bump travel by 0.02 m.

For the corrected common-shaft ARB fixture, equal +0.035 m front heave produced `8.88e-16 rad` twist. Opposite +0.035/-0.035 m jounce produced `-0.40679 rad` twist. Reversing both lever axes while reversing their angle limits preserved canonical angles and twist. A noncoaxial lever axis returned `bar_axis_invalid`. Both synthetic geometry files now use transverse y-axis levers with tips 0.13 m forward of their pivots.

## Limits and handoff notes

The model is rigid and quasi-static. The configured rocker angle limit is reached before the synthetic geometry's +0.08 m jounce bound; downstream study axes should cap rocker bump travel at +0.06 m or the synthetic rocker limit should be intentionally revised. No claims are made for unsupplied vehicle geometry. The planned study runner, maps, plots, and CLI are outside Task 2.
