# Eight-hard-point naming report

Implementation commit: `6608b59fb39bde13b59a3275e5a5288b17abea3f` (`feat: name wishbone inboard pivots by direction`).

The geometry API now requires `inboard_forward`, `inboard_rearward`, and `outboard` under each corner's upper and lower arms. Legacy `inboard_a`/`inboard_b` fields are rejected with a field path and a hint to choose forward or rearward from chassis x position. No automatic remapping or schema-version change was added.

For the examples, the larger-x pivot maps to `inboard_forward` and the smaller-x pivot maps to `inboard_rearward`. This also holds at the rear, where forward x is less negative. For example, RL lower pivots are `inboard_rearward = [-1.42, 0.35, 0.20]` and `inboard_forward = [-0.98, 0.35, 0.20]`. Right-side reflection changes y and preserves the names. The hinge axis remains `inboard_forward - inboard_rearward`, which is +x for both example fixtures and preserves their previous physical hinge direction. The eight exact FL/FR/RL/RR paths are documented in `examples/suspension_geometry/README.md`.

The implementation commit contains these nine files: `suspension_geometry/config.py`, `suspension_geometry/kinematics.py`, `suspension_geometry/actuation.py`, `tests/suspension_geometry/test_config.py`, `tests/suspension_geometry/test_kinematics.py`, `tests/suspension_geometry/test_actuation.py`, `configs/suspension_geometry/geometries/synthetic_direct.yaml`, `configs/suspension_geometry/geometries/synthetic_rocker.yaml`, and `examples/suspension_geometry/README.md`. The parent's modified plan document was left unstaged.

Verification:

- Test-first RED: `uv --cache-dir .superpowers/sdd/suspension-geometry/uv-cache-naming run --extra dev python -m pytest --basetemp .superpowers/sdd/suspension-geometry/naming-pytest-tmp -p no:cacheprovider tests/suspension_geometry/test_config.py::test_geometry_length_units_convert_mm_to_si -q` exited 1 at the old parser's expected `geometry.corners.FL.lower.inboard_forward: unknown field` error.
- `$env:UV_CACHE_DIR = 'C:\VD Tools\.superpowers\sdd\suspension-geometry\uv-cache-naming'; $env:PYTEST_ADDOPTS = '--basetemp=.superpowers/sdd/suspension-geometry/pytest-temp-focused-20261005 -p no:cacheprovider'; uv run --extra dev python -m pytest -q tests/suspension_geometry/test_config.py tests/suspension_geometry/test_kinematics.py tests/suspension_geometry/test_actuation.py` exited 0 with 38 passed in 1.77 s.
- Full `uv run --extra dev python -m pytest -q` exited 0 with 38 passed in 1.64 s. The successful run used a task-local uv cache and pytest temporary directory; scoped escalation was needed because the default sandbox denied NumPy's compiled DLLs.
- `git diff --check` over the nine owned files exited 0.
- A read-only jounce check at -0.04, 0, and +0.04 m found valid rigid corner states for direct and rocker fixtures. Maximum reported constraint residual was `2.22e-16 m`; left/right position and orientation reflection error was at most `6.42e-16`. Equal front heave gave ARB twist at most `1.33e-15 rad`; opposite +0.04/-0.04 m jounce gave `-0.465945 rad` twist. The lower hinge unit axis remained `[1, 0, 0]`.
- Rocker spring compression at -0.04/0/+0.04 m was `-0.025707/0/+0.039307 m`; both sides agreed. Direct spring actuation at +0.04 m was `+0.028610 m` on both sides. At direct -0.04 m, the rigid corner remained valid but the configured spring returned `stroke_limit`, so no compression value was available there.
