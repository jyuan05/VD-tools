# Suspension Geometry Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. The user authorized implementation with GPT Luna Max subagents and the primary agent as harness/manager. That authorization supplies the execution method and permission to proceed without another planning approval.

**Goal:** Deliver a runnable Python double-wishbone suspension metric tool with config-driven geometry, finite-pose rate maps, plots, and MATLAB-readable model exports.

**Architecture:** Validated dictionaries hold SI geometry/setup/study inputs. Rigid corner kinematics and separate actuation feed elastic energies; finite chassis pose composes the corners and derivatives. A result object feeds exports and plots identically.

**Tech Stack:** Python >=3.11, NumPy, SciPy, Matplotlib, PyYAML, h5py, pytest, uv; MATLAB reader using h5read.

**Spec:** C:/VD Tools/docs/suspension-geometry-plan.md

## Global Constraints

- Python tool with MATLAB-friendly exports is the selected implementation.
- Right-handed chassis frame: x forward, y left, z up; corner order FL, FR, RL, RR.
- q = [heave_m, roll_rad, pitch_rad]; positive heave lowers chassis, positive roll raises left, positive pitch lowers nose; R = Ry(pitch) Rx(roll).
- Spring motion ratio = spring compression / wheel jounce, derived from actual geometry; no extra cosine multiplier.
- Preserve forces, preload/geometric stiffness, left/right ARB coupling, 3 x 3 body stiffness and validity masks.
- Constant reference world hub heights define the base road-support approximation; permit longitudinal/lateral migration. No animation or dynamic simulation.
- Invalid results are NaN with reason codes. Interpolation rejects unsupported major schema, invalid stencils and out-of-domain queries.
- Configuration and export schema version 1.0.0; canonical data use SI. Config unit conversions are explicit.
- Synthetic geometry is clearly labelled. No claim of validation against an actual vehicle without supplied data.
- The parent writes planning/ledger material and manages infrastructure, review and validation. All product code and tests are implemented by Luna Max subagents.
- Subagents do not spawn children. One implementer at a time; a fresh task reviewer follows each implementation. All work remains local on feat/suspension-geometry; no remote creation/push.

## Review Focus

1. Unit conversion and mirrored axial directions: parser tests must exercise mm/deg/N-per-mm and physically equivalent left/right geometry.
2. Incorrect nominal closure or impossible travel: validate link constraints, return explicit invalid states, preserve valid data at neighboring samples.
3. Loaded nonlinear leverage: finite-difference force checks must detect missing preload/geometric stiffness.
4. Consumers reading non-cubic array grids: HDF5/MATLAB round trips use unequal axis lengths and sentinel values.
5. Boundary interpolation/derivatives and absent tyres: invalid stencils never interpolate; unavailable tyre results never become valid zeros.

## Stable interfaces and input contract

Use ordinary dict configurations and NumPy arrays, with dataclasses for solver and map results where helpful. Parsers return detached resolved dictionaries carrying `schema_version`, `id`, `units` (canonical SI), and `source` provenance. Keep unknown-field validation explicit rather than silently accepting typos.

`load_geometry(path) -> dict`, `load_setup(path, geometry=None) -> dict`, `load_study(path) -> dict`, `validate_geometry(geometry) -> None`, `resolve_study(path) -> (geometry, setup, study)` live in config.py. Loading and numeric closure checks remain distinct so config.py never depends on the solver.

Geometry dictionary:

```text
schema_version, id, synthetic, units {length: m, angle: rad}, reference_origin_world [3]
corners {FL, FR, RL, RR}, optionally mirror_right: true for supplied FL/RL
each corner:
  wheel_center [3], spindle_axis [3], radius scalar
  lower {inboard_rearward [3], inboard_forward [3], outboard [3]}
  upper {inboard_rearward [3], inboard_forward [3], outboard [3]}
  tie {inboard [3], outboard [3]}
  jounce_limits [min,max]
  spring {type: direct|rocker, fixed [3], moving: Attachment, length_limits [min,max]}
  damper: same geometry as spring, optional (defaults explicitly to spring geometry)
  rocker (if used): pivot [3], axis [3], rod_point [3],
                     rod_mount Attachment, angle_limits [min,max]
Attachment = {body: chassis|upright|lower|upper|rocker, point: [3]} with all point positions at the reference pose
arbs: optional {front,rear}; each axle {left: Lever, right: Lever}
Lever = {pivot [3], axis [3], tip [3], pickup: Attachment, angle_limits [min,max]}
```

Rocker moving attachments are allowed only for spring/damper endpoints in corners with a rocker; spring.moving.point and damper.moving.point supply their own reference positions. spring.fixed/damper.fixed are chassis-fixed. There is no redundant rocker.spring_point field. Moving reference points rotate with their specified body; the upright rotates about its reference wheel centre, and wishbone points rotate about the inboard pivot axis. Axial vectors (rocker/lever axes) mirror with det(S) S; polar vectors/points mirror with S = diag(1,-1,1). Reflection of a joint's entire geometry must preserve physical actuation. Document spindle-axis convention consistently and normalize direction vectors. Explicit four-corner input remains supported.

Setup dictionary: `schema_version`, `id`, `units {length,force,angle}`, `corners {FL,...}`, `arbs {front,rear}`. Each corner has `spring {rate, preload_force}` or `spring {curve: [[compression,force],...]}`; exclusive alternatives, finite increasing abscissae, passive tangent; optional `bump_stop {engagement,rate}` or force curve; optional `tyre {rate,reference_force}` or force curve. Preload compression, if supported, is converted to reference force and conflicts are rejected. ARB law is `{rate,reference_twist}` or torque/twist curve. Omitted bars mean zero contribution; an active setup bar with no geometry is a config error. Curve domain violations are invalid states, not extrapolation.

Study dictionary: geometry/setup relative paths; `axes {heave,roll,pitch}` as explicit arrays or `{min,max,count}`; `units {length,angle}`; `reference [3]`; `metrics` list; `plots` list; `solver {max_samples, max_nfev, residual_tolerance, derivative_steps [3], jounce_step}`. Default limits: max_samples=2000, max_nfev=100, residual_tolerance=1e-9 m, derivative_steps=[1e-4,1e-4,1e-4], jounce_step=1e-4 m. Reject nonfinite axes, repeated/unsorted values, invalid sample counts and unknown metric names at the correct layer. One-element axes are allowed; reference need not be the grid midpoint.

`solve_corner(corner, jounce, seed=None, max_nfev=100, tolerance=1e-9) -> CornerState`:
state has `valid`, `reason`, `jounce`, `wheel_center [3]`, `rotation [3,3]`, `upper_outboard [3]`, `lower_outboard [3]`, `tie_outboard [3]`, `residual`, `condition`, `solution [6]`. Invalid states carry finite diagnostic values where available but invalid physical outputs are NaN. Rotation is the increment from nominal upright orientation. `solve_corner_sweep(corner,jounces,...)` starts at zero, continues separately outward in both directions and preserves caller order. `attachment_position(corner, state, attachment) -> [3]` and `actuation_state(corner,state) -> dict` expose moving positions, spring/damper compression, rocker angle and stroke margins. `arb_angles(geometry,states) -> dict` exposes signed left/right lever angles.

`component_law(spec, displacement) -> (energy, force, tangent)` includes preload consistently relative to the reference; energy differences may be negative around a loaded reference. `wheel_energy(geometry,setup,jounce_vector,...) -> dict` returns total/component energy, corner states and actuation. `wheel_response(...) -> dict` returns energy, gradient [4], stiffness [4,4], material rates [4], motion ratios [4], validity and component contributions.

`body_rotation(q) -> [3,3]`, `pose_jounces(geometry,q,...) -> dict` solve exact fixed-world-hub-height constraints using rigid corner solutions (or validated caches). `evaluate_pose(geometry,setup,q,solver=None) -> dict` returns metric dictionary plus masks/diagnostics. Compute energy derivatives via a reusable scaled finite-difference routine with stencil validity and documented one-sided boundary behavior. `tyre_effective_response(...) -> dict` solves local vertical-wheel equilibrium including tyre reference forces and condenses the full q/u Hessian; explicit unavailable/invalid state when omitted, unsupported or singular.

`run_study(geometry,setup,study) -> MapResult` with `axes {heave_m,roll_rad,pitch_rad}`, `metrics {name: ndarray}`, `validity {name: ndarray}`, `metadata dict`, `corner_maps dict`. Metric leading shape is always (Nh,Nr,Np); corner fields append (4,), body stiffness appends (3,3), wheel stiffness appends (4,4). Component decomposition uses separate named metric arrays or a declared component axis. `metric_definitions()` supplies unit, axes and meaning for every metric.

`write_map(result,path)`, `read_map(path) -> MapResult`, `write_csv(result,path)`, `interpolate_map(result,q,metric)` live in export.py. `plot_metric(result,name,output,held=None) -> list[Path]` and `plot_study(...)` consume results only. `main(argv=None) -> int` exposes validate/run/plot commands via `vd-suspension`.

### Task 1: Package, validated configuration and synthetic fixtures

**Files:** pyproject.toml, uv.lock, suspension_geometry/__init__.py, suspension_geometry/config.py, tests/suspension_geometry/test_config.py, configs/suspension_geometry/{geometries,setups,studies}/*.yaml, examples/suspension_geometry/README.md.

**Interfaces:** Produce the five parser/validator functions and canonical dictionaries above. Do not implement solver/physics. Use bundled Python for testing until uv environment is ready.

- [ ] Write parser tests first: mm input equivalent to SI, degrees convert correctly, asymmetric explicit four corners stay asymmetric, mirror axes correctly, zero-length pivot and missing toe link identify corner/field, NaN rejected, conflicting preload rejected, missing ARB geometry rejected, irregular/one-point axes allowed, 2001 samples rejected.
- [ ] Run the parser test file and record the missing-feature failure. Implement strict parsing, resolving relative paths, conversion and geometric validation. Do not import not-yet-existing solver modules.
- [ ] Supply two clearly synthetic working geometries, direct and rocker, with matching setups and studies. Use front x=1.3, rear x=-1.2, half-track around .75 m, radius .32 m, horizontal reference wishbones with distinct heights and a behind-centre tie-link point. Make all rods/lever reference lengths geometrically consistent. Provide 1D demo studies and a small unequal-axis 3D study (7 x 5 x 3).
- [ ] Run parser tests and full current suite; report commands/results. Commit only owned files. Build package metadata with console entry point referring to future cli.py; no eager imports of missing modules.

### Task 2: Rigid corner kinematics, moving attachments and actuation

**Files:** suspension_geometry/kinematics.py, suspension_geometry/actuation.py, tests/suspension_geometry/test_kinematics.py, tests/suspension_geometry/test_actuation.py.

**Interfaces:** Consume Task 1 normalized dictionaries; produce CornerState and solver/attachment/actuation/bar-angle functions above. Explicitly coordinate any fixture correction with the parent instead of silently modifying Task 1 files.

- [ ] Write independent constraint tests before implementation: verify each outboard point's distance to both inboard pivots equals its nominal length; upright pairwise distances preserved; toe-link length preserved; wheel z moves by requested jounce. Verify zero pose closes, repeated/reversed sweeps agree, +/- travel mirror equivalently, impossible jounce returns invalid and nominal neighbors remain valid.
- [ ] Implement six pose unknowns (wheel-centre translation and rotvec), four wishbone distance constraints plus tie-link length and jounce. Scale residuals in metres. Use bounded solver effort and seed continuation. Reject discontinuous assembly solutions and poorly conditioned constraints.
- [ ] Test moving attachment closure independently. Direct spring length change equals compression, inclination already included. Rocker rotates around its actual pivot axis; rod length stays constant, nearest reference branch stays continuous and configured rocker/stroke limits invalidate outputs.
- [ ] Implement left/right ARB drop-link closure and signed angles using the configured rotating levers. Verify mirror-invariant energy-relevant twist orientation and equal/opposite motion cases. A torsional bar's twist is a physically declared signed relative rotation, not an arbitrary subtraction of inconsistent mirror angles.
- [ ] Run focused tests and full current suite; record actual residuals, branches and numerical limitations. Commit owned files.

### Task 3: Elastic energies, loaded wheel stiffness and component rates

**Files:** suspension_geometry/elasticity.py, suspension_geometry/numerics.py, tests/suspension_geometry/test_elasticity.py, tests/suspension_geometry/test_numerics.py.

**Interfaces:** Consume corner/actuation APIs; produce component_law, wheel_energy, wheel_response and reusable derivative routines. Full-energy differentiation must preserve setup-dependent force and preload.

- [ ] Write analytic component tests: linear law U(c)=F0*c+0.5*k*c², force F0+k*c, tangent k; force curve integral/derivative consistency; bump-stop inactive/engaged behavior; invalid curve domain and kink derivatives; finite difference polynomial including mixed terms.
- [ ] Write loaded leverage regression using c(j)=a*j+b*j² with nonzero reference force; tangent must equal k*(a+2*b*j)²+(F0+k*c)*2*b. A version omitting the force-times-curvature term must fail.
- [ ] Implement scalar laws, a numerically controlled energy gradient/Hessian with validity-aware central or one-sided stencils and reason codes. Expose steps and report derivative quality. Avoid treating failed perturbations as valid zero derivatives.
- [ ] Compose physical spring/stop and actual ARB twist energies; expose forces and full 4 x 4 tangent matrix. Ideal symmetric heave leaves ARB energy unchanged; differential motion produces the expected coupling. Component sums equal total energy/force/stiffness.
- [ ] Verify selected actual-geometry derivative values with an independent smaller-step calculation and record convergence. Run focused tests/full current suite and commit owned files.

### Task 4: Finite chassis pose, coupled rates, tyre ride rates and sweeps

**Files:** suspension_geometry/vehicle.py, suspension_geometry/metrics.py, suspension_geometry/sweeps.py, tests/suspension_geometry/test_vehicle.py, tests/suspension_geometry/test_sweeps.py, tests/suspension_geometry/test_tyres.py.

**Interfaces:** Consume parsers, corner and energy APIs; produce body/pose/tyre functions, metric_definitions and MapResult/run_study. Export metric names and shapes in a report so Task 5 consumes exactly them.

- [ ] Write finite-pose tests for Ry Rx and signs; world wheel-centre heights fixed, x/y migration free; small-pose solution tends to h-y*phi+x*theta. Nominal energy and component gradients/stiffness agree across body/wheel composition.
- [ ] Verify body gradient/Hessian against independent perturbations and symmetric linear fixture with K_hh=sum(k_i), K_phiphi=sum(k_i*y_i²), K_thetatheta=sum(k_i*x_i²). Check off-diagonal couplings and symmetry, retaining geometric second derivatives at nonzero force.
- [ ] Implement exact pose solving and metrics: camber road/chassis, toe, caster, migration, jounce, compression, ratios, actuation gains, energy, gradients/restoring reactions, material/full wheel rates, wheel/body matrices, axle heave/roll/pitch rates and front elastic roll fraction. Clearly mark constrained rates and unavailable values.
- [ ] Implement tyre-inclusive equilibrium/condensation; test scalar k_w*k_t/(k_w+k_t), a coupled quadratic energy analytic Schur complement, absent tyres, singular internal matrix and failed equilibrium. Nonlinear force laws/reference tyre force must remain consistent with the reference operating point.
- [ ] Build bounded sweeps, resolve metadata and masks and reusable per-corner maps. Cache only geometry, invalidate by geometry identity/settings, do not bridge invalid samples. Run small unequal-axis study and measure its runtime.
- [ ] Run focused/full suite and commit owned files. Document actual limits and derivative errors rather than claim unmeasured performance.

### Task 5: Versioned export, interpolation, plotting, CLI and MATLAB integration

**Files:** suspension_geometry/export.py, suspension_geometry/plotting.py, suspension_geometry/cli.py, matlab/readSuspensionMap.m, matlab/interpolateSuspensionMap.m, tests/suspension_geometry/test_export.py, tests/suspension_geometry/test_plotting.py, tests/suspension_geometry/test_cli.py, README.md, docs/suspension-conventions.md, docs/suspension-export-schema.md.

**Interfaces:** Consume actual MapResult and metric definitions from Task 4; produce specified export/plot/CLI APIs and reader. Write HDF5 axes and named datasets with metric unit/axes/definition/validity metadata. Do not add independent physics to plots.

- [ ] Write sentinel export test first with 2 x 3 x 4 grid, four-corner and matrix fields, masks and missing values; reading reproduces exact metric values and shapes. Test major-version rejection, interpolated linear sentinel agreement, bounds rejection, one-point axes and invalid-cell rejection, including boundary nodes next to invalid cells.
- [ ] Implement HDF5 plus JSON companion and optional long CSV. Persist resolved config/provenance, source/config hashes, actual run cost, masks/reasons and corner maps. Python interpolation returns declared metric shape and rejects incomplete validity stencils.
- [ ] Implement metric curves, selected 2D heatmaps and setup overlays at compatible slices; PNG/SVG/PDF as requested. Plot titles carry held pose and SI/display units. Invalid outputs stay gaps. Sanitize metric-derived filenames.
- [ ] Implement validate/run/plot CLI with useful error messages and nonzero exit on invalid input. `run` supports `--output` and writes map.h5, metadata.json, optional data.csv, summary.json and configured plots. Document exact commands using uv and clearly synthetic examples.
- [ ] Implement MATLAB reader plus interpolation with explicit Python-to-MATLAB dimension permutation, schema/units checks, strict bounds/masks, including scalars, singleton grid axes, corner and matrix tails. Supply a MATLAB sentinel check script or generated fixture instructions. Parent will run actual MATLAB batch round-trip; agent may run it if safe and available.
- [ ] Run full pytest and demo CLI end-to-end; record produced artifacts and runtime. Commit owned files.

## Harness validation and completion

Each task gets a fresh Luna Max reviewer with spec, task brief, report and a base-to-head diff. The parent settles interface ambiguities and sends bounded fix rounds to the original implementer. Local feature commits are permitted; no push or merge.

At the end run the full Python suite, direct and rocker demo studies, a nonzero heave/roll/pitch unequal-grid map, no-tyre and invalid-domain checks, and actual MATLAB HDF5/interpolation round trip. Inspect representative PNG outputs. Record runtime, validity counts, output size, branch, revision and dirty state. A broad final Luna Max review must identify correctness/spec gaps; resolve material findings before reporting completion. No tests or simulations in the unrelated C:/VD checkout.


## Hard-point naming clarification (2026-10-05)

The user names the chassis-side wishbone points: front upper forward, front upper rearward, front lower forward, front lower rearward, rear upper forward, rear upper rearward, rear lower forward, rear lower rearward. Each label applies to the corresponding left and right corner. Within the existing corner/upper-or-lower hierarchy, canonical keys are inboard_forward and inboard_rearward. The outboard ball joint is unchanged. In the synthetic fixtures, the forward pivot is the larger chassis-x coordinate: old inboard_b becomes inboard_forward; old inboard_a becomes inboard_rearward. Coordinates, units and physical constraints remain unchanged. This naming update applies to all remaining task interfaces and exports; it does not expand the physics scope.
