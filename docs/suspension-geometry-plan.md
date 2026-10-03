# VD Tools: repository and suspension geometry plan

Date: 2026-10-03  
Status: proposed design; no implementation or repository initialization performed.

## Purpose and first release

Create a standalone VD Tools repository whose first tool is `suspension_geometry/`. It will solve rigid front and rear double-wishbone geometry from configuration files and plot engineering metrics against wheel travel and vehicle heave, roll, and pitch. Suspension animation and drawings are outside the requested scope.

Success means that a geometry configuration produces repeatable, validated metric plots and a versioned data export usable by a future vehicle model. The plotting layer must consume the same data as the export; it must not contain separate physics calculations.

This is a new-tool design, not an audit of the existing C:\VD suspension implementation. No current vehicle hard points have been supplied or verified. Example geometries must be explicitly synthetic until actual configurations are available.

## Implementation approach

Selected approach: Python numerical core, command-line entry point, and metric plotting, with HDF5 and CSV exports and a small MATLAB reader. The user selected this direction during planning. This keeps the tool independently runnable while supporting the existing MATLAB vehicle-model workflow.

Alternatives:

| Approach | Benefit | Cost |
| --- | --- | --- |
| Python core + MATLAB reader | Portable tool, clean testing and batch operation, reusable export contract | Two languages at the consumer boundary |
| MATLAB core + the same exports | Familiar execution and integration with the existing VD workflow | MATLAB needed to run the generator |
| Separate Python and MATLAB solvers | Native implementation in either environment | Duplicated physics and validation; avoid for the first release |

For the Python direction, use NumPy/SciPy for numerical work, Matplotlib for metric plots, h5py for HDF5, a validated YAML loader, and pytest for focused numerical tests. Resolve and lock dependency versions during implementation. A desktop GUI is a later decision; the first release needs a reproducible config-to-plots/export workflow.

## Proposed repository layout

```text
VD Tools/
  README.md
  pyproject.toml
  uv.lock
  docs/
    suspension-geometry-plan.md
    suspension-conventions.md
    suspension-export-schema.md
  suspension_geometry/
    __init__.py
    config.py          # parsing, units, topology and geometry validation
    kinematics.py      # corner rigid-link constraints and branch tracking
    actuation.py       # spring/damper/rocker/ARB displacement geometry
    elasticity.py      # component forces, energies and tangent stiffness
    vehicle.py         # body pose, road support and corner composition
    sweeps.py          # bounded sweeps, caching and validity handling
    metrics.py         # named, documented engineering outputs
    plotting.py        # curves and metric heatmaps from result data
    export.py          # stable data contract and provenance
    cli.py
  configs/
    suspension_geometry/
      geometries/      # named vehicle geometry configurations
      setups/          # springs, preloads, bars, tyres and stop settings
      studies/         # axes, held variables, selected metrics and outputs
  examples/
    suspension_geometry/
      synthetic_direct/
      synthetic_rocker/
  matlab/
    readSuspensionMap.m
  tests/
    suspension_geometry/
  outputs/             # ignored generated results
```

Keep one repository and initially one installable package. Add future tools as separate tool folders when they are requested. Introduce shared utilities only when a second tool actually needs them.

## Configuration contract

Separate geometry, elastic setup, and study configuration. A spring-rate change should not require copying hard points or repeating a geometry sweep.

Every configuration has a schema version and stable ID. The loader validates finite coordinates, units, complete corner definitions, referenced components, reference pose closure, and non-degenerate joint axes. Errors identify the exact configuration field and corner.

Geometry inputs:

- Chassis reference frame and reference pose; front/rear locations and wheel centre locations.
- Each wishbone's two chassis pivots and upright ball joint; upright geometry and wheel spindle orientation.
- Steering/toe-link chassis and upright points. The rear also needs a toe constraint; it cannot be left as an unconstrained upright rotation.
- Spring/damper mounts, with each point explicitly attached to chassis, upright, a wishbone, or rocker. Moving mounts use local rigid-body coordinates.
- Direct actuation or a push/pull rod and rocker: rocker pivot axis, rocker local attachment points, rod attachment and reference length, spring/damper attachment and usable stroke. Include both actuation types in the first release.
- Optional front/rear ARB pickup geometry, lever axes, drop links and torsional force law. Omitting a bar explicitly means zero bar contribution.
- Wheel nominal rolling radius and geometric/joint travel limits.
- Explicit four-corner geometry, or an explicitly enabled left-to-right mirror operation. Mirroring transforms directions and local frames as well as point coordinates.

Setup inputs:

- Linear spring rate or a force-versus-compression curve; reference preload or reference force. Reject conflicting preload specifications.
- Spring/damper reference length, bump and droop limits, and optional bump-stop force curves with engagement positions.
- ARB torsional stiffness or torque-versus-twist curve, and reference twist.
- Optional tyre vertical stiffness/force curve for tyre-inclusive ride rates.

Study inputs:

- Geometry and setup references; heave, roll and pitch ranges and sample counts.
- A required reference operating point; which coordinates vary and which are held fixed.
- Selected metrics, plots, export destination and numerical settings.
- Resource cap on samples, solver attempts and derivative evaluations.

Configuration quantities may use explicitly declared mm, degrees or N/mm, but calculations and canonical exports use SI. No unit is inferred from a number's magnitude. Version 1 holds the front rack at its configured reference position and the rear toe link fixed; steering sweeps are a later extension.

## Coordinates and boundary conditions

Use a right-handed chassis frame: x forward, y left, z up. The origin and its nominal world position are part of the configuration. Corner order is always FL, FR, RL, RR.

Let q = [h, phi, theta]. Positive h lowers the chassis origin (heave compression); positive phi follows the right-hand rule around x and raises the left side; positive theta follows the right-hand rule around y and lowers the nose. Define the finite rotation as R = Ry(theta) Rx(phi), with yaw fixed. Store these definitions and rotation order in every export.

Positive corner jounce j is upward wheel-centre movement relative to the chassis along the chassis z direction, measured from the configured reference. Positive component compression c shortens its installed length.

For reference and verification only, small-angle body motion gives approximately j_i = h - y_i phi + x_i theta. The full body-pose solver uses the specified finite rotation and rigid-link geometry.

Use a documented constant nominal hub-height road support model for the base pose maps: each wheel centre stays at its reference world height on a flat road, while longitudinal and lateral migration remain free. This is a rigid-support approximation and does not include camber-dependent tyre contact-radius changes. Record this boundary condition in the export rather than presenting it as a complete tyre contact model.

Tyre-inclusive rates are a separate local calculation that permits vertical wheel displacement and tyre compression about an equilibrium. Configured reference geometry is an operating point, not proof of static equilibrium. Prescribed-pose maps export suspension reactions without automatically solving vehicle weight, aerodynamic loads or wheel lift. Report these limitations with each run.

## Kinematic solver and numerical behavior

Model wishbones and upright as rigid bodies with revolute chassis pivots and spherical outboard joints. Solve the upright pose at prescribed jounce using wishbone and toe-link constraints. Solve rocker/rod geometry on the same physical assembly branch.

Start at the reference pose, verify constraint closure, and use continuation through adjacent samples. Do not accept an optimizer exit flag alone: check scaled link/joint residuals, assembly branch, component stroke, limits and conditioning. Retry within fixed limits; never silently jump to another assembly branch.

Cache corner kinematics versus jounce, then evaluate the requested full-body pose states and refine where needed. Keep geometry caches independent of spring settings. Validate any interpolation against direct solves, especially derivative metrics. A cache is an optimization, not permission to bridge invalid geometry.

Calculate local derivatives using constraint differentiation where practical; validate against independently stepped finite differences. Near a valid boundary use documented one-sided derivatives. At singularities or component-law kinks mark the affected derivative invalid or provide explicitly labeled left/right tangents. Track validity separately for geometry, forces and individual derivatives.

## Metrics and physical definitions

### Per-corner geometry and actuation

Plot and export jounce; camber relative to chassis and road; toe; caster; wheel-centre longitudinal/lateral migration; spring/damper compression; rocker angle; and stroke margin.

Define spring motion ratio unambiguously as r_s = dc_s/dj (spring compression divided by wheel jounce). Store a separate damper motion ratio when the spring and damper are not co-located. Compute these from actual moving attachment geometry, which already accounts for installed inclination. Do not apply an additional cosine correction.

For each heave/roll/pitch sweep, plot all four corner ratios evaluated at that pose. Also export dc/dh, dc/dphi and dc/dtheta: these are actuation gains with units m/m or m/rad, not interchangeable dimensionless motion ratios. A ratio such as dc/dj along a pose sweep is undefined when that corner has zero jounce derivative; flag it rather than divide by zero.

### Wheel and axle rates

Let F_s(c) be spring force and k_s(c) = dF_s/dc. For an isolated one-coordinate spring contribution:

```text
wheel-force contribution = F_s r_s
tangent wheel rate       = k_s r_s^2 + F_s dr_s/dj
```

Export both the common transformed material rate k_s r_s^2 and the full tangent rate. The second term retains load/preload acting through changing leverage. Derive stops and bar contributions consistently from their stored energy. For bars, retain left/right cross-coupling in the 4 x 4 wheel stiffness matrix; an axle bar cannot be represented by four independent wheel springs.

Report front/rear axle heave stiffness, front/rear elastic roll stiffness, total elastic roll stiffness, pitch stiffness, and front elastic roll-stiffness fraction. That fraction is not total lateral-load-transfer distribution; geometric load transfer, unsprung effects and external loads are separate vehicle-model concerns.

### Body stiffness matrix

For total elastic stored energy U(q), export the gradient g = dU/dq and tangent matrix K = d²U/dq². The actual generalized suspension restoring reaction is -g; do not hide this sign convention.

The matrix contains heave, roll and pitch rates and all three pairwise couplings:

```text
          h          phi          theta
h       K_hh        K_hphi       K_htheta
phi     K_phih      K_phiphi     K_phitheta
theta   K_thetah    K_thetaphi    K_thetatheta
```

K_hh is N/m; rotational diagonal entries are N m/rad; displacement/angle coupling entries are N/rad (equivalently the corresponding moment/displacement units with radians dimensionless). Supply per-entry units and coordinate definitions, not one unit for the whole matrix. The rotational generalized reactions are conjugate to the declared finite rotation coordinates; they should not be assumed to equal arbitrary world-axis torque components at finite attitude.

Compute the Hessian of the composed geometry/energy map, including preload and geometric second derivatives, rather than only multiplying spring rates by squared motion ratios. Report each component's contribution and their sum.

Each diagonal is a constrained rate with the other coordinates held fixed. If free-to-relax effective rates are provided, label their released coordinates and obtain them by a Schur complement only where the eliminated block is well-conditioned. Do not silently substitute a relaxed rate for a constrained one.

### Tyre-inclusive ride rates

Reserve the name ride rate for explicitly tyre-inclusive results. For an uncoupled scalar linear check, k_ride = k_w k_t/(k_w + k_t). Keep suspension-only wheel rate available separately. This distinction is material to model reuse.

For coupled nonlinear suspension, solve the declared local vertical-wheel/tyre equilibrium and condense those internal coordinates from the total-energy Hessian:

```text
K_effective = H_qq - H_qu solve(H_uu, H_uq)
```

Here u comprises vertical wheel/tyre degrees of freedom. Include tyre-force derivatives and any applicable geometric terms before condensation. Singular, non-converged or unsupported contact states produce invalid tyre-inclusive outputs, not fabricated series rates. If no tyre parameters are supplied, those fields are explicitly unavailable.

Roll-centre/roll-axis migration and anti-dive/anti-squat can follow after the core map is validated. Their force-path and contact assumptions need their own definitions; they are not prerequisites for the requested motion-ratio and elastic-rate plots.

## Study and plotting workflow

Provide three standard 1D studies: heave at held roll/pitch, roll at held heave/pitch, and pitch at held heave/roll. Allow nonzero held operating points. Add metric heatmaps for heave-roll, heave-pitch and roll-pitch slices, and a bounded 3D heave-roll-pitch export grid.

Use front/rear and left/right comparisons, setup overlays, component contributions and reference-point markers. Show invalid regions as gaps or masked cells. Put held coordinates, rate definition and units in plot labels. Output PNG plus SVG/PDF for reports; no suspension animation is required.

Proposed command flow, to be implemented:

```text
vd-suspension validate <geometry-config>
vd-suspension run <study-config> --output <run-directory>
vd-suspension plot <map.h5> --metric <metric-name>
```

Running a study writes the resolved input configuration, a run summary, map data and selected plots. Report sample count, valid/invalid counts, actual runtime, output size and precise result paths. Changing only a plot must not rerun the geometry solver.

## Export contract for future vehicle models

Use HDF5 as the authoritative array-based format, with companion JSON metadata and optional long-form CSV for inspection. Provide a MATLAB reader and an explicit axis-order/shape round-trip test; Python/MATLAB array conventions must be tested, not assumed. A MAT export can be added if a concrete consumer requires it.

Proposed HDF5 groups:

```text
/meta                     schema version, tool version, hashes, conventions
/axes/heave_m             [Nh]
/axes/roll_rad            [Nr]
/axes/pitch_rad           [Np]
/axes/corner              FL, FR, RL, RR
/pose/corners/...         [Nh, Nr, Np, 4]
/pose/components/...      component-indexed compression, force and gains
/pose/reactions/...       energy, gradients, wheel reactions
/pose/stiffness/...       wheel [4,4], body [3,3], component contributions
/pose/validity/...        masks, status codes, residuals, derivative quality
/corner_maps/FL/...       reusable jounce-indexed geometry/actuation tables
/corner_maps/FR/...
/corner_maps/RL/...
/corner_maps/RR/...
```

Each metric records its axes, SI unit, physical definition, held/released-coordinate conditions and validity association. Include the resolved geometry/setup/study configuration, reference pose and preloads, road/tyre assumptions, timestamp, solver tolerances, source/config hashes and tool revision when available.

Export both raw reusable corner geometry and the assembled setup-dependent pose map. Geometry alone can support future camber/toe and damper transformations; pose reactions and stiffness support a faster lookup-based vehicle model. Preserve forces as well as rates because incremental stiffness alone cannot reconstruct absolute loads.

Invalid numeric results use NaN plus an explicit validity mask and reason code. Consumers reject unsupported major schema versions, out-of-domain queries and interpolation stencils containing invalid points. Default to no extrapolation and no silent clipping. Supply a small interpolation API in Python and equivalent MATLAB reader behavior with consistent units and axis order.

## Delivery sequence and acceptance evidence

1. **Conventions and schema:** document config/export versions and all signs, frames and boundary conditions for the selected Python implementation. Create synthetic direct and rocker fixtures. Validate malformed, mirrored and asymmetric configurations.
2. **Corner kinematics and actuation:** solve wishbone/toe/rocker closure, establish continuation and limits, produce camber/toe/travel and motion-ratio curves. Verify independent constraint residuals, known constant-ratio examples, sweep reversal and mirror symmetry.
3. **Elastic forces and rates:** add springs, preload, stops and front/rear bars. Verify force/energy consistency, the preload-dependent geometric term, and bar cross-coupling. Equal motion in a symmetric ideal bar fixture must yield no twist; differential motion must yield the expected roll contribution.
4. **Full vehicle pose maps:** add finite body rotations, prescribed road support, operating-point slices, full Hessian and axle breakdowns. At the symmetric linear reference check K_heave = sum(k_i), K_roll = sum(k_i y_i²), K_pitch = sum(k_i x_i²), with the expected coupling terms. Check Hessian symmetry and independent finite-difference convergence at selected interior and boundary points.
5. **Plots and model exports:** create all requested slices/heatmaps and the bounded 3D map; verify plotting/export consistency, schema metadata, invalid masks and Python/MATLAB round trips using deliberately unequal axis lengths and uniquely identifiable values. Compare interpolated valid points with direct solves.
6. **Tyre-inclusive ride results and actual-car validation:** validate the scalar series-rate limit and coupled condensation against independent local perturbations. Import supplied hard points, inspect nominal closure and compare selected metrics against CAD, measurements or another independently trusted kinematics tool. Synthetic examples are not validation of the real vehicle.

Before claiming the first release complete, all solver, plotting and interoperability acceptance checks above must pass on the synthetic fixtures. Real-vehicle validation is an additional check once actual data are supplied; report it as unverified until then. Report numerical residuals and derivative error/convergence; choose final tolerances from fixture scaling and conditioning rather than undocumented optimizer defaults. Establish a runtime baseline on a declared grid and hardware before promising performance.

## Deferred work

Dynamic simulation, suspension animation, structural/bushing compliance, geometry optimization, a CAD importer, steering sweeps, frequency/modal analysis, third-spring mechanisms, aero coupling and changes to the existing VD vehicle model are outside the first release. Keep the energy-component interface extensible enough to add a third spring later without implementing it now.

## Reference checks

- [OptimumG: Of springs and dampers](https://optimumg.com/wp-content/uploads/2021/10/racecar-2020_11.pdf) explains squared motion-ratio transformation, differing ratio conventions and tyre stiffness in series. This plan deliberately declares spring/wheel motion ratio; its full tangent-rate formulation is derived from energy and extends the simple transformed-rate expression.
- [MathWorks: Independent Suspension — Double Wishbone](https://www.mathworks.com/help/vdynblks/ref/independentsuspensiondoublewishbone.html) documents wheel orientation outputs and a suspension block using a different, z-down frame. A future adapter must explicitly transform conventions; sharing a topology name does not guarantee compatible signs or physics.
- [MathWorks: h5read](https://www.mathworks.com/help/matlab/ref/h5read.html) documents MATLAB HDF5 dataset access. Actual multi-axis interoperability remains an implementation acceptance check.
