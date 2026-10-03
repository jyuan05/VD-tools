# Synthetic suspension geometry examples

These examples exercise the configuration contract for the suspension geometry
tool. They are invented fixtures for software development; they do not describe
or validate a real vehicle.

The direct example places its spring between the chassis and lower wishbone. The
rocker example uses the rocker pivot, axial direction, pushrod point, and
rocker-mounted spring endpoint. Both examples provide matching linear spring,
preload, anti-roll bar, bump-stop, and tyre settings. Each tyre has an explicit
nonzero reference force for loaded local equilibrium calculations.

Geometry and setup values use SI units. The loader also accepts explicit mm,
deg, and kN units and returns canonical SI dictionaries. The chassis frame
is right-handed with x forward, y left, and z up. A reflected right-side axial
direction uses det(S) S; points use S, where S = diag(1, -1, 1). The
spindle axis is normalized as an axial direction oriented by the right-hand
rule. The reference origin is retained in the resolved geometry.

The studies include one-dimensional heave examples and a bounded 7 x 5 x 3
heave/roll/pitch grid. Study file references are relative to each study file.
After the command-line runner is available, the direct example can be checked
and run with:

\`\`\`powershell
uv run vd-suspension validate configs/suspension_geometry/geometries/synthetic_direct.yaml
uv run vd-suspension run configs/suspension_geometry/studies/direct_heave.yaml --output outputs/suspension_geometry/direct-heave
\`\`\`

The resulting map uses the configured rigid reference geometry and declared
study boundary conditions. It does not imply static vehicle equilibrium,
measured road contact, or validation against supplied vehicle hard points.
