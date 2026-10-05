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

Wishbone chassis-side hard points use `inboard_forward` and
`inboard_rearward` under each corner's `upper` and `lower` arms. The eight
front/rear names map to the configuration paths as follows:

| Hard-point name | Left corner path | Right corner path |
| --- | --- | --- |
| Front upper forward | `corners.FL.upper.inboard_forward` | `corners.FR.upper.inboard_forward` |
| Front upper rearward | `corners.FL.upper.inboard_rearward` | `corners.FR.upper.inboard_rearward` |
| Front lower forward | `corners.FL.lower.inboard_forward` | `corners.FR.lower.inboard_forward` |
| Front lower rearward | `corners.FL.lower.inboard_rearward` | `corners.FR.lower.inboard_rearward` |
| Rear upper forward | `corners.RL.upper.inboard_forward` | `corners.RR.upper.inboard_forward` |
| Rear upper rearward | `corners.RL.upper.inboard_rearward` | `corners.RR.upper.inboard_rearward` |
| Rear lower forward | `corners.RL.lower.inboard_forward` | `corners.RR.lower.inboard_forward` |
| Rear lower rearward | `corners.RL.lower.inboard_rearward` | `corners.RR.lower.inboard_rearward` |

The chassis convention is x forward, so the existing larger-x pivot is
`inboard_forward` and the smaller-x pivot is `inboard_rearward`, including
the rear fixtures whose x coordinates are negative. Right-side reflection
changes y only and keeps the forward/rearward names attached to the same
physical hard points.

The studies include one-dimensional heave examples and a bounded 7 x 5 x 3
heave/roll/pitch grid. Study file references are relative to each study file.
After the command-line runner is available, the direct example can be checked
and run with:

```powershell
uv run vd-suspension validate configs/suspension_geometry/geometries/synthetic_direct.yaml
uv run vd-suspension run configs/suspension_geometry/studies/direct_heave.yaml --output outputs/suspension_geometry/direct-heave
```

The resulting map uses the configured rigid reference geometry and declared
study boundary conditions. It does not imply static vehicle equilibrium,
measured road contact, or validation against supplied vehicle hard points.
