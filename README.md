# partingline

Parting-line analysis for injection / compression mold design. Load a STEP file,
choose a pull direction, get the parting line as STEP / IGES / DXF / CSV plus a
JSON report. Pure Python on OpenCascade (OCP); no commercial CAD needed.

It also splits the part's faces into cavity and core halves, builds a parting
surface, ranks pull directions automatically (`--auto`), and has a small 3D
viewer (`partingline view`).

## Install

Python 3.11 or newer on Windows, macOS or Linux.

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate`. macOS/Linux: `source .venv/bin/activate`. Then:

```bash
pip install -e .
```

Runtime dependencies are `cadquery-ocp` (OpenCascade bindings) and `numpy`.
To run the tests as well, use `pip install -e ".[test]"`, which adds `pytest`
and `cadquery` (used only to generate the test fixtures).

## Example

```bash
partingline analyze prop.step --pull +Z --formats step,dxf,csv --out prop_parting
```

The command prints a summary like this one (a real folding-prop blade):

```
Faces   : cavity 26 (3268.5 mm2)  core 22 (642.9 mm2)  vertical 3 (81.9 mm2)  mixed 12 (2635.8 mm2)
Vertical: 3 face(s), 81.9 mm2 need draft  faces 1,3,17
Undercut: 0 face(s), 0.0 mm2
Line    : 2 loop(s), 2 closed, 0 open, total 365.369 mm
          #1: closed    352.489 mm  [silhouette 345.5, vertical 7.0]
          #2: closed     12.881 mm  [vertical 12.9]
```

To get everything (line, cavity/core face shells and parting surface) in one go:

```bash
partingline analyze prop.step --pull +Z --formats all
```

Run `partingline faces part.step` to list the faces with their numbers, types,
plane normals and cylinder axes. Use the numbers with `--pull-face`.

## Pull direction

| Input | Meaning |
|---|---|
| `--pull +Z` | presets `+X -X +Y -Y +Z -Z` |
| `--pull 0,0,1` | any vector; it is normalised |
| `--pull-face 17` | planar face: its outward normal. Cylinder, cone or revolved face: its axis (e.g. the hub bore) |
| `--flip` | reverse whichever direction you gave |
| `--auto` | rank directions and use the best one (see below) |

The cavity half moves along +pull and the core half along -pull.

## Options

| Option | Default | |
|---|---|---|
| `--draft-tol DEG` | 0.5 | faces whose draft stays within ±this angle are *vertical* (t = sin(tol)) |
| `--vertical top\|bottom\|middle` | top | where the line goes on vertical bands (see below) |
| `--gap-tol MM` | 0.01 | chaining tolerance |
| `--bridge-tol MM` | 0.5 | close gaps left in the line up to this length with straight *bridges* (0 = off); every bridge is listed in the warnings and in `report.json` |
| `--formats` | step | any of `step,iges,dxf,csv` (line), `split` (cavity/core faces), `surface` (parting surface), or `all` |
| `--surface sweep\|ruled` | sweep | parting-surface shape (see Outputs) |
| `--surface-distance MM` | 20 | sweep distance, or margin of the ruled rectangle |
| `--surface-height MM` | mean line height | ruled mode: where the rectangle plane sits along the pull |
| `--low-draft DEG` | 1.0 | report the area with less draft than this |
| `--no-undercut` | | skip the ray-cast check |
| `--undercut-frac`, `--undercut-min-area` | 0.05, 0.5 mm² | thresholds for flagging a face as an undercut |
| `--no-heal` | | skip ShapeFix / sewing / crease splitting |

## Outputs

* `parting_line.step`: the line as 3D curves. OpenCascade writes wireframes as a
  `GEOMETRIC_CURVE_SET`, so there is one trimmed curve per parting segment;
  `report.json` records how the segments form loops.
* `parting_line.iges`: the same curves.
* `parting_line.dxf`: R12, the line projected on the plane normal to the pull.
  Axes are stored in the report (`e1` is global X projected onto that plane).
* `parting_line.csv`: `loop, closed, index, x, y, z`, sampled to 1 µm deflection.
* `cavity_faces.step`, `core_faces.step` (`split`): the part's faces cut along the
  parting line (OpenCascade general fuse) and sorted into the two halves, as shells.
  Each half's open boundary equals the parting line; the report gives face
  counts, areas and any conflicts (faces that face the other way from their half).
* `parting_surface.step` (`surface`): `sweep` pushes the line outward,
  perpendicular to the pull, by `--surface-distance`. `ruled` rules the line to a
  rectangle in a plane normal to the pull, `--surface-distance` beyond the part.
  Bores (inner loops) are capped. Self-intersections are reported, not hidden.
* `report.json`: units, healing, face classes and areas, min draft per side,
  vertical faces, undercuts, loops (count, closure, length, make-up), warnings
  and timings.

All lengths are in mm. If a STEP file declares another unit, the OpenCascade
reader converts it to mm. That conversion is reported as a warning in the
summary and in the report; it never happens silently.

## Choosing the pull direction automatically

```bash
partingline auto part.step                       # just show the ranking
partingline analyze part.step --auto --formats all
partingline analyze part.step --auto --auto-pick 2
```

About 500 directions over a hemisphere (plus +X, +Y, +Z) are scored on a coarse
mesh. The best 15 are scored again on a fine mesh, and the top 5 are shown:

    score = W x undercut area + near-zero-draft area      (W = --auto-undercut-weight, default 1)

Ties go to the shallower part (smaller depth along the pull). `--auto-samples`
and `--auto-top` set how many directions are tried and shown. The ranking is
stored in `report.json`, together with where +X, +Y and +Z ranked.

Note: on parts with walls that have no draft, a few degrees of tilt gives those
walls "some draft" and so lowers the score. Expect the top ranks to be slight
tilts of the obvious direction (on a blade with an undrafted hub, +Z ranked 16th
and a 21° tilt ranked 1st). Check the table, use `--auto-pick`, or raise
`--auto-undercut-weight` when undercuts matter more than draft.

## Viewer

```bash
partingline view part.step --pull +Z
partingline view part.step --auto --split
partingline view part.step --split --screenshot part.png     # headless PNG
```

A plain VTK window (VTK comes with `cadquery-ocp`, so nothing extra to install).
Faces are coloured cavity red, core blue, vertical yellow, mixed grey and
undercut purple. After a split (`s`) they are coloured by half, with conflicts in
orange. The parting line is drawn as black tubes, bridges in orange, and the
pull direction as a green arrow. The summary is shown in the corner.

| Input | Action |
|---|---|
| left-drag / right-drag / middle-drag / wheel | rotate / zoom / pan / zoom |
| click | pick a face: number, type, class, area, draft range |
| `x` `y` `z` (shift: negative) | pull along an axis |
| `f` | flip the pull |
| `p` | pull from the picked face (plane normal, cylinder/cone axis) |
| `v` | cycle the vertical rule (top, bottom, middle) |
| `s` | toggle cavity/core split colouring |
| `a`, then `1`-`5` | rank directions (`--auto`), then use rank N |
| `e` | export `--formats` (default `step,split,surface`) to `--out` |
| `r` / `q` | reset view / quit |

Every recompute runs the same code as `partingline analyze`. On a Linux server
without a display, `--screenshot` needs an EGL or OSMesa library
(e.g. `apt install libegl1` or `libosmesa6`).

## How it works

Everything runs on the exact B-rep. A mesh is used only for ray casting.

1. **Heal**: ShapeFix, sewing when there are free edges, reversal of inside-out
   solids, and splitting of B-spline faces at C0 creases so that creases become
   real edges.
2. **Classify faces** by sampling n·d over a UV grid and along every edge:
   *cavity* (n·d > t), *core* (n·d < −t), *vertical* (|n·d| ≤ t) or *mixed*.
3. **Smooth silhouettes** inside mixed faces: marching squares on the zero set
   of n·d in UV. Every crossing is solved to |n·d| < 1e-12, the curve is
   densified by Newton projection to 1 µm sagitta, then clipped to the trimmed
   face. Sign jumps across creases are rejected.
4. **Edges**: along every edge, each side is cavity, core or vertical. Where the
   two sides are cavity and core, that stretch of edge is parting line. This
   catches sharp edges such as a prop trailing edge, and also smooth
   silhouettes that lie exactly on an edge or seam. Silhouette end points are
   added as edge samples, so short stretches are not skipped.
5. **Vertical bands** (rule `top`, or `bottom`): for points on the band, rays
   along ±pull test which way the steel can release. Where nothing is in the
   way the rule decides, so the line sits on the band's top (or bottom) edge.
   Where something attached blocks one direction, for example a blade on a
   cylindrical hub, part of the band must go with the other half. The line then
   drops straight along the pull from the blade's root points to the band edge.
   `middle` uses a plane section at mid-height; it is meant for simple walls.
6. **Undercuts**: rays from samples on every face along each direction that
   face could release in. A face is flagged when every allowed direction is
   blocked for at least `--undercut-frac` of its area. Undercuts are reported,
   not resolved (no slides in v1).
7. **Chaining** into loops with a 0.01 mm gap tolerance. Slivers shorter than
   the gap tolerance and dead-end spurs shorter than 0.1 mm are pruned. Gaps up
   to `--bridge-tol` that are still open are closed with straight, reported
   bridges. The report gives open loops, junctions, loop count and lengths.
8. **Split** (`--formats split`): silhouettes become edges on their faces
   (built in UV), band generators become edges on the band. All are cut into
   the part with a general fuse that shares vertices at the line's nodes. Faces
   are then flood-filled across non-parting edges, and each group goes to the
   half that most of its area faces. Small slivers next to the line (where a
   silhouette nearly touches an edge) are moved so the halves meet exactly
   along the line.
9. **Parting surface** (`--formats surface`): built from the closed loops, as
   described under Outputs.

## Tests

```bash
python -m pytest
python scripts/stress_directions.py 24     # analyse + split every fixture at 24 random pulls
```

The viewer tests are skipped when offscreen rendering is not available.

The fixtures are generated with cadquery into `fixtures/generated/`:

* a sphere
* a horizontal cylinder
* boxes with 2° and 0° walls
* a T-slot block
* a twisted NACA 4412 blade
* a hub with three fused blades

Drop real parts (`*.step` / `*.stp`) into `fixtures/` and they are tested too:
the line must close, have no junctions, and finish in under 30 s. A sidecar
`fixtures/<name>.json` sets the pull direction, as `{"pull": "+Z"}` or
`{"pull_face": 17}`. Without one, the axis of the largest cylindrical face is used.

## Known limitations

* A band is not split where something that isn't attached to it casts a shadow
  on it (for example an overhang over a separate wall). The undercut check
  still flags such cases.
* `--vertical middle` uses a flat section, so it does not follow bands whose
  top and bottom edges are at varying heights.
* Undercut and low-draft areas are sample-based estimates: 8–32 samples per
  face direction.
* Undercuts are found and reported, but not resolved: there are no slides or
  lifters, and an undercut region goes with whichever half the rules give it.
