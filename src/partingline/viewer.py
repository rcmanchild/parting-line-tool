"""Lightweight interactive viewer (plain VTK, which cadquery-ocp already ships).

    partingline view part.step --pull +Z
    partingline view part.step --split --screenshot shot.png    # headless

Faces are coloured by class (cavity red, core blue, vertical yellow, mixed
grey, undercut purple) or, after a split, by cavity/core.  The parting line is
drawn as tubes; straight bridges are orange.  Every recompute goes through
job.run, exactly like the CLI.

Mouse: left-drag rotate, right-drag/wheel zoom, middle-drag pan, click = pick face.
Keys : x/y/z  pull +X/+Y/+Z (shift: -X/-Y/-Z)     f  flip pull
       p      pull from the picked face (plane normal / cylinder or cone axis)
       v      cycle vertical rule (top/bottom/middle)
       s      toggle cavity/core split colouring
       a      rank pull directions (--auto); 1-5 use that rank
       e      export (formats from --formats) to --out
       r      reset camera                         q  quit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import job
from . import segments as sg
from .analysis import VERTICAL_RULES, Options
from .auto import rank_directions, table
from .export import summary_text
from .faces import CAVITY, CORE, MIXED, VERTICAL
from .pull import describe, pull_from_face
from .split import CAVITY_SIDE
from .undercut import mesh_arrays, mesh_deflection

COLORS = {
    CAVITY: (220, 70, 60),
    CORE: (60, 110, 220),
    VERTICAL: (235, 200, 40),
    MIXED: (150, 150, 150),
    "undercut": (150, 60, 200),
    "conflict": (255, 140, 0),
    "line": (20, 20, 20),
    "bridge": (255, 140, 0),
}

PRESET_KEYS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}

HELP = ("click: pick face   x/y/z (shift: -): pull   f: flip   p: pull from picked face   "
        "v: vertical rule   s: split   a: auto, 1-5: rank   e: export   r: reset view   q: quit")


def _vtk():
    import vtk  # noqa: PLC0415  (imported lazily: the CLI must work without a display)
    return vtk


@dataclass
class ViewState:
    pull: np.ndarray
    opts: Options
    split: bool = False
    formats: tuple = ("step", "split", "surface")
    outdir: Path | None = None
    picked: int | None = None                 # 0-based face of the *displayed* mesh
    candidates: list = field(default_factory=list)
    message: str = ""


class Viewer:
    """Builds the VTK scene for one model; `recompute` re-runs the analysis."""

    def __init__(self, model, state: ViewState, size=(1200, 850), offscreen: bool = False):
        vtk = _vtk()
        self.vtk = vtk
        self.model = model
        self.state = state
        self.result = None
        self.lin = mesh_deflection(model.diag) * 4.0
        self._mesh_cache: dict = {}

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.renderer.SetBackground2(0.82, 0.86, 0.92)
        self.renderer.GradientBackgroundOn()
        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(*size)
        self.window.SetWindowName(f"partingline - {model.import_info.path}")
        if offscreen:
            self.window.SetOffScreenRendering(1)
        self.window.AddRenderer(self.renderer)

        self.surface_actor = vtk.vtkActor()
        self.surface_actor.GetProperty().SetInterpolationToFlat()
        self.surface_actor.GetProperty().SetAmbient(0.25)
        self.renderer.AddActor(self.surface_actor)
        self.line_actor = vtk.vtkActor()
        self.bridge_actor = vtk.vtkActor()
        self.pick_actor = vtk.vtkActor()
        self.arrow_actor = vtk.vtkActor()
        for a in (self.line_actor, self.bridge_actor, self.pick_actor, self.arrow_actor):
            self.renderer.AddActor(a)

        self.summary = vtk.vtkTextActor()
        self.summary.GetTextProperty().SetFontFamilyToCourier()
        self.summary.GetTextProperty().SetFontSize(12)
        self.summary.GetTextProperty().SetColor(0.05, 0.05, 0.05)
        self.summary.SetPosition(10, 30)
        self.status = vtk.vtkTextActor()
        self.status.GetTextProperty().SetFontSize(13)
        self.status.GetTextProperty().SetColor(0.1, 0.1, 0.4)
        self.status.SetPosition(10, 8)
        self.renderer.AddViewProp(self.summary)
        self.renderer.AddViewProp(self.status)
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(1e-4)

    # ---------------------------------------------------------------- analysis
    def recompute(self) -> None:
        st = self.state
        self.result, _ = job.run(self.model, st.pull, st.opts, ("split",) if st.split else ())
        st.picked = None
        self._refresh()

    def export(self) -> dict:
        st = self.state
        outdir = st.outdir or Path(f"{Path(self.model.import_info.path).stem}_parting")
        self.result, written = job.run(self.model, st.pull, st.opts, st.formats, outdir)
        st.message = f"wrote {len(written)} file(s) to {outdir}"
        self._refresh()
        return written

    def run_auto(self) -> list:
        cands, info = rank_directions(self.model, low_draft_deg=self.state.opts.low_draft_deg,
                                      draft_tol_deg=self.state.opts.draft_tol_deg)
        print(table(cands, info))
        self.state.candidates = cands
        self.state.message = "auto: " + "  ".join(f"{i}) {c.label}" for i, c in enumerate(cands, 1))
        return cands

    # ------------------------------------------------------------------ scene
    def _displayed(self):
        """(shape, faces, per-face RGB, per-face label) of what is on screen."""
        res = self.result
        sp = res.split if res is not None else None
        if self.state.split and sp is not None:
            conflicts = {n - 1 for n in sp.conflicts}
            cols, labels = [], []
            for i, side in enumerate(sp.side):
                name = CAVITY if side == CAVITY_SIDE else CORE
                cols.append(COLORS["conflict"] if i in conflicts else COLORS[name])
                labels.append(f"split face {i + 1}: {name}" + (" (conflict)" if i in conflicts else ""))
            return sp.model.shape, sp.model.faces, cols, labels
        cols, labels = [], []
        for fr in (res.faces if res is not None else []):
            cols.append(COLORS["undercut"] if fr.undercut else COLORS[fr.cls])
            dr = ("" if fr.min_draft_deg is None
                  else f", draft {fr.min_draft_deg:.2f}..{fr.max_draft_deg:.2f} deg")
            labels.append(f"face {fr.number}: {fr.type}, {fr.cls}{', UNDERCUT' if fr.undercut else ''}"
                          f", {fr.area:.2f} mm2{dr}")
        return self.model.shape, self.model.faces, cols, labels

    def _mesh(self, shape, faces):
        key = id(shape)
        if key not in self._mesh_cache:
            self._mesh_cache = {key: mesh_arrays(shape, faces, self.lin)}
        return self._mesh_cache[key]

    def _refresh(self) -> None:
        vtk = self.vtk
        shape, faces, cols, self._labels = self._displayed()
        V, T, owner = self._mesh(shape, faces)
        self._owner = owner
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        for p in V:
            pts.InsertNextPoint(*p)
        cells = vtk.vtkCellArray()
        for t in T:
            cells.InsertNextCell(3, [int(t[0]), int(t[1]), int(t[2])])
        poly.SetPoints(pts)
        poly.SetPolys(cells)
        rgb = vtk.vtkUnsignedCharArray()
        rgb.SetNumberOfComponents(3)
        rgb.SetName("rgb")
        fid = vtk.vtkIntArray()
        fid.SetName("face")
        for f in owner:
            rgb.InsertNextTuple3(*(cols[f] if f < len(cols) else COLORS[MIXED]))
            fid.InsertNextValue(int(f))
        poly.GetCellData().SetScalars(rgb)
        poly.GetCellData().AddArray(fid)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.ComputeCellNormalsOn()
        normals.SplittingOff()
        normals.ConsistencyOff()
        normals.AutoOrientNormalsOff()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        self.surface_actor.SetMapper(mapper)

        res = self.result
        r = 0.003 * self.model.diag
        loops = [lp.points for lp in res.loops] if res is not None else []
        bridges = [s.points for s in res.segments if s.kind == sg.BRIDGE] if res is not None else []
        self._tubes(self.line_actor, loops, r, COLORS["line"])
        self._tubes(self.bridge_actor, bridges, 2.0 * r, COLORS["bridge"])
        self._arrow()
        self._highlight()
        self._text()

    def _tubes(self, actor, polylines, radius, color) -> None:
        vtk = self.vtk
        app = vtk.vtkAppendPolyData()
        n = 0
        for P in polylines:
            if len(P) < 2:
                continue
            line = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            for p in P:
                pts.InsertNextPoint(*p)
            ids = vtk.vtkCellArray()
            ids.InsertNextCell(len(P), list(range(len(P))))
            line.SetPoints(pts)
            line.SetLines(ids)
            clean = vtk.vtkCleanPolyData()
            clean.SetInputData(line)
            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(clean.GetOutputPort())
            tube.SetRadius(radius)
            tube.SetNumberOfSides(10)
            tube.CappingOn()
            app.AddInputConnection(tube.GetOutputPort())
            n += 1
        actor.SetVisibility(n > 0)
        if n:
            m = vtk.vtkPolyDataMapper()
            m.SetInputConnection(app.GetOutputPort())
            m.ScalarVisibilityOff()
            actor.SetMapper(m)
            actor.GetProperty().SetColor(*(c / 255 for c in color))

    def _arrow(self) -> None:
        vtk = self.vtk
        lo, hi = self._bounds()
        c = 0.5 * (lo + hi)
        d = self.state.pull / np.linalg.norm(self.state.pull)
        length = 0.18 * self.model.diag
        top = max(float(np.dot(np.array([x, y, z]) - c, d))
                  for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2]))
        start = c + d * (top + 0.03 * self.model.diag)
        src = vtk.vtkArrowSource()
        src.SetTipResolution(24)
        src.SetShaftResolution(24)
        # orient the +x arrow along d
        x = d
        tmp = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = np.cross(tmp, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        mat = vtk.vtkMatrix4x4()
        for i in range(3):
            mat.SetElement(i, 0, x[i] * length)
            mat.SetElement(i, 1, y[i] * length)
            mat.SetElement(i, 2, z[i] * length)
            mat.SetElement(i, 3, start[i])
        tf = vtk.vtkTransform()
        tf.SetMatrix(mat)
        tpf = vtk.vtkTransformPolyDataFilter()
        tpf.SetTransform(tf)
        tpf.SetInputConnection(src.GetOutputPort())
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(tpf.GetOutputPort())
        self.arrow_actor.SetMapper(m)
        self.arrow_actor.GetProperty().SetColor(0.1, 0.6, 0.2)

    def _bounds(self):
        from OCP.Bnd import Bnd_Box  # noqa: PLC0415
        from OCP.BRepBndLib import BRepBndLib  # noqa: PLC0415
        box = Bnd_Box()
        BRepBndLib.Add_s(self.model.shape, box)
        x0, y0, z0, x1, y1, z1 = box.Get()
        return np.array([x0, y0, z0]), np.array([x1, y1, z1])

    def _highlight(self) -> None:
        """Outline the picked face's triangles."""
        vtk = self.vtk
        f = self.state.picked
        if f is None:
            self.pick_actor.SetVisibility(False)
            return
        mapper = self.surface_actor.GetMapper()
        mapper.Update()
        sel = vtk.vtkThreshold()
        sel.SetInputData(mapper.GetInput())
        sel.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "face")
        sel.SetLowerThreshold(f - 0.5)
        sel.SetUpperThreshold(f + 0.5)
        geo = vtk.vtkGeometryFilter()
        geo.SetInputConnection(sel.GetOutputPort())
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(geo.GetOutputPort())
        m.ScalarVisibilityOff()
        self.pick_actor.SetMapper(m)
        p = self.pick_actor.GetProperty()
        p.SetRepresentationToWireframe()
        p.SetColor(0.0, 0.9, 0.3)
        p.SetLineWidth(2.0)
        self.pick_actor.SetVisibility(True)

    def _text(self) -> None:
        st = self.state
        text = summary_text(self.model, self.result) if self.result is not None else ""
        if st.split and self.result is not None and self.result.split is None:
            text += "\n(no split: no parting line)"
        self.summary.SetInput(text)
        lines = []
        if st.picked is not None and st.picked < len(self._labels):
            lines.append("picked " + self._labels[st.picked])
        if st.message:
            lines.append(st.message)
        lines.append(HELP)
        self.status.SetInput("\n".join(lines))
        self.summary.SetPosition(10, 30 + 16 * (len(lines) - 1))

    def reset_camera(self) -> None:
        cam = self.renderer.GetActiveCamera()
        d = self.state.pull / np.linalg.norm(self.state.pull)
        lo, hi = self._bounds()
        c = 0.5 * (lo + hi)
        side = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        view = 0.8 * side + 0.45 * d + 0.4 * np.cross(d, side)
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + view * 3.0 * self.model.diag))
        cam.SetViewUp(*d)
        self.renderer.ResetCamera(self.surface_actor.GetBounds())
        cam.Zoom(1.15)
        self.renderer.ResetCameraClippingRange()

    # --------------------------------------------------------------- actions
    def pick(self, x: int, y: int) -> int | None:
        self.picker.Pick(x, y, 0, self.renderer)
        cell = self.picker.GetCellId()
        self.state.picked = None if cell < 0 else int(self._owner[cell])
        self.state.message = ""
        self._highlight()
        self._text()
        return self.state.picked

    def set_pull(self, d) -> None:
        d = np.asarray(d, dtype=float)
        self.state.pull = d / np.linalg.norm(d)
        self.state.message = f"pull {describe(self.state.pull)}"
        self.recompute()

    def pull_from_picked(self) -> None:
        f = self.state.picked
        if f is None:
            self.state.message = "pick a face first"
            self._text()
            return
        if self.state.split and self.result is not None and self.result.split is not None:
            self.state.message = "press s to leave split colouring, then pick a model face"
            self._text()
            return
        try:
            d, how = pull_from_face(self.model, f + 1)
        except ValueError as exc:
            self.state.message = str(exc)
            self._text()
            return
        self.set_pull(d)
        self.state.message = f"pull {describe(self.state.pull)} from {how}"
        self._text()

    def key(self, sym: str, shift: bool = False) -> bool:
        """Handle one key; returns False to quit."""
        st = self.state
        k = sym.lower()
        if k in ("q", "escape"):
            return False
        if k in PRESET_KEYS:
            sign = -1.0 if (shift or sym.isupper()) else 1.0
            self.set_pull(sign * np.array(PRESET_KEYS[k]))
        elif k == "f":
            self.set_pull(-st.pull)
        elif k == "p":
            self.pull_from_picked()
        elif k == "v":
            rules = list(VERTICAL_RULES)
            st.opts.vertical = rules[(rules.index(st.opts.vertical) + 1) % len(rules)]
            st.message = f"vertical rule '{st.opts.vertical}'"
            self.recompute()
        elif k == "s":
            st.split = not st.split
            st.message = "split colouring " + ("on" if st.split else "off")
            self.recompute()
        elif k == "a":
            self.status.SetInput("ranking pull directions ...")
            self.window.Render()
            self.run_auto()
            self._text()
        elif k in "12345" and len(k) == 1:
            i = int(k)
            if i <= len(st.candidates):
                self.set_pull(st.candidates[i - 1].d)
                st.message = f"auto rank {i}: {describe(st.pull)}"
                self._text()
            else:
                st.message = "press a first to rank pull directions"
                self._text()
        elif k == "e":
            self.status.SetInput("exporting ...")
            self.window.Render()
            self.export()
        elif k == "r":
            self.reset_camera()
        return True

    # ---------------------------------------------------------------- output
    def screenshot(self, path: Path) -> Path:
        vtk = self.vtk
        self.window.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.window)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        return Path(path)

    def show(self) -> None:
        vtk = self.vtk
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(self.window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)
        press = {}

        def on_char(obj, ev):  # replaces VTK's default one-letter shortcuts
            pass

        def on_key(obj, ev):
            if not self.key(iren.GetKeySym() or "", bool(iren.GetShiftKey())):
                iren.GetRenderWindow().Finalize()
                iren.TerminateApp()
                return
            self.window.Render()

        def on_press(obj, ev):
            press["xy"] = iren.GetEventPosition()
            style.OnLeftButtonDown()

        def on_release(obj, ev):
            style.OnLeftButtonUp()
            xy = iren.GetEventPosition()
            p0 = press.get("xy")
            if p0 is not None and abs(xy[0] - p0[0]) + abs(xy[1] - p0[1]) <= 3:
                self.pick(*xy)
                self.window.Render()

        style.AddObserver("CharEvent", on_char)
        style.AddObserver("KeyPressEvent", on_key)
        style.AddObserver("LeftButtonPressEvent", on_press)
        style.AddObserver("LeftButtonReleaseEvent", on_release)
        iren.Initialize()
        self.window.Render()
        iren.Start()


def offscreen_available() -> bool:
    """True when VTK can render without a display (probed in a subprocess,
    because a missing GL stack can crash the interpreter)."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    code = ("import vtk; w=vtk.vtkRenderWindow(); w.SetOffScreenRendering(1); w.SetSize(8,8);"
            "w.AddRenderer(vtk.vtkRenderer()); w.Render(); print('ok')")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and "ok" in out.stdout
