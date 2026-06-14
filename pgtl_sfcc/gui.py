"""Tkinter GUI for interactive PG-TL SFCC curve plotting."""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    VERTICAL,
    BooleanVar,
    Button,
    Checkbutton,
    Entry,
    Frame,
    Label,
    Listbox,
    Scrollbar,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from typing import Mapping, Sequence

import numpy as np

from .features import ALL_FEATURES


DEFAULT_SAMPLE = {
    "Sand": 10.0,
    "Silt": 60.0,
    "Clay": 30.0,
    "SOM": 0.12,
    "BD": 800.0,
    "Salinity": 0.05,
    "Porosity": 0.65,
    "Saturation": 0.95,
    "PL": 28.0,
    "LL": 52.0,
    "S_a": 40.0,
}
DEFAULT_T_MIN_C = -20.0
DEFAULT_T_MAX_C = 2.0
DEFAULT_N_POINTS = 150
MAX_IMPORT_CURVES = 5


@dataclass
class CurveSpec:
    """One curve definition for the GUI and CSV import."""

    label: str
    sample: dict[str, float]
    t_min_c: float = DEFAULT_T_MIN_C
    t_max_c: float = DEFAULT_T_MAX_C
    n_points: int = DEFAULT_N_POINTS

    def temperatures(self) -> np.ndarray:
        return np.linspace(self.t_min_c, self.t_max_c, self.n_points)


def _coerce_float(value: str, *, name: str, row_number: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV row {row_number}: '{name}' must be numeric, got {value!r}.") from exc


def curve_spec_from_mapping(row: Mapping[str, str | float], *, row_number: int = 1) -> CurveSpec:
    """Build a curve specification from a CSV row or form mapping."""

    missing = [name for name in ALL_FEATURES if str(row.get(name, "")).strip() == ""]
    if missing:
        raise ValueError(f"CSV row {row_number}: missing required field(s): {', '.join(missing)}.")

    sample = {
        name: _coerce_float(str(row[name]).strip(), name=name, row_number=row_number)
        for name in ALL_FEATURES
    }
    label = str(row.get("label") or row.get("Label") or f"Curve {row_number}").strip()
    t_min = _coerce_float(str(row.get("T_min_c", DEFAULT_T_MIN_C)), name="T_min_c", row_number=row_number)
    t_max = _coerce_float(str(row.get("T_max_c", DEFAULT_T_MAX_C)), name="T_max_c", row_number=row_number)
    n_points = int(_coerce_float(str(row.get("n_points", DEFAULT_N_POINTS)), name="n_points", row_number=row_number))
    if t_min >= t_max:
        raise ValueError(f"CSV row {row_number}: T_min_c must be less than T_max_c.")
    if n_points < 2:
        raise ValueError(f"CSV row {row_number}: n_points must be at least 2.")
    return CurveSpec(label=label, sample=sample, t_min_c=t_min, t_max_c=t_max, n_points=n_points)


def load_curve_specs_csv(path: str | Path, *, max_curves: int = MAX_IMPORT_CURVES) -> list[CurveSpec]:
    """Load up to `max_curves` curve definitions from a CSV file."""

    specs: list[CurveSpec] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or has no header row.")
        for row_number, row in enumerate(reader, start=2):
            if len(specs) >= max_curves:
                break
            specs.append(curve_spec_from_mapping(row, row_number=row_number))
    if not specs:
        raise ValueError("CSV file did not contain any curve rows.")
    return specs


class SFCCGui:
    """Small desktop GUI for plotting one or more SFCC curves."""

    def __init__(self, root: Tk):
        self.root = root
        self.root.title("PG-TL SFCC Curve Explorer")
        self.root.geometry("1320x780")
        self.root.minsize(1080, 640)
        self.curves: list[tuple[CurveSpec, dict[str, np.ndarray | float]]] = []
        self.entries: dict[str, StringVar] = {}
        self.use_pitzer_iteration = BooleanVar(value=False)
        self._curve_counter = itertools.count(1)

        import matplotlib

        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        outer = Frame(self.root)
        outer.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=0, minsize=320)
        outer.grid_columnconfigure(1, weight=1)

        controls = Frame(outer, width=320)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls.grid_propagate(False)
        plot_frame = Frame(outer)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(8.6, 5.8), dpi=140)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Import PyTorch-backed inference after TkAgg is initialized to avoid
        # duplicate OpenMP runtime initialization in some conda environments.
        from .inference import PGTLSFCCModel

        self.model = PGTLSFCCModel()

        self._build_inputs(controls)
        self._build_curve_list(controls)
        self._set_form_from_spec(CurveSpec(label="Default", sample=dict(DEFAULT_SAMPLE)))
        self._redraw()

    def _build_inputs(self, parent: Frame) -> None:
        Label(parent, text="Soil inputs", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        for name in ["label", *ALL_FEATURES, "T_min_c", "T_max_c", "n_points"]:
            row = Frame(parent)
            row.pack(fill="x", pady=1)
            Label(row, text=name, width=11, anchor="w").pack(side=LEFT)
            var = StringVar()
            Entry(row, textvariable=var, width=16).pack(side=RIGHT)
            self.entries[name] = var

        Checkbutton(
            parent,
            text="Pitzer full iteration",
            variable=self.use_pitzer_iteration,
            command=self.refresh_predictions,
        ).pack(anchor="w", pady=(6, 2))

        button_grid = Frame(parent)
        button_grid.pack(fill="x", pady=(8, 4))
        Button(button_grid, text="Add curve", command=self.add_curve).pack(fill="x", pady=1)
        Button(button_grid, text="Update selected", command=self.update_selected).pack(fill="x", pady=1)
        Button(button_grid, text="Remove selected", command=self.remove_selected).pack(fill="x", pady=1)
        Button(button_grid, text="Clear all", command=self.clear_curves).pack(fill="x", pady=1)
        Button(button_grid, text="Import CSV (max 5)", command=self.import_csv).pack(fill="x", pady=(6, 1))

    def _build_curve_list(self, parent: Frame) -> None:
        Label(parent, text="Curves", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        list_frame = Frame(parent)
        list_frame.pack(fill=BOTH, expand=True)
        scrollbar = Scrollbar(list_frame, orient=VERTICAL)
        self.listbox = Listbox(list_frame, height=8, width=32, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

    def _set_form_from_spec(self, spec: CurveSpec) -> None:
        self.entries["label"].set(spec.label)
        for name, value in spec.sample.items():
            self.entries[name].set(f"{value:g}")
        self.entries["T_min_c"].set(f"{spec.t_min_c:g}")
        self.entries["T_max_c"].set(f"{spec.t_max_c:g}")
        self.entries["n_points"].set(str(spec.n_points))

    def _spec_from_form(self, fallback_label: str | None = None) -> CurveSpec:
        data = {key: var.get() for key, var in self.entries.items()}
        if not data.get("label") and fallback_label:
            data["label"] = fallback_label
        return curve_spec_from_mapping(data, row_number=1)

    def _predict(self, spec: CurveSpec) -> dict[str, np.ndarray | float]:
        return self.model.predict_sfcc_curve(
            spec.sample,
            spec.temperatures(),
            use_pitzer_iteration=bool(self.use_pitzer_iteration.get()),
        )

    def _label_for_list(self, spec: CurveSpec, result: Mapping[str, np.ndarray | float]) -> str:
        sal = spec.sample["Salinity"]
        som = spec.sample["SOM"]
        return f"{spec.label} | S={sal:g} mol/L, SOM={som:g}, n={float(result['n']):.3f}"

    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, END)
        for spec, result in self.curves:
            self.listbox.insert(END, self._label_for_list(spec, result))

    def _selected_index(self) -> int | None:
        selected = self.listbox.curselection()
        return int(selected[0]) if selected else None

    def add_curve(self) -> None:
        try:
            label = self.entries["label"].get().strip() or f"Curve {next(self._curve_counter)}"
            spec = self._spec_from_form(fallback_label=label)
            if not spec.label:
                spec.label = label
            self.curves.append((spec, self._predict(spec)))
            self._refresh_listbox()
            self.listbox.selection_clear(0, END)
            self.listbox.selection_set(len(self.curves) - 1)
            self._redraw()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot add curve", str(exc))

    def update_selected(self) -> None:
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("No curve selected", "Select a curve before updating it.")
            return
        try:
            spec = self._spec_from_form()
            self.curves[idx] = (spec, self._predict(spec))
            self._refresh_listbox()
            self.listbox.selection_set(idx)
            self._redraw()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot update curve", str(exc))

    def remove_selected(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        del self.curves[idx]
        self._refresh_listbox()
        self._redraw()

    def clear_curves(self) -> None:
        self.curves.clear()
        self._refresh_listbox()
        self._redraw()

    def refresh_predictions(self) -> None:
        idx = self._selected_index()
        try:
            self.curves = [(spec, self._predict(spec)) for spec, _result in self.curves]
            self._refresh_listbox()
            if idx is not None and idx < len(self.curves):
                self.listbox.selection_set(idx)
            self._redraw()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot refresh curves", str(exc))

    def import_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Import SFCC curve CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            specs = load_curve_specs_csv(path)
            self.curves = [(spec, self._predict(spec)) for spec in specs]
            self._refresh_listbox()
            self.listbox.selection_set(0)
            self._set_form_from_spec(specs[0])
            self._redraw()
            messagebox.showinfo("CSV imported", f"Imported {len(specs)} curve(s).")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Cannot import CSV", str(exc))

    def _on_select(self, _event=None) -> None:
        idx = self._selected_index()
        if idx is not None:
            self._set_form_from_spec(self.curves[idx][0])

    def _redraw(self) -> None:
        self.ax.clear()
        if not self.curves:
            self.ax.text(0.5, 0.5, "Add or import curves", transform=self.ax.transAxes, ha="center", va="center")
        for spec, result in self.curves:
            self.ax.plot(result["temperature_c"], result["theta_u"], lw=2.0, label=spec.label)
        self.ax.set_xlabel("Temperature (deg C)")
        self.ax.set_ylabel("Unfrozen water content, theta_u (m3/m3)")
        self.ax.set_title("PG-TL SFCC curves")
        self.ax.grid(True, alpha=0.25)
        if self.curves:
            self.ax.legend(loc="best", fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    root = Tk()
    SFCCGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
