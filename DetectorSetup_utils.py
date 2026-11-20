import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import pandas as pd

DeadtimeMode = Literal["NE", "E", "NE-E", "E-NE"]
VALID_DEADTIME_MODES = {"NE", "E", "NE-E", "E-NE"}

ExperimentMode = Literal["Critical", "Subcritical", "Reactivity"]
VALID_EXPERIMENT_MODES = {"Critical", "Subcritical", "Reactivity"}


@dataclass
class DetectorChannel:
    """Metadata for a single detector position / electronic chain."""
    fc_name: str
    fc_id: Optional[str] = None
    region: Optional[str] = None
    location: Optional[Tuple[int, int]] = None
    deposit_nuclide: Optional[str] = None
    cable: Optional[str] = None
    acquisition: Optional[str] = None
    digitalization: Optional[str] = None
    time_resolution: Optional[float] = None
    gain: Optional[float] = None
    lld: Optional[int] = None
    caen_channel: Optional[int] = None
    tau_ne: Optional[float] = None
    tau_e: Optional[float] = None
    deadtime_mode: Optional[DeadtimeMode] = None

    def __post_init__(self) -> None:
        if self.deadtime_mode is not None and self.deadtime_mode not in VALID_DEADTIME_MODES:
            raise ValueError(
                f"Invalid deadtime_mode '{self.deadtime_mode}'. Must be one of {sorted(VALID_DEADTIME_MODES)}."
            )
        if self.gain is not None:
            self.gain = float(self.gain)
        if self.time_resolution is not None:
            self.time_resolution = float(self.time_resolution)
        if self.tau_ne is not None:
            self.tau_ne = float(self.tau_ne)
        if self.tau_e is not None:
            self.tau_e = float(self.tau_e)
        if self.lld is not None:
            self.lld = int(self.lld)
        if self.caen_channel is not None:
            self.caen_channel = int(self.caen_channel)
        if self.location is not None:
            self.location = (int(self.location[0]), int(self.location[1]))

    @property
    def label(self) -> str:
        """Returns 'fc_name fc_id' if fc_id is set, otherwise 'fc_name'."""
        return f"{self.fc_name} {self.fc_id}" if self.fc_id is not None else self.fc_name


@dataclass
class ExperimentSetup:
    """
    Container combining reactor/run operating conditions with a list of DetectorChannel objects.
    """
    experiment_name: str = ""
    power: Optional[float] = None
    lin_vi: Optional[float] = None
    mode: Optional[ExperimentMode] = None
    subcriticality_level: Optional[Union[float, str]] = None
    cr1_height: Optional[float] = None
    cr2_height: Optional[float] = None
    time_start: Optional[Union[str, datetime]] = None
    time_stop: Optional[Union[str, datetime]] = None
    objective: Optional[str] = None
    detectors: List[DetectorChannel] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode not in VALID_EXPERIMENT_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of {sorted(VALID_EXPERIMENT_MODES)}."
            )
        if self.subcriticality_level is not None:
            if self.mode != "Subcritical":
                raise ValueError(
                    f"'subcriticality_level' is only allowed when mode='Subcritical' (got mode={self.mode!r})."
                )
            if isinstance(self.subcriticality_level, (int, float)) and not isinstance(self.subcriticality_level, bool):
                self.subcriticality_level = float(self.subcriticality_level)
            elif isinstance(self.subcriticality_level, str):
                self.subcriticality_level = self.subcriticality_level.strip()
            else:
                raise TypeError(
                    f"'subcriticality_level' must be float or str (got {type(self.subcriticality_level).__name__})."
                )
        if self.power is not None:
            self.power = float(self.power)
        if self.lin_vi is not None:
            self.lin_vi = float(self.lin_vi)
        if self.cr1_height is not None:
            self.cr1_height = float(self.cr1_height)
        if self.cr2_height is not None:
            self.cr2_height = float(self.cr2_height)
        if isinstance(self.time_start, datetime):
            self.time_start = self.time_start.isoformat()
        if isinstance(self.time_stop, datetime):
            self.time_stop = self.time_stop.isoformat()

    @property
    def duration_seconds(self) -> Optional[float]:
        """Returns elapsed run duration in seconds if both time_start and time_stop are set."""
        if self.time_start is None or self.time_stop is None:
            return None
        t0 = datetime.fromisoformat(self.time_start)
        t1 = datetime.fromisoformat(self.time_stop)
        return (t1 - t0).total_seconds()

    # ------------------------------------------------------------
    # Lookup & Table Helpers
    # ------------------------------------------------------------
    def by_caen_channel(self, ch_id: int) -> DetectorChannel:
        """Retrieve detector metadata matching a CAEN digitizer channel ID."""
        for det in self.detectors:
            if det.caen_channel == ch_id:
                return det
        raise KeyError(f"No detector mapped to CAEN channel {ch_id}")

    def by_detector(self, fc_name: str, fc_id: Optional[str] = None) -> DetectorChannel:
        """
        Retrieve detector metadata by fc_name and optional fc_id.
        Also accepts a combined label like 'CFUF 34' if fc_id is omitted.
        """
        for det in self.detectors:
            if det.fc_name == fc_name and det.fc_id == fc_id:
                return det
            if fc_id is None and det.label == fc_name:
                return det
        target = f"{fc_name} {fc_id}" if fc_id is not None else fc_name
        raise KeyError(f"No detector found matching '{target}'")

    def filter_region(self, region: str) -> List[DetectorChannel]:
        """Return a list of DetectorChannel objects in the specified region."""
        return [
            d for d in self.detectors
            if d.region is not None and d.region.lower() == region.lower()
        ]

    def active_caen_map(self) -> Dict[int, DetectorChannel]:
        """Returns {caen_channel: DetectorChannel} for all connected CAEN channels."""
        return {d.caen_channel: d for d in self.detectors if d.caen_channel is not None}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the detector list into a clean Pandas DataFrame."""
        rows = []
        for d in self.detectors:
            row = {"label": d.label, **asdict(d)}
            row["location"] = (
                f"({d.location[0]:+d},{d.location[1]:+d})" if d.location is not None else None
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_series(self) -> pd.Series:
        """Returns the experiment-level operating parameters as a Pandas Series."""
        return pd.Series({
            "experiment_name": self.experiment_name,
            "mode": self.mode,
            "subcriticality_level": self.subcriticality_level,
            "power": self.power,
            "lin_vi": self.lin_vi,
            "cr1_height": self.cr1_height,
            "cr2_height": self.cr2_height,
            "time_start": self.time_start,
            "time_stop": self.time_stop,
            "duration_seconds": self.duration_seconds,
            "objective": self.objective,
            "num_detectors": len(self.detectors),
        })

    # ------------------------------------------------------------
    # Serialization & File I/O (JSON, CSV & HDF5)
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "power": self.power,
            "lin_vi": self.lin_vi,
            "mode": self.mode,
            "subcriticality_level": self.subcriticality_level,
            "cr1_height": self.cr1_height,
            "cr2_height": self.cr2_height,
            "time_start": self.time_start,
            "time_stop": self.time_stop,
            "objective": self.objective,
            "detectors": [asdict(d) for d in self.detectors],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentSetup":
        detectors = [DetectorChannel(**d) for d in payload.get("detectors", [])]
        return cls(
            experiment_name=payload.get("experiment_name", ""),
            power=payload.get("power"),
            lin_vi=payload.get("lin_vi"),
            mode=payload.get("mode"),
            subcriticality_level=payload.get("subcriticality_level"),
            cr1_height=payload.get("cr1_height"),
            cr2_height=payload.get("cr2_height"),
            time_start=payload.get("time_start"),
            time_stop=payload.get("time_stop"),
            objective=payload.get("objective"),
            detectors=detectors,
        )

    def to_json(self, filepath: str) -> None:
        """Save full experiment + detector list to JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "ExperimentSetup":
        """Load full experiment + detector list from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_csv(self, filepath: str) -> None:
        """Export detector table to CSV."""
        self.to_dataframe().to_csv(filepath, index=False)

    def attach_to_hdf5(self, h5_path: str) -> None:
        """Embed the full ExperimentSetup JSON into an HDF5 file's root attributes."""
        import tables as tb
        with tb.open_file(h5_path, mode="a") as h5:
            h5.root._v_attrs["experiment_setup_json"] = json.dumps(self.to_dict())

    @classmethod
    def from_hdf5(cls, h5_path: str) -> "ExperimentSetup":
        """Read an embedded ExperimentSetup out of an HDF5 file."""
        import tables as tb
        with tb.open_file(h5_path, mode="r") as h5:
            raw = h5.root._v_attrs["experiment_setup_json"]
        return cls.from_dict(json.loads(raw))

    def __repr__(self) -> str:
        sub_str = f" ({self.subcriticality_level})" if self.subcriticality_level is not None else ""
        return (
            f"<ExperimentSetup '{self.experiment_name}' | mode={self.mode}{sub_str} | "
            f"power={self.power} | {len(self.detectors)} detectors>"
        )


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    exp = ExperimentSetup(
        experiment_name="Run_042",
        power=15.0,
        lin_vi=1.24e-9,
        mode="Subcritical",
        subcriticality_level="45 pcm",  # Accepts either float (e.g., -45.0) or str (e.g., "20 mm", "45 pcm")
        cr1_height=420.5,
        cr2_height=310.0,
        time_start="2026-11-17T09:15:00",
        time_stop="2026-11-17T10:00:00",
        objective="Rossi-alpha reactivity measurement at subcritical level 45 pcm",
        detectors=[
            DetectorChannel(
                fc_name="CFUF",
                fc_id="34",
                region="Fuel",
                location=(+1, +1),
                deposit_nuclide="U235",
                cable="X6",
                acquisition="PHSPA A7",
                digitalization="CAEN DT5730",
                time_resolution=10e-9,
                gain=10296.0,
                lld=376,
                caen_channel=7,
                tau_ne=120e-9,
                tau_e=50e-9,
                deadtime_mode="NE-E",
            ),
            DetectorChannel(fc_name="CFUL", fc_id="676", region="Fuel", location=(-1, +1), cable="X3", acquisition="PHSPA A5", gain=9979.0, lld=391, caen_channel=6),
            DetectorChannel(fc_name="CFUL", fc_id="627", region="Fuel", location=(+3, +3), cable="X2", acquisition="PHSPA A0", gain=9803.0, lld=485, caen_channel=5),
            DetectorChannel(fc_name="CFUL", fc_id="629", region="Fuel", location=(+4, +4), cable="X15", acquisition="ADS E", lld=400, caen_channel=3),
            DetectorChannel(fc_name="HE*", region="Pb", location=(+2, -3)),
            DetectorChannel(fc_name="CFUL", fc_id="600", region="Graphite", location=(-1, +5), cable="X5", acquisition="PHSPA A3", lld=425, caen_channel=2),
            DetectorChannel(fc_name="CFUL", fc_id="659", region="Graphite", location=(-5, +2), cable="X7", acquisition="ADS G", lld=420, caen_channel=1),
            DetectorChannel(fc_name="CFUL", fc_id="628", region="Graphite", location=(+5, -1), cable="X14", acquisition="ADS F", lld=439, caen_channel=4),
            DetectorChannel(fc_name="CFUL", fc_id="658", region="Graphite", location=(-4, -4), cable="X1", acquisition="ADS D", lld=498),
        ],
    )

    exp.to_json("run_042_setup.json")
    reloaded = ExperimentSetup.from_json("run_042_setup.json")
    print(reloaded)
    print(reloaded.by_caen_channel(7).label)  # "CFUF 34"