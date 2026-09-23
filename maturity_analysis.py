"""Connect the v6 capture GUI to the September 23 maturity/DUD pipeline.

The pooled calibration uses the new pipeline unchanged. Older PCA/regression
pickles use v6's original analysis and cannot report a calibrated DUD.
This module never initializes cameras, GPIO, or Tk.
"""
from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

pmes = import_module("Shaoqi_codes.09_23_26_main")


def discover_model_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pkl"),
        key=lambda path: path.name.lower(),
    )


class MaturityAnalyzer:
    def __init__(self, yolo_path: str | Path, device: str = "cpu"):
        self.yolo_path = Path(yolo_path).resolve()
        self.device = device
        self._model_signature = None
        self._model = None
        self._yolo_signature = None
        self._yolo = None

    @staticmethod
    def _signature(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size

    def load_model(self, model_path: str | Path) -> dict[str, Any]:
        path = Path(model_path).resolve()
        if path.suffix.lower() != ".pkl" or not path.is_file():
            raise FileNotFoundError(f"Select an existing .pkl maturity model: {path}")
        signature = self._signature(path)
        if signature == self._model_signature:
            return self._model

        raw = pmes.load_model_pickle(path)
        if not isinstance(raw, dict) or not callable(getattr(raw.get("pca"), "transform", None)):
            raise ValueError(f"{path.name} must contain a fitted 'pca' object.")
        # A calibration with DUD metadata must pass the new schema validation;
        # an invalid calibration must never silently fall back to regression.
        has_dud_metadata = any(key in raw for key in (
            "pc1_nodes", "dud_nodes", "mapping", "thresholds", "thresholds_pc1",
        ))
        if "reg" in raw and not has_dud_metadata:
            if not callable(getattr(raw["reg"], "predict", None)):
                raise ValueError(f"{path.name} must contain a fitted 'reg' predictor.")
            loaded = {"kind": "legacy_regression", "pca": raw["pca"], "reg": raw["reg"]}
        else:
            try:
                calibration = pmes.validate_pca_model(raw)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path.name} is not a compatible DUD calibration: {exc}") from exc
            loaded = {"kind": "pooled_pca_dud", "calibration": calibration}
        loaded["path"] = path
        loaded["sha256"] = pmes.sha256_file(path)
        # Commit only after validation, so a failed switch cannot poison the cache.
        self._model = loaded
        self._model_signature = signature
        return loaded

    def analyze(
        self, npy_path: str | Path, model_path: str | Path, output_dir: str | Path,
        legacy_processor: Callable | None = None, legacy_options: dict | None = None,
    ) -> dict[str, Any]:
        npy_path = Path(npy_path).resolve()
        model = self.load_model(model_path)
        if not self.yolo_path.is_file():
            raise FileNotFoundError(f"YOLO segmentation model not found: {self.yolo_path}")
        yolo_signature = self._signature(self.yolo_path)
        if yolo_signature != self._yolo_signature:
            self._yolo = pmes.load_yolo_model(self.yolo_path)
            self._yolo_signature = yolo_signature

        result_dir = Path(output_dir).resolve() / f"{npy_path.stem}__{model['path'].stem}"
        result_dir.mkdir(parents=True, exist_ok=True)
        if model["kind"] == "pooled_pca_dud":
            args = pmes.parse_args(["--input", str(npy_path), "--device", str(self.device)])
            parameters = pmes.resolved_parameters(args, model["calibration"])
            result = pmes.process_one_npy(
                npy_path, result_dir, model["calibration"], self._yolo, parameters,
                white_reference=None, save_numeric_maps=True,
            )
            values = [row["continuous_maturity_index_0_1"] for row in result["pod_results"]]
            decision, outputs = result["decision"], result["outputs"]
            result.update({
                "n_peanuts": result["pod_count"],
                "mean_maturity": float(np.mean(values)),
                "std_maturity": float(np.std(values)),
                "days_left": decision["mpb_derived_dud_at_q95_maturity"],
                "brown_ratio": decision["brown_ratio"],
                "black_ratio": decision["black_ratio"],
                "brown_black_ratio": decision["brown_black_ratio"],
                "heatmap_path": outputs["continuous_maturity_index_blue_red_map"],
                "dud_map_path": outputs["continuous_dud_map"],
                "warnings": result["quality"]["warnings"],
            })
        else:
            if legacy_processor is None:
                raise ValueError("This model requires the v6 PCA/regression processor.")
            result = legacy_processor(
                npy_path=str(npy_path), pca_model=model["pca"], reg_model=model["reg"],
                yolo_model=self._yolo, out_dir=str(result_dir), **(legacy_options or {}),
            )
            result.update({
                "days_left": None, "brown_ratio": None, "black_ratio": None,
                "brown_black_ratio": None,
                "warnings": ["DUD unavailable: this regression model has no PC1-to-DUD calibration."],
                "outputs": {
                    "continuous_maturity_index_map": result["heatmap_path"],
                    "annotated_pods": result["annotated_path"],
                    "eroded_masks": result["binary_mask_path"],
                    "result_json": str(result_dir / "results.json"),
                    "summary_text": str(result_dir / "summary.txt"),
                },
            })
            summary = [
                f"File: {npy_path.name}", f"Peanut number: {result['n_peanuts']}",
                f"Mean maturity index: {result['mean_maturity']}",
                f"Std maturity index: {result['std_maturity']}", *result["warnings"],
            ]
            (result_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

        result.update({
            "model_name": model["path"].name,
            "selected_model": {
                "path": str(model["path"]), "sha256": model["sha256"], "kind": model["kind"],
            },
            "result_directory": str(result_dir),
        })
        with open(result["outputs"]["result_json"], "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=pmes.json_default)
        with open(result["outputs"]["summary_text"], "a", encoding="utf-8") as handle:
            handle.write(f"Model: {model['path'].name}\n")
        return result
