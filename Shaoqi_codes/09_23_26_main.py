"""Run the manuscript-aligned PMES pipeline directly from three-band NPY images.

The NPY input must be H x W x bands and contain the 405, 720, and 760 nm
images at the band indices stored in the PCA calibration (normally 0, 1, 2).
The spectral values used by PCA are never contrast-stretched or normalized per
pod. Percentile scaling is used only for the pseudo-RGB YOLO/display image.

Processing follows the verified manuscript rerun: YOLO segmentation, explicit
duplicate suppression, a 104-instance tray cap, mask erosion, three-band pod
medians, fixed-PCA projection, five-class/continuous maturity estimation, and
MPB-derived DUD reporting. The tray output also includes the mature-tail Q95
DUD and Brown, Black, and combined Brown + Black proportions.

The DUD value is an operational translation of the MPB scale. It is not a
prospectively validated economically optimal digging date.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion


SCRIPT_DIR = Path(__file__).resolve().parent
CATEGORIES = ["white", "yellow", "orange", "brown", "black"]
DISPLAY = [name.title() for name in CATEGORIES]
CLASS_COLORS = {
    "white": "lightgray",
    "yellow": "gold",
    "orange": "darkorange",
    "brown": "saddlebrown",
    "black": "black",
}
WAVELENGTHS_NM = [405, 720, 760]
PIPELINE_VERSION = "pmes-article-inference-1.1.1"

# Verified manuscript rerun settings. Model metadata may supply the same values;
# CLI arguments are available only for explicitly documented sensitivity runs.
ARTICLE_DEFAULTS = {
    "confidence": 0.30,
    "iou_threshold": 0.00,
    "image_size": 640,
    "raw_max_detections": 300,
    "minimum_raw_mask_pixels": 500,
    "explicit_duplicate_iou_threshold": 0.80,
    "duplicate_containment_threshold": 0.90,
    "maximum_instances_after_deduplication": 104,
    "input_channel_order": (2, 1, 0),
    "erosion_kernel": 3,
    "erosion_iterations": 5,
    "minimum_retained_pixels_per_pod": 30,
}

DEFAULT_YOLO = SCRIPT_DIR / "Models" / "peanut_segmentation" / "04_26_26_peanut_seg.pt"
DEFAULT_POOLED_PCA = (
    SCRIPT_DIR
    / "Manuscript"
    / "Fig5_7_from_raw_images"
    / "03_additional_pca_analyses"
    / "results"
    / "analysis1_pooled_reference_model.pkl"
)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_device(value: str | int | None) -> str | int | None:
    if value is None or str(value).strip().lower() in {"", "auto", "none"}:
        return None
    return int(value) if re.fullmatch(r"-?\d+", str(value).strip()) else str(value)


def to_uint8(channel: np.ndarray) -> np.ndarray:
    """Robust display scaling; these values are never supplied to PCA."""
    values = np.asarray(channel, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.quantile(finite, [0.01, 0.99])
    if not high > low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.nan_to_num(
        (values - low) / (high - low), nan=0.0, posinf=1.0, neginf=0.0
    )
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def npy_to_yolo_image(cube: np.ndarray, channel_order: tuple[int, int, int]) -> np.ndarray:
    if cube.ndim != 3 or cube.shape[2] <= max(channel_order):
        raise ValueError(f"Cube shape {cube.shape} cannot provide YOLO channels {channel_order}.")
    return np.stack([to_uint8(cube[:, :, index]) for index in channel_order], axis=2)


def normalize_with_white_reference(cube: np.ndarray, reference_path: Path) -> np.ndarray:
    """Optionally divide a raw cube by a band-matched white-reference NPY.

    Use this only when the PCA calibration was fitted from equivalently
    normalized cubes. Project NPY files normally already use the calibration
    scale, so no extra normalization is applied by default.
    """
    reference = np.load(reference_path, allow_pickle=False)
    if reference.ndim == 1 and reference.shape[0] == cube.shape[2]:
        reference = reference.reshape(1, 1, -1)
    if reference.ndim != 3 or reference.shape[2] != cube.shape[2]:
        raise ValueError(
            f"White-reference shape {reference.shape} is incompatible with cube {cube.shape}."
        )
    if reference.shape[:2] not in {(1, 1), cube.shape[:2]}:
        raise ValueError(
            "White reference must be 1 x 1 x bands or match the sample height and width."
        )
    reference = np.asarray(reference, dtype=np.float64)
    finite_positive = reference[np.isfinite(reference) & (reference > 0)]
    if finite_positive.size == 0:
        raise ValueError("White reference contains no finite positive values.")
    epsilon = max(float(np.quantile(finite_positive, 0.001)) * 1e-6, 1e-12)
    return np.divide(
        np.asarray(cube, dtype=np.float64),
        reference,
        out=np.full(np.broadcast_shapes(cube.shape, reference.shape), np.nan, dtype=np.float64),
        where=np.isfinite(reference) & (reference > epsilon),
    )


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """Read NumPy 2 array pickles on the imaging box's NumPy 1 environment."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module.startswith("numpy._core."):
                return super().find_class(module.replace("numpy._core.", "numpy.core.", 1), name)
            raise


def load_model_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return _NumpyCompatibleUnpickler(handle).load()


def load_pca_model(path: Path) -> dict[str, Any]:
    return validate_pca_model(load_model_pickle(path))


def validate_pca_model(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or "pca" not in raw:
        raise ValueError("PCA model must be a dictionary containing a fitted 'pca' object.")

    if not callable(getattr(raw["pca"], "transform", None)):
        raise ValueError("The model's 'pca' object must provide transform().")

    categories = [str(x).lower() for x in raw.get("categories", CATEGORIES)]
    if categories != CATEGORIES:
        raise ValueError(f"Five classes in this order are required: {CATEGORIES}; got {categories}.")
    thresholds = np.asarray(raw.get("thresholds_pc1", raw.get("thresholds", [])), dtype=float)
    pc1_nodes = np.asarray(
        raw.get("pc1_nodes", raw.get("mapping", {}).get("pc1_nodes", [])), dtype=float
    )
    dud_nodes = np.asarray(
        raw.get("dud_nodes", raw.get("mapping", {}).get("dud_nodes", [])), dtype=float
    )
    centers_source = raw.get("centers_pc1_median", raw.get("centers"))
    if isinstance(centers_source, dict):
        centers = np.asarray([centers_source[name] for name in CATEGORIES], dtype=float)
    else:
        centers = np.asarray(centers_source, dtype=float)
    band_indices = tuple(int(x) for x in raw.get("band_indices", (0, 1, 2)))

    if (thresholds.shape != (4,) or not np.all(np.isfinite(thresholds))
            or not np.all(np.diff(thresholds) > 0)):
        raise ValueError("Model needs four strictly increasing PC1 class thresholds.")
    if (centers.shape != (5,) or not np.all(np.isfinite(centers))
            or not np.all(np.diff(centers) > 0)):
        raise ValueError("Model needs five strictly increasing PC1 class centers.")
    if (pc1_nodes.shape != (6,) or not np.all(np.isfinite(pc1_nodes))
            or not np.all(np.diff(pc1_nodes) > 0)):
        raise ValueError("Model needs six strictly increasing PC1-to-DUD nodes.")
    if (dud_nodes.shape != (6,) or not np.all(np.isfinite(dud_nodes))
            or not np.all(np.diff(dud_nodes) < 0)):
        raise ValueError("Model needs six strictly decreasing DUD nodes.")
    if len(band_indices) != 3 or len(set(band_indices)) != 3 or min(band_indices) < 0:
        raise ValueError(f"Expected three distinct non-negative PCA band indices, got {band_indices}.")
    return {
        "raw": raw,
        "pca": raw["pca"],
        "thresholds": thresholds,
        "centers": centers,
        "pc1_nodes": pc1_nodes,
        "dud_nodes": dud_nodes,
        "band_indices": band_indices,
        "model_version": raw.get("model_version", "pooled-reference-pca"),
    }


def load_yolo_model(weights: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required. Install it with: pip install ultralytics"
        ) from exc
    return YOLO(str(weights))


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def mask_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return 0.0, 0.0
    area_a, area_b = int(a.sum()), int(b.sum())
    union = area_a + area_b - intersection
    return intersection / union, intersection / min(area_a, area_b)


def deduplicate_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float,
    containment_threshold: float,
    maximum_instances: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    suppressed = 0
    for candidate in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        duplicate = False
        for accepted in kept:
            if box_iou(candidate["bbox_xyxy"], accepted["bbox_xyxy"]) < iou_threshold:
                continue
            overlap_iou, containment = mask_overlap(candidate["mask"], accepted["mask"])
            if overlap_iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break
        if duplicate:
            suppressed += 1
        else:
            kept.append(candidate)
    unique_before_cap = len(kept)
    kept = kept[:maximum_instances]
    audit = {
        "raw_detection_count": len(detections),
        "suppressed_duplicate_count": suppressed,
        "unique_before_cap_count": unique_before_cap,
        "truncated_after_deduplication_count": unique_before_cap - len(kept),
        "final_detection_count": len(kept),
    }
    return kept, audit


def segment_with_yolo(
    cube: np.ndarray, model: Any, options: dict[str, Any]
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, int]]:
    yolo_image = npy_to_yolo_image(cube, tuple(options["channel_order"]))
    kwargs: dict[str, Any] = {
        "source": yolo_image,
        "conf": options["confidence"],
        "iou": options["iou_threshold"],
        "imgsz": options["image_size"],
        "max_det": options["raw_max_detections"],
        "retina_masks": True,
        "verbose": False,
    }
    if options["device"] is not None:
        kwargs["device"] = options["device"]
    if options["class_id"] is not None:
        kwargs["classes"] = [options["class_id"]]
    result = model.predict(**kwargs)[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        audit = {key: 0 for key in (
            "raw_detection_count", "suppressed_duplicate_count", "unique_before_cap_count",
            "truncated_after_deduplication_count", "final_detection_count"
        )}
        return [], yolo_image, audit

    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    height, width = cube.shape[:2]
    detections: list[dict[str, Any]] = []
    for index, raw_mask in enumerate(masks):
        if raw_mask.shape != (height, width):
            raw_mask = np.asarray(
                Image.fromarray(np.rint(np.clip(raw_mask, 0, 1) * 255).astype(np.uint8)).resize(
                    (width, height), Image.Resampling.NEAREST
                ), dtype=float
            ) / 255.0
        mask = raw_mask > 0.5
        area = int(mask.sum())
        if area < options["minimum_raw_mask_pixels"]:
            continue
        detections.append({
            "mask": mask,
            "bbox_xyxy": [float(value) for value in boxes[index]],
            "confidence": float(confidences[index]),
            "class_id": int(classes[index]),
            "mask_pixels": area,
        })
    kept, audit = deduplicate_detections(
        detections,
        options["duplicate_iou_threshold"],
        options["duplicate_containment_threshold"],
        options["maximum_instances"],
    )
    return kept, yolo_image, audit


def ellipse_structure(kernel_size: int) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Erosion kernel must be a positive odd integer.")
    radius = kernel_size // 2
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return xx * xx + yy * yy <= radius * radius


def transform_pc1(pca: Any, pixels: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
    parts = [
        pca.transform(pixels[start:start + batch_size])[:, 0]
        for start in range(0, len(pixels), batch_size)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def pc1_to_stage(pc1: np.ndarray | float, centers: np.ndarray) -> np.ndarray | float:
    return np.interp(pc1, centers, np.arange(5, dtype=float), left=0.0, right=4.0)


def maturity_colormap(maximum_dud: float) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("peanut_dud", [
        (0.0, "black"), (21.0 / maximum_dud, "saddlebrown"),
        (31.0 / maximum_dud, "darkorange"), (45.0 / maximum_dud, "gold"),
        (73.0 / maximum_dud, "lightgray"), (1.0, "white"),
    ])


def maturity_index_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "peanut_maturity_index",
        [(0.0, "white"), (0.25, "gold"), (0.5, "darkorange"),
         (0.75, "saddlebrown"), (1.0, "black")],
    )


def blue_red_maturity_colormap() -> LinearSegmentedColormap:
    """High-contrast maturity scale: blue is less mature and red is more mature."""
    return LinearSegmentedColormap.from_list(
        "blue_red_maturity_index",
        [(0.0, "#2166AC"), (0.25, "#67A9CF"), (0.50, "#F7F7F7"),
         (0.75, "#EF8A62"), (1.0, "#B2182B")],
    )


def estimate_kde_bandwidth(values: np.ndarray, minimum: float = 1.0, maximum: float = 12.0) -> float:
    if len(values) <= 1:
        return minimum
    standard_deviation = float(np.std(values, ddof=1))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    robust_sigma = float((q75 - q25) / 1.349)
    positive = [scale for scale in (standard_deviation, robust_sigma) if scale > 0]
    scale = min(positive) if positive else minimum
    return float(np.clip(0.9 * scale * len(values) ** (-0.2), minimum, maximum))


def boundary_corrected_kde(
    values: np.ndarray, x_grid: np.ndarray, bandwidth: float, lower: float, upper: float
) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), lower, upper)
    x = x_grid[:, None]
    density = np.zeros_like(x_grid, dtype=float)
    normalizer = len(values) * bandwidth * np.sqrt(2.0 * np.pi)
    for reflected in (values, 2 * lower - values, 2 * upper - values):
        z = (x - reflected[None, :]) / bandwidth
        density += np.exp(-0.5 * z * z).sum(axis=1) / normalizer
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(integrate(density, x_grid))
    return density / area if area > 0 else density


def draw_annotation(image: Image.Image, bbox_xyxy: list[float], text: str) -> None:
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
    left, top, right, bottom = draw.textbbox((0, 0), text)
    text_width, text_height = right - left, bottom - top
    label_y = max(0, y1 - text_height - 9)
    draw.rectangle((x1, label_y, x1 + text_width + 8, label_y + text_height + 7), fill=(0, 255, 0))
    draw.text((x1 + 4, label_y + 3), text, fill=(0, 0, 0))


def save_union_mask(shape: tuple[int, int], masks: list[np.ndarray], path: Path) -> None:
    union = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        union = np.maximum(union, mask.astype(np.uint8) * 255)
    Image.fromarray(union).save(path)


def save_scalar_map(
    background: np.ndarray, values: np.ndarray, path: Path, title: str,
    colorbar_label: str, cmap: Any, vmin: float, vmax: float, alpha: float = 0.88,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.7))
    ax.imshow(background)
    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_density_profile(
    pod_duds: np.ndarray, dud_nodes: np.ndarray, q95_dud: float,
    class_counts: dict[str, int], class_ratios: dict[str, float], output: Path,
) -> None:
    maximum_dud = float(dud_nodes[0])
    x_grid = np.linspace(0.0, maximum_dud, 1001)
    density = boundary_corrected_kde(
        pod_duds, x_grid, estimate_kde_bandwidth(pod_duds), 0.0, maximum_dud
    )
    ymax = max(float(density.max()), 1e-9)
    fig, ax = plt.subplots(figsize=(13.4, 7.6))
    for index, category in enumerate(CATEGORIES):
        high, low = float(dud_nodes[index]), float(dud_nodes[index + 1])
        ax.axvspan(high, low, color=CLASS_COLORS[category], alpha=0.22, zorder=0)
        text_color = "white" if category in {"brown", "black"} else "black"
        ax.text((high + low) / 2, ymax * 0.97, category.title(), ha="center", va="top",
                color=text_color, fontweight="bold",
                bbox=dict(facecolor=CLASS_COLORS[category], edgecolor="none", alpha=0.72, pad=2))
    for boundary in dud_nodes[1:-1]:
        ax.axvline(boundary, color="dimgray", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.plot(x_grid, density, color="royalblue", linewidth=3.0, label="Pod-level maturity density")
    ax.fill_between(x_grid, density, color="royalblue", alpha=0.16)
    ax.scatter(pod_duds, np.zeros_like(pod_duds), marker="|", s=95, color="navy", alpha=0.45,
               label="Individual pod DUD")
    ax.axvline(q95_dud, color="crimson", linewidth=2.8,
               label=f"Maturity Q95 decision = {q95_dud:.1f} DUD")
    harvestable_dud = float(dud_nodes[1])
    smk_dud = float((dud_nodes[1] + dud_nodes[2]) / 2.0)
    ax.annotate("Harvestable\n(Y1 starts)", xy=(harvestable_dud, ymax * 0.72),
                xytext=(harvestable_dud + 7, ymax * 1.14), ha="center", color="darkgreen",
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="darkgreen", lw=2))
    ax.annotate("SMK\n(Y2 starts)", xy=(smk_dud, ymax * 0.58),
                xytext=(smk_dud + 4, ymax * 1.14), ha="center", color="purple",
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="purple", lw=2))
    brown_black = class_ratios["brown"] + class_ratios["black"]
    summary = (
        f"Pods: {len(pod_duds)}\nBrown: {class_counts['brown']} ({class_ratios['brown'] * 100:.1f}%)\n"
        f"Black: {class_counts['black']} ({class_ratios['black'] * 100:.1f}%)\n"
        f"Brown + Black: {brown_black * 100:.1f}%"
    )
    ax.text(0.985, 0.96, summary, transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.9))
    ax.set_xlim(maximum_dud, 0.0)
    ax.set_ylim(0.0, ymax * 1.31)
    ax.set_xlabel("Days Until Digging (less mature  →  more mature)")
    ax.set_ylabel("Pod-level probability density")
    ax.set_title("Peanut maturity density profile and farmer decision indicators")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def model_scale_check(model: dict[str, Any], pod_band_medians: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(
        model["raw"].get("calibration", {}).get(
            "reference_band_statistics_after_filtering", {}
        ).get("median", []), dtype=float
    )
    observed = np.median(pod_band_medians, axis=0) if len(pod_band_medians) else np.full(3, np.nan)
    ratio = (
        observed / reference
        if reference.shape == (3,) and np.all(np.isfinite(reference)) and np.all(reference != 0)
        else np.full(3, np.nan)
    )
    mismatch = bool(np.all(np.isfinite(ratio)) and (np.all(ratio < 0.1) or np.all(ratio > 10.0)))
    return {
        "reference_band_median": reference.tolist(),
        "input_pod_band_median": observed.tolist(),
        "input_over_reference_ratio": ratio.tolist(),
        "probable_scale_mismatch": mismatch,
    }


def write_pod_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "pod_id", "detection_index", "yolo_confidence", "bbox_xyxy",
        "mask_pixels_before_erosion", "finite_pixels_after_erosion",
        "band_405_median", "band_720_median", "band_760_median",
        "pc1_median", "pc1_iqr", "predicted_class", "predicted_class_index",
        "continuous_maturity_stage_0_4", "continuous_maturity_index_0_1",
        "mpb_derived_dud_days", "pc1_below_white_anchor", "pc1_above_black_anchor",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_one_npy(
    npy_path: Path, result_dir: Path, pca_model: dict[str, Any], yolo_model: Any,
    parameters: dict[str, Any], white_reference: Path | None, save_numeric_maps: bool,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    cube = np.load(npy_path, allow_pickle=False)
    if cube.ndim != 3 or cube.shape[2] <= max(pca_model["band_indices"]):
        raise ValueError(
            f"{npy_path} has shape {cube.shape}; model needs bands {pca_model['band_indices']}."
        )
    if white_reference is not None:
        cube = normalize_with_white_reference(cube, white_reference)
    selected = np.asarray(cube[:, :, pca_model["band_indices"]], dtype=np.float64)
    detections, yolo_image, detection_audit = segment_with_yolo(
        cube, yolo_model, parameters["segmentation"]
    )
    height, width = cube.shape[:2]
    eroded_masks: list[np.ndarray] = []
    accepted_detections: list[dict[str, Any]] = []
    pod_rows: list[dict[str, Any]] = []
    heat_dud = np.full((height, width), np.nan, dtype=float)
    heat_index = np.full((height, width), np.nan, dtype=float)
    structure = ellipse_structure(parameters["erosion"]["kernel_size_pixels"])

    for detection_index, detection in enumerate(detections):
        mask = binary_erosion(
            detection["mask"], structure=structure,
            iterations=parameters["erosion"]["iterations"], border_value=0,
        )
        rows, cols = np.where(mask)
        pixels = selected[rows, cols]
        finite = np.all(np.isfinite(pixels), axis=1)
        rows, cols, pixels = rows[finite], cols[finite], pixels[finite]
        if len(pixels) < parameters["erosion"]["minimum_retained_pixels_per_pod"]:
            continue
        pod_band_median = np.median(pixels, axis=0)
        pod_pc1 = float(pca_model["pca"].transform(pod_band_median.reshape(1, -1))[0, 0])
        predicted_index = int(np.digitize(pod_pc1, pca_model["thresholds"]))
        stage = float(pc1_to_stage(pod_pc1, pca_model["centers"]))
        maturity_index = stage / 4.0
        dud = float(np.interp(pod_pc1, pca_model["pc1_nodes"], pca_model["dud_nodes"]))
        pixel_pc1 = transform_pc1(pca_model["pca"], pixels)
        pixel_stage = np.asarray(pc1_to_stage(pixel_pc1, pca_model["centers"]), dtype=float)
        heat_dud[rows, cols] = np.interp(
            pixel_pc1, pca_model["pc1_nodes"], pca_model["dud_nodes"]
        )
        heat_index[rows, cols] = pixel_stage / 4.0
        q25, q75 = np.quantile(pixel_pc1, [0.25, 0.75])
        pod_result = {
            "pod_id": len(pod_rows) + 1,
            "detection_index": detection_index,
            "yolo_confidence": detection["confidence"],
            "bbox_xyxy": detection["bbox_xyxy"],
            "mask_pixels_before_erosion": detection["mask_pixels"],
            "finite_pixels_after_erosion": int(len(pixels)),
            "band_405_median": float(pod_band_median[0]),
            "band_720_median": float(pod_band_median[1]),
            "band_760_median": float(pod_band_median[2]),
            "pc1_median": pod_pc1,
            "pc1_iqr": float(q75 - q25),
            "predicted_class": CATEGORIES[predicted_index],
            "predicted_class_index": predicted_index,
            "continuous_maturity_stage_0_4": stage,
            "continuous_maturity_index_0_1": maturity_index,
            "mpb_derived_dud_days": dud,
            "pc1_below_white_anchor": bool(pod_pc1 < pca_model["pc1_nodes"][0]),
            "pc1_above_black_anchor": bool(pod_pc1 > pca_model["pc1_nodes"][-1]),
        }
        pod_rows.append(pod_result)
        eroded_masks.append(mask)
        accepted = dict(detection)
        accepted["eroded_mask"] = mask
        accepted["result"] = pod_result
        accepted_detections.append(accepted)

    if not pod_rows:
        raise ValueError("No usable pod remained after YOLO, de-duplication, and mask erosion.")
    annotated = Image.fromarray(yolo_image).convert("RGB")
    for detection in accepted_detections:
        result = detection["result"]
        draw_annotation(
            annotated, detection["bbox_xyxy"],
            f"{result['predicted_class'].title()} | MI {result['continuous_maturity_index_0_1']:.2f} | {result['mpb_derived_dud_days']:.1f} d",
        )

    files = {
        "annotated_pods": result_dir / "annotated_pods.png",
        "eroded_masks": result_dir / "eroded_masks.png",
        "continuous_maturity_index_map": result_dir / "continuous_maturity_index_map.png",
        "continuous_maturity_index_blue_red_map": (
            result_dir / "continuous_maturity_index_blue_red_map.png"
        ),
        "continuous_dud_map": result_dir / "continuous_dud_map.png",
        "digital_mpb_profile": result_dir / "digital_mpb_profile.png",
        "per_pod_csv": result_dir / "per_pod_predictions.csv",
        "result_json": result_dir / "results.json",
        "summary_text": result_dir / "summary.txt",
        "numeric_maps": result_dir / "numeric_maps.npz" if save_numeric_maps else None,
    }
    annotated.save(files["annotated_pods"])
    save_union_mask((height, width), eroded_masks, files["eroded_masks"])
    save_scalar_map(
        yolo_image, heat_index, files["continuous_maturity_index_map"],
        "Peanut maturity heat map",
        "Maturity index (0 = least mature, 1 = most mature)",
        maturity_index_colormap(), 0.0, 1.0,
    )
    gray_background = np.full((height, width, 3), 128, dtype=np.uint8)
    save_scalar_map(
        gray_background, heat_index, files["continuous_maturity_index_blue_red_map"],
        "Peanut maturity heatmap",
        "Maturity index (0 = less mature, 1 = more mature)",
        blue_red_maturity_colormap(), 0.0, 1.0, alpha=1.0,
    )
    save_scalar_map(
        yolo_image, heat_dud, files["continuous_dud_map"],
        "Peanut continuous heatmap",
        "MPB-derived Days Until Digging",
        maturity_colormap(float(pca_model["dud_nodes"][0])),
        0.0, float(pca_model["dud_nodes"][0]),
    )
    pod_pc1 = np.asarray([row["pc1_median"] for row in pod_rows], dtype=float)
    pod_duds = np.asarray([row["mpb_derived_dud_days"] for row in pod_rows], dtype=float)
    q95_pc1 = float(np.quantile(pod_pc1, 0.95))
    q95_dud = float(np.interp(q95_pc1, pca_model["pc1_nodes"], pca_model["dud_nodes"]))
    counts = Counter(row["predicted_class"] for row in pod_rows)
    class_counts = {category: int(counts[category]) for category in CATEGORIES}
    class_ratios = {category: class_counts[category] / len(pod_rows) for category in CATEGORIES}
    brown_black_ratio = class_ratios["brown"] + class_ratios["black"]
    save_density_profile(
        pod_duds, pca_model["dud_nodes"], q95_dud,
        class_counts, class_ratios, files["digital_mpb_profile"],
    )
    write_pod_csv(files["per_pod_csv"], pod_rows)
    if save_numeric_maps:
        np.savez_compressed(
            files["numeric_maps"], maturity_index=heat_index, dud_days=heat_dud,
            eroded_union=np.any(np.stack(eroded_masks), axis=0),
        )

    band_medians = np.asarray([
        [row["band_405_median"], row["band_720_median"], row["band_760_median"]]
        for row in pod_rows
    ], dtype=float)
    scale_quality = model_scale_check(pca_model, band_medians)
    anchor_clipped_fraction = float(np.mean([
        row["pc1_below_white_anchor"] or row["pc1_above_black_anchor"] for row in pod_rows
    ]))
    warnings: list[str] = []
    if scale_quality["probable_scale_mismatch"]:
        warnings.append("Probable input/calibration spectral-scale mismatch.")
    if anchor_clipped_fraction >= 0.20:
        warnings.append(f"{anchor_clipped_fraction * 100:.1f}% of pods lie beyond the PC1 anchors.")

    decision = {
        "mature_tail_quantile": 0.95,
        "q95_pc1_on_increasing_maturity_axis": q95_pc1,
        "mpb_derived_dud_at_q95_maturity": q95_dud,
        "display_dud_days": round(q95_dud, 1),
        "interpretation": (
            "DUD at the 95th percentile of increasing pod maturity; equivalent to the 5th "
            "percentile on the decreasing DUD axis. This is an MPB-derived indicator, not a "
            "prospectively validated optimal digging date."
        ),
        "class_counts": class_counts,
        "class_ratios": class_ratios,
        "brown_ratio": class_ratios["brown"],
        "black_ratio": class_ratios["black"],
        "brown_black_ratio": brown_black_ratio,
        "harvestable_boundary_dud_y1_start": float(pca_model["dud_nodes"][1]),
        "smk_boundary_dud_y2_start": float(
            (pca_model["dud_nodes"][1] + pca_model["dud_nodes"][2]) / 2.0
        ),
    }
    result = {
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "npy_file": str(npy_path.resolve()),
            "npy_sha256": sha256_file(npy_path),
            "cube_shape": list(cube.shape),
            "white_reference": str(white_reference.resolve()) if white_reference else None,
            "white_reference_applied": white_reference is not None,
        },
        "calibration": {
            "model_version": pca_model["model_version"],
            "band_indices": list(pca_model["band_indices"]),
            "wavelengths_nm": WAVELENGTHS_NM,
            "pc1_class_centers": pca_model["centers"].tolist(),
            "pc1_class_thresholds": pca_model["thresholds"].tolist(),
            "pc1_to_dud_nodes": pca_model["pc1_nodes"].tolist(),
            "dud_nodes": pca_model["dud_nodes"].tolist(),
        },
        "parameters": parameters,
        "pod_count": len(pod_rows),
        "detection_audit": detection_audit,
        "decision": decision,
        "quality": {
            "anchor_clipped_fraction": anchor_clipped_fraction,
            "spectral_scale_check": scale_quality,
            "warnings": warnings,
        },
        "pod_results": pod_rows,
        "outputs": {key: str(value) if value is not None else None for key, value in files.items()},
    }
    with open(files["result_json"], "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=json_default)
    summary_lines = [
        f"File: {npy_path.name}", f"Resolution: {width} x {height}",
        f"Raw YOLO detections: {detection_audit['raw_detection_count']}",
        f"Suppressed duplicates: {detection_audit['suppressed_duplicate_count']}",
        f"Usable pods after erosion: {len(pod_rows)}",
        f"MPB-derived Q95 maturity DUD: {q95_dud:.1f} days",
        f"Brown: {class_counts['brown']} ({class_ratios['brown'] * 100:.1f}%)",
        f"Black: {class_counts['black']} ({class_ratios['black'] * 100:.1f}%)",
        f"Brown + Black: {brown_black_ratio * 100:.1f}%",
        "DUD caveat: MPB-derived indicator; not yet a prospectively validated optimal digging date.",
    ]
    summary_lines.extend(f"WARNING: {warning}" for warning in warnings)
    files["summary_text"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n" + "=" * 72 + "\n" + "\n".join(summary_lines) + "\n" + "=" * 72)
    return result


def discover_npy(input_path: Path, recursive: bool, excluded_output: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            raise ValueError(f"Input file must be .npy: {input_path}")
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    iterator = input_path.rglob("*.npy") if recursive else input_path.glob("*.npy")
    files = []
    for path in sorted(iterator):
        resolved = path.resolve()
        try:
            resolved.relative_to(excluded_output.resolve())
            continue
        except ValueError:
            files.append(resolved)
    if not files:
        raise FileNotFoundError(f"No NPY files found under: {input_path}")
    return files


def result_directory(input_root: Path, file: Path, output: Path) -> Path:
    if input_root.is_file():
        return output / file.stem
    relative = file.relative_to(input_root.resolve())
    return output / relative.parent / relative.stem


def resolved_parameters(args: argparse.Namespace, model: dict[str, Any]) -> dict[str, Any]:
    raw = model["raw"]
    segmentation = raw.get("calibration", {}).get("segmentation", {})
    erosion = raw.get("calibration", {}).get("erosion", {})

    def choose(cli_value: Any, metadata: dict[str, Any], key: str, article_key: str) -> Any:
        return cli_value if cli_value is not None else metadata.get(key, ARTICLE_DEFAULTS[article_key])

    return {
        "segmentation": {
            "confidence": float(choose(args.yolo_confidence, segmentation, "confidence", "confidence")),
            "iou_threshold": float(choose(args.yolo_iou, segmentation, "iou_threshold", "iou_threshold")),
            "image_size": int(choose(args.yolo_imgsz, segmentation, "image_size", "image_size")),
            "raw_max_detections": int(choose(
                args.yolo_max_det, segmentation, "max_detections", "raw_max_detections"
            )),
            "minimum_raw_mask_pixels": int(choose(
                args.yolo_min_mask_area, segmentation, "minimum_raw_mask_pixels",
                "minimum_raw_mask_pixels"
            )),
            "duplicate_iou_threshold": float(choose(
                args.yolo_duplicate_iou, segmentation, "explicit_duplicate_iou_threshold",
                "explicit_duplicate_iou_threshold"
            )),
            "duplicate_containment_threshold": float(
                args.yolo_duplicate_containment
                if args.yolo_duplicate_containment is not None
                else ARTICLE_DEFAULTS["duplicate_containment_threshold"]
            ),
            "maximum_instances": int(choose(
                args.yolo_max_instances, segmentation, "maximum_instances_after_deduplication",
                "maximum_instances_after_deduplication"
            )),
            "channel_order": tuple(
                args.yolo_channel_order if args.yolo_channel_order is not None
                else segmentation.get("input_channel_order", ARTICLE_DEFAULTS["input_channel_order"])
            ),
            "class_id": args.yolo_class_id if args.yolo_class_id is not None else segmentation.get("class_id"),
            "device": parse_device(args.device),
        },
        "erosion": {
            "kernel_shape": "ellipse",
            # The inference model contains legacy erosion metadata (1 iteration,
            # 20 pixels).  Use the manuscript settings by default; only an
            # explicit CLI sensitivity-run option may override them.
            "kernel_size_pixels": int(
                args.erosion_kernel
                if args.erosion_kernel is not None else ARTICLE_DEFAULTS["erosion_kernel"]
            ),
            "iterations": int(
                args.erosion_iterations
                if args.erosion_iterations is not None else ARTICLE_DEFAULTS["erosion_iterations"]
            ),
            "minimum_retained_pixels_per_pod": int(
                args.min_eroded_pixels
                if args.min_eroded_pixels is not None
                else ARTICLE_DEFAULTS["minimum_retained_pixels_per_pod"]
            ),
        },
        "pod_feature": "median of all finite pixels in each band within the eroded pod mask",
        "pca_decision_feature": "PC1 of the three-band pod-median vector",
        "continuous_maturity": "piecewise linear interpolation through five PC1 median centers",
        "dud_mapping": "piecewise linear interpolation through six MPB-derived PC1/DUD nodes",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="One NPY file or a directory.")
    parser.add_argument("--output", type=Path, help="Default: <input>/pmes_article_output.")
    parser.add_argument("--pca-model", type=Path, default=DEFAULT_POOLED_PCA)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--device", default="auto", help="Examples: 0, cpu, auto.")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--max-files", type=int, help="Process only the first N files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Validate and print parameters only.")
    parser.add_argument("--white-reference", type=Path, help="Optional band-matched white-reference NPY.")
    parser.add_argument("--no-numeric-maps", action="store_true")
    parser.add_argument("--yolo-confidence", type=float)
    parser.add_argument("--yolo-iou", type=float)
    parser.add_argument("--yolo-imgsz", type=int)
    parser.add_argument("--yolo-max-det", type=int)
    parser.add_argument("--yolo-min-mask-area", type=int)
    parser.add_argument("--yolo-duplicate-iou", type=float)
    parser.add_argument("--yolo-duplicate-containment", type=float)
    parser.add_argument("--yolo-max-instances", type=int)
    parser.add_argument("--yolo-channel-order", nargs=3, type=int)
    parser.add_argument("--yolo-class-id", type=int)
    parser.add_argument("--erosion-kernel", type=int)
    parser.add_argument("--erosion-iterations", type=int)
    parser.add_argument("--min-eroded-pixels", type=int)
    return parser.parse_args(argv)


def validate_run(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], list[Path]]:
    input_path = args.input.resolve()
    pca_path = args.pca_model.resolve()
    yolo_path = args.yolo_model.resolve()
    if not pca_path.is_file():
        raise FileNotFoundError(f"PCA model not found: {pca_path}")
    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
    if args.white_reference is not None and not args.white_reference.resolve().is_file():
        raise FileNotFoundError(f"White reference not found: {args.white_reference}")
    output = (
        args.output.resolve() if args.output is not None
        else ((input_path.parent if input_path.is_file() else input_path) / "pmes_article_output").resolve()
    )
    model = load_pca_model(pca_path)
    parameters = resolved_parameters(args, model)
    if parameters["erosion"]["iterations"] < 0:
        raise ValueError("Erosion iterations cannot be negative.")
    ellipse_structure(parameters["erosion"]["kernel_size_pixels"])
    if parameters["segmentation"]["maximum_instances"] > 104:
        raise ValueError("The article-aligned tray cap cannot exceed 104 instances.")
    files = discover_npy(input_path, not args.no_recursive, output)
    if args.white_reference is not None:
        files = [path for path in files if path != args.white_reference.resolve()]
    if args.max_files is not None:
        if args.max_files < 1:
            raise ValueError("--max-files must be positive.")
        files = files[:args.max_files]
    if not files:
        raise FileNotFoundError("No sample NPY files remain after exclusions.")
    return input_path, output, pca_path, model, parameters, files


def main() -> None:
    args = parse_args()
    input_path, output, pca_path, model, parameters, files = validate_run(args)
    yolo_path = args.yolo_model.resolve()
    white_reference = args.white_reference.resolve() if args.white_reference else None
    preflight = {
        "pipeline_version": PIPELINE_VERSION,
        "input": str(input_path), "npy_files": len(files), "output": str(output),
        "pca_model": str(pca_path), "pca_model_sha256": sha256_file(pca_path),
        "pca_model_version": model["model_version"],
        "yolo_model": str(yolo_path), "yolo_model_sha256": sha256_file(yolo_path),
        "band_indices": model["band_indices"], "wavelengths_nm": WAVELENGTHS_NM,
        "white_reference": str(white_reference) if white_reference else None,
        "parameters": parameters,
    }
    print(json.dumps(preflight, indent=2, ensure_ascii=False, default=json_default))
    if args.preflight:
        return
    output.mkdir(parents=True, exist_ok=True)
    yolo_model = load_yolo_model(yolo_path)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    skipped: list[str] = []
    for index, npy_path in enumerate(files, 1):
        destination = result_directory(input_path, npy_path, output)
        if (destination / "results.json").exists() and not args.overwrite:
            skipped.append(str(npy_path))
            print(f"[{index}/{len(files)}] SKIP {npy_path.name}: results.json already exists")
            continue
        try:
            print(f"[{index}/{len(files)}] Processing {npy_path}")
            result = process_one_npy(
                npy_path, destination, model, yolo_model, parameters,
                white_reference, not args.no_numeric_maps,
            )
            completed.append({
                "npy_file": str(npy_path), "result_directory": str(destination),
                "pod_count": result["pod_count"],
                "q95_dud": result["decision"]["mpb_derived_dud_at_q95_maturity"],
                "brown_ratio": result["decision"]["brown_ratio"],
                "black_ratio": result["decision"]["black_ratio"],
                "brown_black_ratio": result["decision"]["brown_black_ratio"],
                "warnings": result["quality"]["warnings"],
            })
        except Exception as exc:
            failures.append({"npy_file": str(npy_path), "error": str(exc)})
            print(f"[{index}/{len(files)}] ERROR {npy_path.name}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise RuntimeError(f"Failed on {npy_path}: {exc}") from exc

    summary_csv = output / "batch_summary.csv"
    if completed:
        fields = [
            "npy_file", "result_directory", "pod_count", "q95_dud",
            "brown_ratio", "black_ratio", "brown_black_ratio", "warnings",
        ]
        with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in completed:
                row = dict(item)
                row["warnings"] = " | ".join(row["warnings"])
                writer.writerow(row)
    manifest = {
        **preflight, "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed": completed, "skipped_existing": skipped, "failures": failures,
        "batch_summary_csv": str(summary_csv) if completed else None,
        "dud_caveat": (
            "All DUD outputs are derived from MPB category/node mapping and have not yet been "
            "prospectively validated as economically optimal digging dates."
        ),
    }
    manifest_path = output / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=json_default)
    print(json.dumps({
        "completed": len(completed), "skipped": len(skipped), "failures": len(failures),
        "output": str(output), "manifest": str(manifest_path),
    }, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
