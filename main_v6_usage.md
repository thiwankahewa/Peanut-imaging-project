# v6 model selection and DUD

Run from the project directory using the existing camera environment:

```bash
.venv/bin/python main_v6.py
```

On the **Capture** tab, choose a file under **Maturity model (.pkl)** before
starting capture. The dropdown lists `.pkl` files in the project's `models/`
folder. Click **Refresh** after adding a model. The selection stays fixed during
capture and analysis; selecting a different model applies to the next capture.

- `pooled_reference_model.pkl` is selected by default. It runs the September 23
  pipeline: pod segmentation, duplicate suppression, mask erosion, three-band
  pod medians, fixed PCA, maturity classes, and MPB-derived DUD.
- `pca12_regression_model.pkl` retains the previous v6 regression analysis. Its
  pickle contains no DUD calibration, so DUD and class proportions display as
  unavailable. The previous hard-coded 17-day estimate has been removed.

The Capture results show the pod count, mean/std maturity index, DUD, and Brown,
Black, and combined percentages. DUD is interpolated from the **95th percentile
of increasing pod maturity (PC1)**, rather than the 95th percentile of DUD.
It retains the source pipeline's interpretation as an MPB-derived indicator.

The **Analysis** tab previews maturity maps, the DUD map, the maturity profile,
annotated pods, and masks. Each pooled-model run saves these images plus
`per_pod_predictions.csv`, `results.json`, `summary.txt`, and `numeric_maps.npz`
under `analysis_outputs/<capture name>__<model name>/`. JSON and text summaries
identify the selected model. Existing v6 heatmaps remain browsable.

Keep `maturity_analysis.py` and `Shaoqi_codes/09_23_26_main.py` with `main_v6.py`.
The new path uses the captured calibration-ratio cube without per-pod rescaling
or a second white-reference division. Calibrate the imaging box before use.

Run the hardware-free tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
