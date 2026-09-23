"""Hardware-free checks for model selection, DUD math, and GUI result handling."""
import csv
import importlib
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from maturity_analysis import MaturityAnalyzer, discover_model_files, pmes


class IdentityPCA:
    def transform(self, values):
        return np.asarray(values)


class ConstantRegression:
    def predict(self, values):
        return np.full(len(values), 0.5)


class ArrayTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Boxes:
    def __init__(self, boxes, scores):
        self.xyxy = ArrayTensor(boxes)
        self.conf = ArrayTensor(scores)
        self.cls = ArrayTensor(np.zeros(len(boxes)))
        self.count = len(boxes)

    def __len__(self):
        return self.count


def calibration():
    return {
        "pca": IdentityPCA(), "centers": np.arange(5.),
        "thresholds": np.arange(0.5, 4., 1.),
        "pc1_nodes": np.arange(-0.5, 5., 1.),
        "dud_nodes": np.array([87., 73., 45., 31., 21., 0.]),
    }


def fake_tray():
    cube = np.zeros((38, 158, 3), dtype=np.float32)
    masks, boxes = [], []
    for index in range(5):
        left = 4 + index * 30
        mask = np.zeros(cube.shape[:2], dtype=bool)
        mask[6:32, left:left + 26] = True
        cube[mask] = [index, 0.4, 0.6]
        masks.append(mask)
        boxes.append([left, 6, left + 26, 32])
    # Duplicate must be suppressed, not counted as a sixth pod.
    masks.append(masks[0].copy())
    boxes.append(boxes[0])
    yolo = Mock()
    yolo.predict.return_value = [SimpleNamespace(
        masks=SimpleNamespace(data=ArrayTensor(masks)),
        boxes=Boxes(boxes, [0.9] * 5 + [0.7]),
    )]
    return cube, yolo


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / 'test.pt'
        self.weights.touch()
        self.runner = MaturityAnalyzer(self.weights)
        self.pooled = self.write_model('pooled.pkl', calibration())
        self.legacy = self.write_model('legacy.pkl', {'pca': IdentityPCA(), 'reg': ConstantRegression()})

    def write_model(self, name, model):
        path = self.root / name
        path.write_bytes(pickle.dumps(model))
        return path

    def test_discovery_only_lists_model_files(self):
        (self.root / 'folder.pkl').mkdir()
        (self.root / 'notes.txt').touch()
        upper = self.write_model('MODEL.PKL', calibration())
        self.assertEqual(discover_model_files(self.root), [self.legacy, upper, self.pooled])
        self.assertEqual(discover_model_files(self.root / 'absent'), [])

    def test_switching_and_replacing_model_invalidates_cache(self):
        first = self.runner.load_model(self.pooled)
        self.assertIs(self.runner.load_model(self.pooled), first)
        self.assertEqual(self.runner.load_model(self.legacy)['kind'], 'legacy_regression')
        second = self.runner.load_model(self.pooled)
        self.assertIsNot(first, second)
        updated = calibration()
        updated['model_version'] = 'updated calibration'
        self.write_model(self.pooled.name, updated)
        self.assertEqual(self.runner.load_model(self.pooled)['calibration']['model_version'], 'updated calibration')

    def test_invalid_selection_does_not_reuse_previous_model(self):
        old = self.runner.load_model(self.pooled)
        broken = calibration()
        broken['reg'] = ConstantRegression()
        broken['dud_nodes'][0] = np.inf
        invalid = self.write_model('invalid.pkl', broken)
        with self.assertRaisesRegex(ValueError, 'DUD calibration'):
            self.runner.load_model(invalid)
        with self.assertRaises(FileNotFoundError):
            self.runner.load_model(self.root / 'missing.pkl')
        self.assertIs(self.runner.load_model(self.pooled), old)

    def test_pooled_pipeline_preserves_spectra_and_exports_q95_dud(self):
        cube, yolo = fake_tray()
        sample = self.root / 'tray.npy'
        np.save(sample, cube)
        with patch.object(pmes, 'load_yolo_model', return_value=yolo):
            result = self.runner.analyze(sample, self.pooled, self.root / 'outputs')
        # Increasing maturity PC1 [0,1,2,3,4] has Q95=3.8, giving 14.7 DUD.
        self.assertEqual(result['n_peanuts'], 5)
        self.assertAlmostEqual(result['days_left'], 14.7)
        self.assertAlmostEqual(result['mean_maturity'], 0.5)
        self.assertAlmostEqual(result['brown_black_ratio'], 0.4)
        self.assertEqual(result['detection_audit']['suppressed_duplicate_count'], 1)
        self.assertEqual([row['band_405_median'] for row in result['pod_results']], [0., 1., 2., 3., 4.])
        self.assertEqual(result['decision']['class_counts'], dict.fromkeys(pmes.CATEGORIES, 1))
        self.assertEqual(result['parameters']['erosion']['iterations'], 5)
        self.assertEqual(yolo.predict.call_args.kwargs['max_det'], 300)
        for path in result['outputs'].values():
            self.assertTrue(Path(path).is_file(), path)
        saved = json.loads(Path(result['outputs']['result_json']).read_text())
        self.assertAlmostEqual(saved['days_left'], 14.7)
        self.assertEqual(saved['selected_model']['path'], str(self.pooled))
        with open(result['outputs']['per_pod_csv'], newline='') as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 5)
        with np.load(result['outputs']['numeric_maps']) as maps:
            self.assertEqual(maps['dud_days'].shape, cube.shape[:2])
        self.assertIn('14.7 days', Path(result['outputs']['summary_text']).read_text())

    def test_legacy_dispatch_never_invents_dud(self):
        sample = self.root / 'tray.npy'
        np.save(sample, np.zeros((10, 10, 3)))
        legacy = Mock(return_value={
            'n_peanuts': 1, 'mean_maturity': 0.5, 'std_maturity': 0., 'days_left': 17,
            'heatmap_path': 'heatmap.png', 'annotated_path': 'annotated.png', 'binary_mask_path': 'mask.png',
        })
        with patch.object(pmes, 'load_yolo_model', return_value=Mock()):
            result = self.runner.analyze(sample, self.legacy, self.root / 'outputs', legacy_processor=legacy)
        legacy.assert_called_once()
        self.assertIsNone(result['days_left'])
        self.assertIsNone(result['brown_black_ratio'])
        self.assertIn('DUD unavailable', result['warnings'][0])
        saved = json.loads(Path(result['outputs']['result_json']).read_text())
        self.assertIsNone(saved['days_left'])

    def test_empty_detection_produces_clear_error(self):
        sample = self.root / 'empty.npy'
        np.save(sample, np.zeros((38, 38, 3)))
        yolo = Mock()
        yolo.predict.return_value = [SimpleNamespace(masks=None, boxes=None)]
        with patch.object(pmes, 'load_yolo_model', return_value=yolo):
            with self.assertRaisesRegex(ValueError, 'No usable pod'):
                self.runner.analyze(sample, self.pooled, self.root / 'outputs')


class ResultDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real GUI methods while preventing any hardware access.
        with patch.dict('sys.modules', {
            'PySpin': Mock(), 'gpiozero': SimpleNamespace(OutputDevice=Mock()),
        }):
            cls.gui = importlib.import_module('main_v6')

    def make_app(self):
        app = self.gui.PeanutApp.__new__(self.gui.PeanutApp)
        for name in ['result_model_var', 'total_var', 'maturity_var', 'dud_var',
                     'class_var', 'brown_black_var', 'result_note_var', 'selected_model_var']:
            setattr(app, name, Mock())
        app.after = lambda delay, callback: callback()
        app.refresh_analysis_list = Mock()
        return app

    def test_zero_dud_is_displayed_and_failure_clears_results(self):
        app = self.make_app()
        app.safe_update_results({
            'model_name': 'pooled.pkl', 'n_peanuts': 5, 'mean_maturity': 1., 'std_maturity': 0.,
            'days_left': 0., 'brown_ratio': 0., 'black_ratio': 1., 'brown_black_ratio': 1.,
            'warnings': [], 'heatmap_path': 'heatmap.png',
        })
        app.dud_var.set.assert_called_with('DUD (days until digging): 0.0')
        app.brown_black_var.set.assert_called_with('Brown + Black: 100.0%')
        app.safe_analysis_error('No usable pod', 'pooled.pkl')
        app.dud_var.set.assert_called_with('DUD (days until digging): -')
        self.assertIsNone(app.latest_heatmap_path)
        app.result_note_var.set.assert_called_with('Analysis unavailable: No usable pod')

    def test_output_browser_finds_new_dud_maps_and_legacy_heatmaps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'capture__pooled'
            run.mkdir()
            dud_map = run / 'continuous_dud_map.png'
            for path in [dud_map, root / 'old_maturity_heatmap.png']:
                pmes.Image.new('RGB', (30, 30)).save(path)
            (run / 'results.json').write_text(json.dumps({
                'model_name': 'pooled.pkl', 'days_left': 0.,
            }))
            app = self.make_app()
            # Exercise the real browser method, including image/summary selection.
            del app.refresh_analysis_list
            app.analysis_listbox = Mock()
            app.analysis_preview_label = Mock()
            app.analysis_preview_label.winfo_width.return_value = 400
            app.analysis_preview_label.winfo_height.return_value = 300
            app.analysis_details_var = Mock()
            selected = [0]
            app.analysis_listbox.select_set.side_effect = lambda index: selected.__setitem__(0, index)
            app.analysis_listbox.curselection.side_effect = lambda: (selected[0],)
            with patch.object(self.gui, 'ANALYSIS_OUTPUT_DIR', folder), patch.object(self.gui.ImageTk, 'PhotoImage'):
                app.refresh_analysis_list(preferred_path=str(dud_map))
            self.assertEqual(set(app.analysis_files), {dud_map, root / 'old_maturity_heatmap.png'})
            self.assertEqual(app.analysis_files[selected[0]], dud_map)
            app.analysis_details_var.set.assert_called_with('Model: pooled.pkl\nMPB-derived Q95 DUD: 0.0 days')
            labels = [call.args[1] for call in app.analysis_listbox.insert.call_args_list]
            self.assertTrue(any(label.startswith('DUD map |') for label in labels))

    def test_refresh_prefers_pooled_and_preserves_user_selection(self):
        app = self.make_app()
        app.is_capturing = False
        app.model_combo = Mock()
        app.set_status = Mock()
        app.selected_model_var.get.return_value = ''
        with patch.object(self.gui, 'discover_model_files', return_value=[
            Path('pca12_regression_model.pkl'), Path('pooled_reference_model.pkl'),
        ]):
            app.refresh_model_list()
            app.selected_model_var.set.assert_called_with('pooled_reference_model.pkl')
            app.selected_model_var.set.reset_mock()
            app.selected_model_var.get.return_value = 'pca12_regression_model.pkl'
            app.refresh_model_list()
            app.selected_model_var.set.assert_not_called()
            app.is_capturing = True
            app.model_combo.configure.reset_mock()
            app.refresh_model_list()
            app.model_combo.configure.assert_not_called()


if __name__ == '__main__':
    unittest.main()
