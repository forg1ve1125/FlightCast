"""Real-model serving regression tests; isolated weather fixtures and account DB.

Run from project root: .venv/Scripts/python.exe flight_delay_model/server/test_bucket_only.py
No training, model modification, external weather requests or production history writes.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SERVER = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER))
import predictor_service as service
import weather_serving as wx
import auth
from test_auth_history_api import FLIGHT, TEST_ROOT, load_test_app


class BucketOnlyServing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = service.MetaDB()

    @classmethod
    def tearDownClass(cls):
        cls.db.conn.close()

    def setUp(self):
        self.o = patch.object(wx, 'fetch_forecast_weather', side_effect=lambda *a, **k: wx._missing('o'))
        self.d = patch.object(wx, 'fetch_dest_forecast_weather', side_effect=lambda *a, **k: wx._missing('d'))
        self.o.start()
        self.d.start()
        self.addCleanup(self.o.stop)
        self.addCleanup(self.d.stop)

    def test_only_one_model_loaded_and_inferred_per_request(self):
        from catboost import CatBoostClassifier, CatBoostRegressor
        wx._model_cache.clear()
        original = wx._predict
        with patch.object(wx, '_load_model', wraps=wx._load_model) as loader, \
             patch.object(wx, '_predict', wraps=original) as infer, \
             patch.object(CatBoostRegressor, 'load_model', side_effect=AssertionError('regressor loaded')), \
             patch.object(CatBoostClassifier, 'load_model', autospec=True,
                          side_effect=CatBoostClassifier.load_model) as binary_loader:
            for _ in range(2):
                result = service.predict(FLIGHT, self.db)
                self.assertIsInstance(result, dict)
                self.assertEqual(result['weather_components'], ['bucket+iso'])
            self.assertEqual(infer.call_count, 2)
            self.assertEqual(binary_loader.call_count, 1)  # second request uses cached model
            self.assertTrue(all(c.args[0] == '20261013_122152_buckets_wx_dep' for c in loader.call_args_list))
        self.assertNotIn('04_predict', sys.modules)
        self.assertNotIn('lightgbm', sys.modules)

    def test_visible_probabilities_match_original_snapshot(self):
        result = service.predict(FLIGHT, self.db)
        items = sorted(result['scenarios']['dep_delay']['scenarios'], key=lambda s: s['bucket'])
        self.assertEqual([s['probability'] for s in items], [.0287, .1944, .1903, .2331, .3535])
        self.assertEqual(result['verdicts']['dep_delay']['bucket'], 4)
        self.assertEqual(result['scenarios']['dep_delay']['expected_delay_min_from_buckets'], 51.7)
        derived = result['predictions']['dep_delay']
        self.assertEqual(derived['prob_delay'], .9713)
        self.assertEqual(derived['expected_delay_source'], 'bucket-weighted')

    def test_failure_never_reactivates_other_models(self):
        registry = wx.load_registry()
        for changed in ({}, {**registry, 'bucket_run': 'not-a-model'},
                        {**registry, 'cls_run': registry['disabled_runs']['cls_run']}):
            with self.subTest(registry=changed), patch.object(wx, 'load_registry', return_value=changed), \
                 self.assertLogs('predictor_service', level='ERROR'):
                outcome = service.predict(FLIGHT, self.db)
                self.assertEqual(outcome[1], 503)
                self.assertNotIn('04_predict', sys.modules)

    def test_missing_calibration_never_serves_uncalibrated_predictions(self):
        with patch.object(wx, '_load_isotonic', return_value=False), \
             patch.object(wx, '_predict') as infer, self.assertLogs('predictor_service', level='ERROR'):
            self.assertEqual(service.predict(FLIGHT, self.db)[1], 503)
            infer.assert_not_called()

    def test_inference_error_returns_503_without_a_baseline_fallback(self):
        with patch.object(wx, '_predict', side_effect=RuntimeError('injected inference failure')) as infer, \
             self.assertLogs('predictor_service', level='ERROR'):
            self.assertEqual(service.predict(FLIGHT, self.db)[1], 503)
            self.assertEqual(infer.call_count, 1)
            self.assertNotIn('04_predict', sys.modules)

    def test_prediction_failure_is_not_saved_and_timing_is_visible(self):
        with tempfile.TemporaryDirectory(prefix='bucket-', dir=TEST_ROOT) as tmp:
            conn = auth.init_db(Path(tmp) / 'app.db')
            module = load_test_app(conn, '_bucket_http_test')
            try:
                client = module.app.test_client()
                self.assertEqual(client.post('/api/auth/register', json={
                    'username': 'isolated_user', 'password': 'test-password'}).status_code, 201)
                with patch.object(wx, '_load_isotonic', return_value=False), \
                     self.assertLogs('predictor_service', level='ERROR'):
                    response = client.post('/api/predict', json=FLIGHT)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM query_history').fetchone()[0], 0)
                response = client.post('/api/predict', json=FLIGHT)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.headers['Server-Timing'].startswith('predict;dur='))
                self.assertEqual(len(client.get('/api/history').get_json()['items']), 1)
            finally:
                module.db.conn.close()
                conn.close()
                sys.modules.pop('_bucket_http_test', None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
