"""Measure model-serving costs in a fresh process; never writes account history.

Default weather is deliberately fixed/missing to remove network timing noise.
Run: .venv/Scripts/python.exe scripts/benchmark_serving.py --compare
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'flight_delay_model'
SERVER = CODE / 'server'
BEFORE = ROOT / '.local/bucket-only-before'
FLIGHT = {'FL_DATE': '2021-12-30', 'AIRLINE': 'Southwest Airlines Co.',
          'ORIGIN': 'DEN', 'DEST': 'SMF', 'CRS_DEP_TIME': 2245,
          'CRS_ARR_TIME': 25, 'DISTANCE': 909.0}


def memory_mb():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return round(counters.WorkingSetSize / 1048576, 2)
    return None


def worker(mode, repeats):
    start = time.perf_counter()
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(SERVER))
    if mode == 'before':
        sys.path.insert(0, str(BEFORE))
    import predictor_service as service
    import weather_serving as wx
    if mode == 'before':
        registry = json.loads((BEFORE / 'wx_serving.json').read_text(encoding='utf-8'))
        wx.load_registry = lambda: registry
    # Both versions see identical features; these are not real fetched weather values.
    wx.fetch_forecast_weather = lambda *args, **kwargs: wx._missing('o')
    wx.fetch_dest_forecast_weather = lambda *args, **kwargs: wx._missing('d')

    from catboost import CatBoostClassifier, CatBoostRegressor
    loaded, inferred = [], []
    def instrument(cls, name, collection, label):
        original = getattr(cls, name)
        def wrapped(self, *args, **kwargs):
            collection.append(label)
            return original(self, *args, **kwargs)
        setattr(cls, name, wrapped)
    for cls in (CatBoostClassifier, CatBoostRegressor):
        instrument(cls, 'load_model', loaded, cls.__name__)
    instrument(CatBoostClassifier, 'predict_proba', inferred, 'CatBoostClassifier')
    instrument(CatBoostRegressor, 'predict', inferred, 'CatBoostRegressor')
    # The legacy path lazily imports LightGBM while predicting, so instrument its loader.
    if mode == 'before':
        import lightgbm
        instrument(lightgbm.Booster, '__init__', loaded, 'LightGBM')
        instrument(lightgbm.Booster, 'predict', inferred, 'LightGBM')
    db = service.MetaDB(SERVER / 'flight_meta.db')
    cold = service.predict(FLIGHT, db)
    if not isinstance(cold, dict):
        raise RuntimeError(cold)
    cold_ms = (time.perf_counter() - start) * 1000
    cold_loads, cold_calls = len(loaded), len(inferred)
    rss_warm = memory_mb()
    durations, cpu_times, calls, loads = [], [], [], []
    for _ in range(repeats):
        loaded.clear()
        inferred.clear()
        t, cpu = time.perf_counter(), time.process_time()
        outcome = service.predict(FLIGHT, db)
        durations.append((time.perf_counter() - t) * 1000)
        cpu_times.append((time.process_time() - cpu) * 1000)
        calls.append(len(inferred))
        loads.append(len(loaded))
    answer = {'mode': mode, 'weather': 'fixed missing features; no network calls', 'repeats': repeats,
              'cold_import_and_first_prediction_ms': round(cold_ms, 2),
              'warm_median_ms': round(statistics.median(durations), 2),
              'warm_p95_ms': round(sorted(durations)[max(0, int(len(durations) * .95) - 1)], 2),
              'warm_median_cpu_ms': round(statistics.median(cpu_times), 2),
              'working_set_after_first_prediction_mb': rss_warm, 'working_set_final_mb': memory_mb(),
              'cold_model_loads': cold_loads, 'cold_model_inferences': cold_calls,
              'warm_model_inferences_per_request': sorted(set(calls)),
              'warm_model_loads_per_request': sorted(set(loads)),
              'weather_components': cold.get('weather_components'),
              'frontend_scenarios': cold['scenarios']['dep_delay'],
              'frontend_verdict': cold['verdicts']['dep_delay']}
    db.conn.close()
    print(json.dumps(answer))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--worker', choices=['before', 'after'])
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.repeats)
        return
    modes = ['before', 'after'] if args.compare else ['after']
    results = {}
    for mode in modes:
        if mode == 'before' and not (BEFORE / 'predictor_service.py').is_file():
            raise SystemExit('Original local snapshot is unavailable. Run without --compare.')
        run = subprocess.run([sys.executable, __file__, '--worker', mode, '--repeats', str(args.repeats)],
                             cwd=ROOT, text=True, capture_output=True, check=True)
        results[mode] = json.loads(run.stdout.strip().splitlines()[-1])
    report = {'scope': 'Isolated local serving cost; excludes HTTP, DB writes and external weather latency.',
              'results': results}
    if len(results) == 2:
        before, after = results['before'], results['after']
        report['frontend_outputs_identical'] = (before['frontend_scenarios'] == after['frontend_scenarios']
                                               and before['frontend_verdict'] == after['frontend_verdict'])
        assert report['frontend_outputs_identical'], 'Frontend-visible outputs changed!'
        report['warm_latency_reduction_percent'] = round(100 * (1 - after['warm_median_ms'] / before['warm_median_ms']), 1)
    path = ROOT / '.local/serving-benchmark.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Report:', path)


if __name__ == '__main__':
    main()
