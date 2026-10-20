r"""Authentication/history regression tests through real Flask HTTP handlers.

Run from the project root:
    .venv\Scripts\python.exe flight_delay_model\server\test_auth_history_api.py

Every test uses a temporary account database. Successful predictions call the
unchanged real prediction service; no synthetic probabilities are persisted.
"""
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER = Path(__file__).resolve().parent
ROOT = SERVER.parents[1]
TEST_ROOT = ROOT / '.local/auth-test-databases'
TEST_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SERVER))
import auth

FLIGHT = {
    'FL_DATE': '2021-12-30', 'AIRLINE': 'Southwest Airlines Co.',
    'ORIGIN': 'DEN', 'DEST': 'SMF', 'CRS_DEP_TIME': 2245,
    'CRS_ARR_TIME': 25, 'DISTANCE': 909.0,
}
TEST_PASSWORD = 'test-only-password-123'


def load_test_app(connection, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SERVER / 'app.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Redirect initialization BEFORE importing app.py, not after it opens app.db.
    with patch.object(auth, 'init_db', return_value=connection):
        spec.loader.exec_module(module)
    module.app.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False,
                             SECRET_KEY='isolated-regression-test-key')
    module.app.logger.disabled = True
    return module


class AuthHistoryAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bootstrap_dir = tempfile.TemporaryDirectory(prefix='bootstrap-', dir=TEST_ROOT)
        assert Path(cls.bootstrap_dir.name).resolve().is_relative_to(TEST_ROOT.resolve())
        cls.bootstrap = auth.init_db(Path(cls.bootstrap_dir.name) / 'app.db')
        cls.module = load_test_app(cls.bootstrap, '_is6400_auth_api_tests')

    @classmethod
    def tearDownClass(cls):
        cls.module.db.conn.close()
        cls.bootstrap.close()
        cls.bootstrap_dir.cleanup()
        sys.modules.pop('_is6400_auth_api_tests', None)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='case-', dir=TEST_ROOT)
        assert Path(self.temp.name).resolve().is_relative_to(TEST_ROOT.resolve())
        self.db_path = Path(self.temp.name) / 'app.db'
        self.conn = auth.init_db(self.db_path)
        self.module.app_db = self.conn
        self.app = self.module.app
        self.a, self.b, self.guest = [self.app.test_client() for _ in range(3)]

    def tearDown(self):
        self.conn.close()
        self.temp.cleanup()

    def register(self, client, name='user_a'):
        response = client.post('/api/auth/register', json={'username': name, 'password': TEST_PASSWORD})
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()['user']

    def login(self, client, name='user_a', password=TEST_PASSWORD):
        return client.post('/api/auth/login', json={'username': name, 'password': password})

    def count(self):
        return self.conn.execute('SELECT COUNT(*) FROM query_history').fetchone()[0]

    def predict(self, client, **overrides):
        response = client.post('/api/predict', json={**FLIGHT, **overrides})
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(set(response.get_json()['weather_components']), {'cls', 'single', 'bucket+iso', 'cond'})
        return response.get_json()

    def items(self, client):
        response = client.get('/api/history')
        self.assertEqual(response.status_code, 200)
        return response.get_json()['items']

    def test_01_register_and_login_establish_identity(self):
        """Registration signs in; logout then case-insensitive login works."""
        user = self.register(self.a, 'Alice')
        self.assertEqual(self.a.get('/api/auth/me').get_json()['user'], user)
        self.a.post('/api/auth/logout')
        self.assertEqual(self.login(self.a, 'aLiCe').status_code, 200)
        self.assertEqual(self.a.get('/api/auth/me').get_json()['user']['id'], user['id'])

    def test_02_duplicate_names_are_case_insensitive(self):
        """Exact, case-changed and whitespace-padded duplicate names return 409."""
        self.register(self.a, 'Alice')
        for name in ('Alice', 'ALICE', ' Alice '):
            with self.subTest(name=name):
                response = self.b.post('/api/auth/register', json={'username': name, 'password': TEST_PASSWORD})
                self.assertEqual(response.status_code, 409)
                self.assertIsNone(self.b.get('/api/auth/me').get_json()['user'])
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM users').fetchone()[0], 1)
        self.assertEqual(self.login(self.b, 'alice').status_code, 200)

    def test_03_failed_registration_releases_transaction(self):
        """A duplicate registration leaves no failed open write transaction."""
        self.register(self.a)
        self.assertEqual(self.b.post('/api/auth/register', json={
            'username': 'USER_A', 'password': TEST_PASSWORD}).status_code, 409)
        self.assertFalse(self.conn.in_transaction)
        other = sqlite3.connect(self.db_path, timeout=0.1)
        try:
            other.execute('BEGIN IMMEDIATE')
            other.rollback()
        finally:
            other.close()

    def test_04_registration_field_boundaries(self):
        """Invalid names/password lengths are rejected without creating users."""
        for name, password in (('', TEST_PASSWORD), ('ab', TEST_PASSWORD), ('a' * 33, TEST_PASSWORD),
                               ('bad name', TEST_PASSWORD), ('name!', TEST_PASSWORD), ('valid', '12345')):
            with self.subTest(name=name):
                response = self.guest.post('/api/auth/register', json={'username': name, 'password': password})
                self.assertEqual(response.status_code, 400)
        for name in ('a_1', 'a' * 32):
            self.assertEqual(self.guest.post('/api/auth/register', json={
                'username': name, 'password': '123456'}).status_code, 201)

    def test_05_wrong_credentials_do_not_sign_in(self):
        """Wrong password and unknown/injection-like username never sign in."""
        self.register(self.a)
        for name, password in (('user_a', 'wrong-password'), ('unknown', TEST_PASSWORD),
                               ("' OR 1=1 --", TEST_PASSWORD)):
            with self.subTest(name=name):
                response = self.login(self.guest, name, password)
                self.assertEqual(response.status_code, 401)
                self.assertIsNone(self.guest.get('/api/auth/me').get_json()['user'])

    def test_06_passwords_are_hashed_and_not_returned(self):
        """Equal passwords have distinct salts/hashes; API exposes no secrets."""
        self.register(self.a)
        self.register(self.b, 'user_b')
        rows = self.conn.execute('SELECT password_hash, salt FROM users').fetchall()
        self.assertNotEqual(rows[0]['salt'], rows[1]['salt'])
        self.assertNotEqual(rows[0]['password_hash'], rows[1]['password_hash'])
        self.assertTrue(all(row['password_hash'] != TEST_PASSWORD for row in rows))
        me = self.a.get('/api/auth/me').get_json()['user']
        self.assertEqual(set(me), {'id', 'username'})

    def test_07_history_lists_are_scoped_to_user(self):
        """A and B each see only their own saved prediction IDs."""
        self.register(self.a)
        self.predict(self.a)
        a_id = self.items(self.a)[0]['id']
        self.register(self.b, 'user_b')
        self.assertEqual(self.items(self.b), [])
        self.predict(self.b, FL_DATE='2022-10-30')
        b_id = self.items(self.b)[0]['id']
        self.assertNotEqual(a_id, b_id)
        self.assertEqual([item['id'] for item in self.items(self.a)], [a_id])
        self.assertEqual([item['id'] for item in self.items(self.b)], [b_id])

    def test_08_cross_user_detail_is_not_found(self):
        """Knowing A's record ID still does not let B retrieve it."""
        self.register(self.a)
        self.predict(self.a)
        history_id = self.items(self.a)[0]['id']
        self.register(self.b, 'user_b')
        self.assertEqual(self.b.get(f'/api/history/{history_id}').status_code, 404)
        self.assertEqual(self.a.get(f'/api/history/{history_id}').status_code, 200)

    def test_09_claimed_user_id_cannot_change_owner(self):
        """Payload/query user_id values cannot impersonate a different account."""
        a = self.register(self.a)
        b = self.register(self.b, 'user_b')
        self.predict(self.b, user_id=a['id'])
        owner = self.conn.execute('SELECT user_id FROM query_history').fetchone()[0]
        self.assertEqual(owner, b['id'])
        self.assertEqual(self.a.get(f"/api/history?user_id={b['id']}").get_json()['items'], [])

    def test_10_guest_cannot_read_list_or_detail(self):
        """Guest requests receive 401 for both list and a known record ID."""
        self.register(self.a)
        self.predict(self.a)
        history_id = self.items(self.a)[0]['id']
        self.assertEqual(self.guest.get('/api/history').status_code, 401)
        self.assertEqual(self.guest.get(f'/api/history/{history_id}').status_code, 401)

    def test_11_guest_prediction_is_not_saved(self):
        """Guest real-model prediction succeeds without creating history/users."""
        self.predict(self.guest)
        self.assertEqual(self.count(), 0)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM users').fetchone()[0], 0)

    def test_12_failed_prediction_is_not_saved(self):
        """Invalid prediction requests never create a history row."""
        self.register(self.a)
        for payload in ({}, {**FLIGHT, 'ORIGIN': 'ZZZ'}, {**FLIGHT, 'CRS_DEP_TIME': 2499}):
            with self.subTest(payload=payload):
                self.assertEqual(self.a.post('/api/predict', json=payload).status_code, 400)
        self.assertEqual(self.count(), 0)

    def test_13_saved_result_round_trip_is_exact(self):
        """History returns the submitted input and exact computed output."""
        self.register(self.a)
        predicted = self.predict(self.a)
        item = self.a.get(f"/api/history/{self.items(self.a)[0]['id']}").get_json()
        self.assertEqual(item['payload'], FLIGHT)
        self.assertEqual(item['result'], predicted)

    def test_14_history_survives_logout_login_and_db_reopen(self):
        """History remains after logout, reconnection and a fresh login."""
        self.register(self.a)
        self.predict(self.a)
        before = self.items(self.a)
        self.a.post('/api/auth/logout')
        self.conn.close()
        self.conn = auth.get_conn(self.db_path)
        self.module.app_db = self.conn
        self.assertEqual(self.login(self.a).status_code, 200)
        self.assertEqual(self.items(self.a), before)

    def test_15_history_and_session_survive_backend_reload(self):
        """A second backend instance reads durable history and valid session."""
        self.register(self.a)
        self.predict(self.a)
        before = self.items(self.a)
        cookie = self.a.get_cookie(self.app.config['SESSION_COOKIE_NAME']).value
        other_conn = auth.get_conn(self.db_path)
        restarted = load_test_app(other_conn, '_is6400_auth_reload_test')
        try:
            client = restarted.app.test_client()
            client.set_cookie(self.app.config['SESSION_COOKIE_NAME'], cookie)
            self.assertEqual(client.get('/api/history').get_json()['items'], before)
        finally:
            restarted.db.conn.close()
            other_conn.close()
            sys.modules.pop('_is6400_auth_reload_test', None)

    def test_16_logout_blocks_history_and_guest_queries_are_not_saved(self):
        """Normal logout blocks history and stops subsequent query persistence."""
        self.register(self.a)
        self.predict(self.a)
        self.assertEqual(self.a.post('/api/auth/logout').status_code, 200)
        self.assertIsNone(self.a.get('/api/auth/me').get_json()['user'])
        self.assertEqual(self.a.get('/api/history').status_code, 401)
        self.predict(self.a)
        self.assertEqual(self.count(), 1)

    def test_17_logout_revokes_copied_old_session(self):
        """Replaying the exact pre-logout cookie cannot restore authentication."""
        self.register(self.a)
        self.predict(self.a)
        name = self.app.config['SESSION_COOKIE_NAME']
        copied = self.a.get_cookie(name).value
        self.a.post('/api/auth/logout')
        self.guest.set_cookie(name, copied)
        self.assertEqual(self.guest.get('/api/history').status_code, 401)
        self.assertIsNone(self.guest.get('/api/auth/me').get_json()['user'])
        self.predict(self.guest)
        self.assertEqual(self.count(), 1)

    def test_18_logout_does_not_log_out_other_user(self):
        """Logging out A leaves B's independent session and history intact."""
        self.register(self.a)
        b = self.register(self.b, 'user_b')
        self.predict(self.b)
        self.a.post('/api/auth/logout')
        self.assertEqual(self.b.get('/api/auth/me').get_json()['user']['id'], b['id'])
        self.assertEqual(len(self.items(self.b)), 1)

    def test_19_tampered_session_cookie_is_rejected(self):
        """An edited cookie cannot authenticate as any account."""
        self.register(self.a)
        name = self.app.config['SESSION_COOKIE_NAME']
        cookie = self.a.get_cookie(name).value
        self.guest.set_cookie(name, ('x' if cookie[0] != 'x' else 'y') + cookie[1:])
        self.assertEqual(self.guest.get('/api/history').status_code, 401)

    def test_20_account_switch_changes_identity_and_history(self):
        """Logging into B in A's browser returns only B's records."""
        self.register(self.a)
        self.predict(self.a)
        b = self.register(self.b, 'user_b')
        self.assertEqual(self.login(self.a, 'user_b').status_code, 200)
        self.assertEqual(self.a.get('/api/auth/me').get_json()['user']['id'], b['id'])
        self.assertEqual(self.items(self.a), [])

    def test_21_auth_handles_non_object_json(self):
        """Malformed JSON shapes are rejected with 400 instead of server errors."""
        for endpoint in ('register', 'login'):
            for value in ([1], 'abc', 42, True):
                with self.subTest(endpoint=endpoint, value=value):
                    response = self.guest.post('/api/auth/' + endpoint, json=value)
                    self.assertEqual(response.status_code, 400)

    def test_22_auth_handles_non_string_fields(self):
        """Unexpected field types cannot crash registration or login."""
        self.register(self.a)
        for endpoint in ('register', 'login'):
            for fields in ({'username': ['user_a'], 'password': TEST_PASSWORD},
                           {'username': 'user_a', 'password': ['abcdef']},
                           {'username': 123, 'password': TEST_PASSWORD},
                           {'username': 'user_a', 'password': None}):
                with self.subTest(endpoint=endpoint, fields=fields):
                    response = self.guest.post('/api/auth/' + endpoint, json=fields)
                    self.assertEqual(response.status_code, 400)

    def test_23_private_responses_are_not_cacheable(self):
        """Browsers/proxies must not cache private identity/history responses."""
        self.register(self.a)
        self.predict(self.a)
        history_id = self.items(self.a)[0]['id']
        for client, endpoint in ((self.a, '/api/auth/me'), (self.a, '/api/history'),
                                 (self.a, f'/api/history/{history_id}'), (self.guest, '/api/history')):
            with self.subTest(endpoint=endpoint):
                self.assertIn('no-store', client.get(endpoint).headers.get('Cache-Control', ''))

    def test_24_unknown_history_id_returns_404(self):
        """Missing/negative record IDs never expose another resource."""
        self.register(self.a)
        self.assertEqual(self.a.get('/api/history/999999').status_code, 404)
        self.assertEqual(self.a.get('/api/history/-1').status_code, 404)

    def test_25_expired_session_is_rejected(self):
        """A server-expired session cannot access history despite a valid cookie."""
        self.register(self.a)
        self.conn.execute('UPDATE login_sessions SET expires_at=0')
        self.conn.commit()
        self.assertEqual(self.a.get('/api/history').status_code, 401)
        self.assertIsNone(self.a.get('/api/auth/me').get_json()['user'])

    def test_26_switching_accounts_revokes_old_browser_session(self):
        """After switching accounts, the old browser cookie cannot restore A."""
        self.register(self.a)
        self.register(self.b, 'user_b')
        name = self.app.config['SESSION_COOKIE_NAME']
        copied = self.a.get_cookie(name).value
        self.assertEqual(self.login(self.a, 'user_b').status_code, 200)
        self.guest.set_cookie(name, copied)
        self.assertEqual(self.guest.get('/api/history').status_code, 401)

    def test_27_non_object_flight_payload_is_not_saved(self):
        """Unexpected prediction JSON shapes return 400 and create no history."""
        self.register(self.a)
        for value in ([1], 'abc', 42, True):
            with self.subTest(value=value):
                self.assertEqual(self.a.post('/api/predict', json=value).status_code, 400)
        self.assertEqual(self.count(), 0)

    def test_28_logout_only_revokes_current_device_session(self):
        """The same user on another device stays signed in after A logs out."""
        a = self.register(self.a)
        self.assertEqual(self.login(self.b).status_code, 200)
        self.a.post('/api/auth/logout')
        self.assertEqual(self.b.get('/api/auth/me').get_json()['user']['id'], a['id'])

    def test_29_legacy_user_id_cookie_requires_fresh_login(self):
        """The old unrevocable user_id-only cookie format no longer authenticates."""
        a = self.register(self.a)
        with self.guest.session_transaction() as session:
            session['user_id'] = a['id']
        self.assertEqual(self.guest.get('/api/history').status_code, 401)


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({'test': test.id(), 'description': test.shortDescription(), 'status': 'passed'})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({'test': test.id(), 'description': test.shortDescription(), 'status': 'failed',
                             'detail': self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({'test': test.id(), 'description': test.shortDescription(), 'status': 'error',
                             'detail': self._exc_info_to_string(err, test)})

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err:
            self.records.append({'test': subtest.id(), 'description': test.shortDescription(), 'status': 'failed',
                                 'detail': self._exc_info_to_string(err, test)})


if __name__ == '__main__':
    # Guard the real user database byte-for-byte while tests execute.
    production_before = auth.APP_DB_PATH.read_bytes() if auth.APP_DB_PATH.exists() else None
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuthHistoryAPI)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    production_after = auth.APP_DB_PATH.read_bytes() if auth.APP_DB_PATH.exists() else None
    unchanged = production_before == production_after
    destination = ROOT / '.local/auth-history-api-test-results.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps({'tests_run': result.testsRun, 'successful': result.wasSuccessful(),
                                      'real_account_database_unchanged': unchanged,
                                      'uses_real_prediction_service': True, 'records': result.records},
                                     ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nReal account database unchanged: {unchanged}\nReport: {destination}')
    sys.exit(0 if result.wasSuccessful() and unchanged else 1)
