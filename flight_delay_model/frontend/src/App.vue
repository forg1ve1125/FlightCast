<script setup>
/**
 * FlightCast — flight delay prediction (IS6400 group project).
 * Cathay-inspired layout: utility bar, white nav, warm hero panel, and a white
 * booking card with tabs ("Predict a delay" / "My predictions") overlapping it.
 *
 * Guests can use every feature without an account; signing in (SQLite) only
 * keeps the prediction history, and clicking a history entry re-displays the
 * prediction that was stored at the time.
 */
import { onMounted, reactive, ref } from 'vue'
import SearchSelect from './components/SearchSelect.vue'
import DelayPanel from './components/DelayPanel.vue'
import AuthModal from './components/AuthModal.vue'

const form = reactive({
  FL_DATE: '', AIRLINE: '', ORIGIN: '', DEST: '',
  CRS_DEP_TIME: '', CRS_ARR_TIME: '', DISTANCE: '',
})

const loading = ref(false)
const error = ref('')
const result = ref(null)

const user = ref(null)
const tab = ref('predict')          // 'predict' | 'history'
const authOpen = ref(false)
const authMode = ref('signin')
const history = ref([])
const historyLoading = ref(false)
const historyDetail = ref(null)     // {id, created_at, payload, result}

const DIST_SOURCE = {
  user: 'entered by you',
  route_history: 'route history median',
  origin_median: 'median from origin airport',
  dest_median: 'median to destination airport',
  global_median: 'global median',
}

async function submit() {
  error.value = ''
  result.value = null
  const required = [['FL_DATE', 'flight date'], ['AIRLINE', 'airline'], ['ORIGIN', 'origin airport'],
                    ['DEST', 'destination airport'], ['CRS_DEP_TIME', 'scheduled departure time'],
                    ['CRS_ARR_TIME', 'scheduled arrival time']]
  for (const [k, name] of required) {
    if (!String(form[k]).trim()) { error.value = `Please fill in the ${name}.`; return }
  }
  const payload = { ...form }
  if (!String(payload.DISTANCE).trim()) delete payload.DISTANCE

  loading.value = true
  try {
    const resp = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const data = await resp.json().catch(() => ({}))
    if (!resp.ok) { error.value = data.error || `Request failed (${resp.status})`; return }
    result.value = data
    if (user.value) loadHistory()   // silent refresh; saves happen server-side
  } catch (e) {
    error.value = 'Network error: ' + e.message
  } finally {
    loading.value = false
  }
}

async function loadHistory() {
  if (!user.value) return
  historyLoading.value = true
  try {
    const resp = await fetch('/api/history')
    const data = await resp.json().catch(() => ({}))
    if (resp.ok) history.value = data.items || []
  } finally {
    historyLoading.value = false
  }
}

function openTab(name) {
  tab.value = name
  historyDetail.value = null
  if (name === 'history' && user.value) loadHistory()
}

async function openHistoryItem(item) {
  error.value = ''
  try {
    const resp = await fetch(`/api/history/${item.id}`)
    const data = await resp.json().catch(() => ({}))
    if (!resp.ok) { error.value = data.error || `Request failed (${resp.status})`; return }
    historyDetail.value = data
  } catch (e) {
    error.value = 'Network error: ' + e.message
  }
}

function openAuth(mode = 'signin') {
  authMode.value = mode
  authOpen.value = true
}

function onAuthenticated(u) {
  user.value = u
  authOpen.value = false
  loadHistory()
}

async function signOut() {
  await fetch('/api/auth/logout', { method: 'POST' }).catch(() => {})
  user.value = null
  history.value = []
  historyDetail.value = null
  tab.value = 'predict'
}

function fmtWhen(s) {
  const d = new Date(String(s).replace(" ", "T"))
  return isNaN(d) ? s : d.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" })
}

onMounted(async () => {
  try {
    const resp = await fetch('/api/auth/me')
    const data = await resp.json()
    user.value = data.user
    if (user.value) loadHistory()
  } catch { /* offline/dev — stay guest */ }
})
</script>

<template>
  <div class="site">
    <div class="utilbar">
      <span>IS6400 Group Project · U.S. domestic flights 2019–2023</span>
      <span class="util-right">Machine-learning delay forecasts</span>
    </div>

    <header class="nav">
      <div class="nav-inner">
        <div class="brand" @click="openTab('predict')">
          <svg class="logo" viewBox="0 0 64 64" aria-hidden="true">
            <rect width="64" height="64" rx="14" fill="#00332E"/>
            <path d="M32 9 L36.5 27.5 L55 35 L36.5 39.5 L32 55 L27.5 39.5 L9 35 L27.5 27.5 Z"
                  fill="#fff" opacity="0.95"/>
          </svg>
          <span class="word">FlightCast</span>
        </div>
        <nav class="links">
          <a :class="{ active: tab === 'predict' }" @click.prevent="openTab('predict')" href="#">Predict</a>
          <a :class="{ active: tab === 'history' }" @click.prevent="openTab('history')" href="#">My predictions</a>
        </nav>
        <div class="auth">
          <template v-if="user">
            <span class="hello">Hi, {{ user.username }}</span>
            <button class="linklike" @click="signOut">Sign out</button>
          </template>
          <button v-else class="pill" @click="openAuth('signin')">Sign in / Join</button>
        </div>
      </div>
    </header>

    <section v-if="tab === 'predict'" class="hero">
      <div class="hero-inner">
        <h1>Know your delay<br />before you fly.</h1>
        <p>Probability forecasts for U.S. domestic flights — the chance of your
           departure running on time, and of every band of delay.</p>
      </div>
    </section>

    <main class="main">
      <div class="bookcard">
        <div class="tabs">
          <button :class="{ on: tab === 'predict' }" @click="openTab('predict')">Predict a delay</button>
          <button :class="{ on: tab === 'history' }" @click="openTab('history')">My predictions</button>
        </div>

        <!-- predict tab -->
        <form v-if="tab === 'predict'" class="booking" @submit.prevent="submit">
          <div class="grid">
            <div class="field">
              <label>Flight date (YYYY-MM-DD)<span class="req">*</span></label>
              <input v-model="form.FL_DATE" placeholder="2024-07-04" inputmode="numeric" />
            </div>
            <SearchSelect label="Airline" placeholder="e.g. DL or Delta"
                          search-url="/api/airlines" value-key="name" v-model="form.AIRLINE" required>
              <template #default="{ opt }">
                <span class="opt-code">{{ opt.code }}</span><span class="opt-name">{{ opt.name }}</span>
              </template>
            </SearchSelect>
            <SearchSelect label="From" placeholder="e.g. ATL or Atlanta"
                          search-url="/api/airports" value-key="code" v-model="form.ORIGIN" required>
              <template #default="{ opt }">
                <span class="opt-code">{{ opt.code }}</span><span class="opt-name">{{ opt.city }}, {{ opt.state }}</span>
              </template>
            </SearchSelect>
            <SearchSelect label="To" placeholder="e.g. DEN or Denver"
                          search-url="/api/airports" value-key="code" v-model="form.DEST" required>
              <template #default="{ opt }">
                <span class="opt-code">{{ opt.code }}</span><span class="opt-name">{{ opt.city }}, {{ opt.state }}</span>
              </template>
            </SearchSelect>
            <div class="field">
              <label>Scheduled departure (hhmm)<span class="req">*</span></label>
              <input v-model="form.CRS_DEP_TIME" placeholder="1830" inputmode="numeric" />
            </div>
            <div class="field">
              <label>Scheduled arrival (hhmm)<span class="req">*</span></label>
              <input v-model="form.CRS_ARR_TIME" placeholder="2015" inputmode="numeric" />
            </div>
            <div class="field span2">
              <label>Distance (miles, optional)</label>
              <input v-model="form.DISTANCE" placeholder="Auto-filled from route history" inputmode="decimal" />
            </div>
          </div>
          <div class="actions">
            <p v-if="error" class="error">{{ error }}</p>
            <button class="go" type="submit" :disabled="loading">
              {{ loading ? 'Predicting…' : 'Predict delay' }}
            </button>
          </div>
        </form>

        <!-- history tab -->
        <div v-else class="historypane">
          <div v-if="!user" class="signin-prompt">
            <p><b>Sign in to keep a history of your predictions.</b><br />
               You can still use every feature without an account — signing in simply saves
               each successful prediction so you can open it again later.</p>
            <button class="go" @click="openAuth('signin')">Sign in / Join</button>
          </div>
          <template v-else>
            <div v-if="historyLoading" class="empty">Loading your predictions…</div>
            <div v-else-if="!history.length" class="empty">
              No saved predictions yet — run a prediction and it will appear here.
            </div>
            <ul v-else class="hlist">
              <li v-for="it in history" :key="it.id" @click="openHistoryItem(it)">
                <div class="hleft">
                  <span class="hroute">{{ it.origin }} → {{ it.dest }}</span>
                  <span class="hmeta">{{ it.airline }} · {{ it.fl_date }}</span>
                </div>
                <div class="hright">
                  <span class="hv">{{ it.verdicts?.dep_delay }}</span>
                  <span class="hwhen">{{ fmtWhen(it.created_at) }}</span>
                </div>
              </li>
            </ul>
          </template>
        </div>
      </div>

      <!-- live prediction results (departure only) -->
      <section v-if="tab === 'predict' && result" class="results">
        <div class="meta">
          {{ result.resolved.airline }} · {{ result.resolved.origin }} → {{ result.resolved.dest }}
          · {{ Number(result.resolved.distance).toLocaleString() }} mi
          ({{ DIST_SOURCE[result.distance_source] || result.distance_source }})
          <span v-if="result.weather_source">· Weather: {{ result.weather_source }}</span>
        </div>
        <div class="panels">
          <DelayPanel title="Departure" :verdict="result.verdicts?.dep_delay"
                      :scenario="result.scenarios?.dep_delay" />
        </div>
        <p class="scenario-note">{{ result.scenarios?.dep_delay?.summary }}</p>
      </section>

      <!-- stored prediction opened from history (departure only) -->
      <section v-if="tab === 'history' && historyDetail" class="results">
        <div class="meta stor">
          <button class="linklike back" @click="historyDetail = null">← All predictions</button>
          Saved prediction from {{ fmtWhen(historyDetail.created_at) }} ·
          {{ historyDetail.result.resolved?.airline }} ·
          {{ historyDetail.result.resolved?.origin }} → {{ historyDetail.result.resolved?.dest }}
          · {{ Number(historyDetail.result.resolved?.distance || 0).toLocaleString() }} mi
          ({{ DIST_SOURCE[historyDetail.result.distance_source] || historyDetail.result.distance_source }})
        </div>
        <div class="panels">
          <DelayPanel title="Departure" :verdict="historyDetail.result.verdicts?.dep_delay"
                      :scenario="historyDetail.result.scenarios?.dep_delay" />
        </div>
      </section>
      <p v-if="tab !== 'predict' && error" class="error center">{{ error }}</p>
    </main>

    <AuthModal v-if="authOpen" :initial-mode="authMode"
               @close="authOpen = false" @authenticated="onAuthenticated" />
  </div>
</template>

<style scoped>
.site { min-height: 100vh; display: flex; flex-direction: column; }

/* thin utility bar */
.utilbar {
  display: flex; justify-content: space-between; padding: 8px 28px;
  font-size: 12px; color: var(--ink-soft); border-bottom: 1px solid var(--hairline);
}

/* main nav */
.nav { background: #fff; border-bottom: 1px solid var(--hairline); position: sticky; top: 0; z-index: 50; }
.nav-inner {
  max-width: 1120px; margin: 0 auto; padding: 12px 28px;
  display: flex; align-items: center; gap: 36px;
}
.brand { display: flex; align-items: center; gap: 10px; cursor: pointer; }
.logo { width: 34px; height: 34px; }
.word { font-size: 20px; font-weight: 700; color: var(--green-ink); letter-spacing: 0.01em; }
.links { display: flex; gap: 26px; flex: 1; }
.links a {
  color: var(--ink-body); text-decoration: none; font-size: 14.5px; cursor: pointer;
  padding-bottom: 4px; border-bottom: 2px solid transparent;
}
.links a:hover { color: var(--ink); }
.links a.active { color: var(--green-ink); font-weight: 600; border-bottom-color: var(--green-brand); }
.auth { display: flex; align-items: center; gap: 14px; }
.hello { font-size: 14px; color: var(--ink-body); }
.linklike { border: 0; background: none; color: var(--green-brand); font-size: 14px; cursor: pointer; padding: 0; }
.linklike:hover { text-decoration: underline; }
.pill {
  border: 1.5px solid var(--green-ink); background: #fff; color: var(--green-ink);
  border-radius: 999px; padding: 8px 20px; font-size: 14px; font-weight: 600; cursor: pointer;
}
.pill:hover { background: var(--panel); }

/* hero */
.hero { background: var(--panel); }
.hero-inner { max-width: 1120px; margin: 0 auto; padding: 58px 28px 110px; }
.hero h1 { margin: 0 0 14px; font-size: 42px; line-height: 1.12; color: var(--green-ink); font-weight: 700; }
.hero p { margin: 0; max-width: 560px; color: var(--ink-body); font-size: 16px; }

/* main column */
.main { flex: 1; width: 100%; max-width: 1120px; margin: -70px auto 0; padding: 0 28px 48px; }

/* booking card with tabs */
.bookcard { background: #fff; border-radius: var(--radius); box-shadow: var(--card-shadow); overflow: visible; }
.tabs { display: flex; gap: 8px; padding: 12px 20px 0; }
.tabs button {
  border: 0; border-radius: 10px 10px 0 0; padding: 11px 22px; font-size: 14.5px;
  background: var(--panel-2); color: var(--ink-body); cursor: pointer; font-family: inherit;
}
.tabs button.on { background: var(--green-ink); color: #fff; font-weight: 600; }

.booking { padding: 22px 24px 24px; border-top: 1px solid var(--hairline); }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 22px 28px; align-items: end; }
@media (max-width: 980px) { .grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .grid { grid-template-columns: 1fr; } .span2 { grid-column: span 1; } }
.span2 { grid-column: span 2; }
.field { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.field label { font-size: 12px; color: var(--ink-body); letter-spacing: 0.02em; }
.req { color: #b23a3a; margin-left: 2px; }
.field input {
  border: 0; border-bottom: 1.5px solid #b9b7af; border-radius: 0; background: transparent;
  padding: 8px 2px; font-size: 15px; color: var(--ink); font-family: inherit; width: 100%;
}
.field input:focus { outline: none; border-bottom-color: var(--green-brand); }
.field input::placeholder { color: #a8a7a0; }
.actions { display: flex; align-items: center; justify-content: flex-end; gap: 18px; margin-top: 24px; flex-wrap: wrap; }
.go {
  background: var(--green-btn); color: #fff; border: 0; border-radius: 8px;
  padding: 12px 30px; font-size: 15px; font-weight: 600; cursor: pointer; font-family: inherit;
}
.go:hover { background: var(--green-btn-hover); }
.go:disabled { opacity: 0.6; cursor: wait; }
.error { color: #b23a3a; font-size: 13.5px; margin: 0; }
.error.center { text-align: center; margin-top: 16px; }

/* history pane */
.historypane { border-top: 1px solid var(--hairline); padding: 24px; }
.signin-prompt { text-align: center; color: var(--ink-body); padding: 22px 8px; }
.signin-prompt .go { margin-top: 14px; }
.empty { color: var(--ink-soft); text-align: center; padding: 28px 8px; }
.hlist { list-style: none; margin: 0; padding: 0; }
.hlist li {
  display: flex; justify-content: space-between; align-items: center; gap: 18px;
  padding: 14px 10px; border-bottom: 1px solid var(--hairline); cursor: pointer; flex-wrap: wrap;
}
.hlist li:hover { background: var(--panel); }
.hleft { display: flex; align-items: baseline; gap: 14px; min-width: 0; }
.hroute { font-weight: 600; color: var(--green-ink); font-size: 15px; }
.hmeta { color: var(--ink-soft); font-size: 13px; }
.hright { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; justify-content: flex-end; }
.hv { font-size: 13px; color: var(--ink-body); }
.hwhen { font-size: 12.5px; color: var(--ink-soft); }

/* results */
.results { margin-top: 26px; }
.meta { font-size: 13.5px; color: var(--ink-body); margin-bottom: 12px; }
.meta.stor { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.back { font-weight: 600; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
.scenario-note { color: var(--ink-soft); font-size: 13px; margin: 14px 0 0; }
/* SearchSelect option rows — slot content is scoped here, not in SearchSelect.
   Named opt-* to avoid colliding with the page-column .main rule above. */
.opt-code { font-weight: 600; flex: 0 0 auto; white-space: nowrap; }
.opt-name {
  color: var(--ink-soft); font-size: 13px; min-width: 0; flex: 1 1 auto;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
</style>
