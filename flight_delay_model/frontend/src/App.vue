/**
 * FlightCast — flight delay prediction (IS6400 group project).
 * Cathay-inspired layout: utility bar, white nav, warm hero panel, and a white
 * booking card with tabs ("Predict a delay" / "My predictions") overlapping it.
 *
 * Guests can use every feature without an account; signing in (SQLite) only
 * keeps the prediction history, and clicking a history entry re-displays the
 * prediction that was stored at the time.
 */
<script setup>
import {
  computed,
  onMounted,
  reactive,
  ref,
  watch,
} from 'vue'

import SearchSelect from './components/SearchSelect.vue'
import DelayPanel from './components/DelayPanel.vue'
import AuthModal from './components/AuthModal.vue'
import DateTimeInput from './components/DateTimeInput.vue'
import { isSelectableFlightDate } from './flightDate.js'

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
const authBusy = ref(false)
const authError = ref('')
const history = ref([])
const historyLoading = ref(false)
const historyListError = ref('')
const historyDetail = ref(null)
const historyDetailLoading = ref(false)
const selectedHistoryItem = ref(null)
const historyDetailError = ref('')

const HISTORY_PAGE_SIZE = 10
const historyPage = ref(1)
let historyDetailRequest = 0
let historyListRequest = 0
let predictionRequest = 0
let authStateVersion = 0

const historyPageCount = computed(() =>
  Math.max(1, Math.ceil(history.value.length / HISTORY_PAGE_SIZE)),
)

const pagedHistory = computed(() => {
  const start = (historyPage.value - 1) * HISTORY_PAGE_SIZE
  return history.value.slice(start, start + HISTORY_PAGE_SIZE)
})

const historyRangeStart = computed(() =>
  history.value.length
    ? (historyPage.value - 1) * HISTORY_PAGE_SIZE + 1
    : 0,
)

const historyRangeEnd = computed(() =>
  Math.min(historyPage.value * HISTORY_PAGE_SIZE, history.value.length),
)

function clearHistoryDetail() {
  historyDetailRequest += 1
  selectedHistoryItem.value = null
  historyDetail.value = null
  historyDetailLoading.value = false
  historyDetailError.value = ''
}

// Discard private data and pending responses synchronously when the account changes.
watch(() => user.value?.id, () => {
  authStateVersion += 1
  historyListRequest += 1
  predictionRequest += 1
  history.value = []
  historyLoading.value = false
  historyListError.value = ''
  historyPage.value = 1
  result.value = null
  loading.value = false
  clearHistoryDetail()
}, { flush: 'sync' })

function expireSession() {
  user.value = null
  authError.value = 'Your session has expired. Please sign in again.'
}

function changeHistoryPage(page) {
  if (page < 1 || page > historyPageCount.value) return
  historyPage.value = page
  error.value = ''
  clearHistoryDetail()
}

watch(historyPageCount, count => {
  if (historyPage.value > count) changeHistoryPage(count)
})

watch([tab, () => user.value?.id], () => {
  historyPage.value = 1
  error.value = ''
  clearHistoryDetail()
})

const DIST_SOURCE = {
  user: 'entered by you',
  route_history: 'route history median',
  origin_median: 'median from origin airport',
  dest_median: 'median to destination airport',
  global_median: 'global median',
}

function timeToHHMM(value) {
  const match = /^(\d{2}):(\d{2})$/.exec(String(value))
  if (!match) return null

  const hours = Number(match[1])
  const minutes = Number(match[2])

  if (hours > 23 || minutes > 59) return null

  return hours * 100 + minutes
}

async function submit() {
  if (loading.value) return

  error.value = ''
  result.value = null

  const required = [
    ['FL_DATE', 'flight date'],
    ['AIRLINE', 'airline'],
    ['ORIGIN', 'origin airport'],
    ['DEST', 'destination airport'],
    ['CRS_DEP_TIME', 'scheduled departure time'],
    ['CRS_ARR_TIME', 'scheduled arrival time'],
  ]

  for (const [key, name] of required) {
    if (!String(form[key]).trim()) {
      error.value = `Please fill in the ${name}.`
      return
    }
  }

  if (!isSelectableFlightDate(form.FL_DATE)) {
    error.value = 'Please use the calendar to select today or a future flight date.'
    return
  }

  const departureTime = timeToHHMM(form.CRS_DEP_TIME)
  const arrivalTime = timeToHHMM(form.CRS_ARR_TIME)

  if (departureTime === null || arrivalTime === null) {
    error.value = 'Please select valid departure and arrival times.'
    return
  }

  const payload = {
    ...form,
    AIRLINE: form.AIRLINE.trim(),
    ORIGIN: form.ORIGIN.trim().toUpperCase(),
    DEST: form.DEST.trim().toUpperCase(),
    CRS_DEP_TIME: departureTime,
    CRS_ARR_TIME: arrivalTime,
  }

  if (payload.ORIGIN === payload.DEST) {
    error.value = 'Please choose different origin and destination airports.'
    return
  }

  const distance = String(form.DISTANCE).trim()

  if (distance) {
    const value = Number(distance)

    if (!Number.isFinite(value) || value <= 0) {
      error.value = 'Distance must be a positive number, or leave it blank.'
      return
    }

    payload.DISTANCE = value
  } else {
    delete payload.DISTANCE
  }

  const request = ++predictionRequest
  const ownerId = user.value?.id
  loading.value = true

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    const data = await resp.json().catch(() => ({}))

    if (request !== predictionRequest || ownerId !== user.value?.id) return

    if (!resp.ok) {
      error.value = data.error || `Request failed (${resp.status})`
      return
    }

    result.value = data

    if (user.value) loadHistory()
  } catch (e) {
    if (request === predictionRequest && ownerId === user.value?.id) {
      error.value = 'Network error: ' + e.message
    }
  } finally {
    if (request === predictionRequest) loading.value = false
  }
}

async function loadHistory() {
  const ownerId = user.value?.id
  if (!ownerId) return
  const request = ++historyListRequest
  historyLoading.value = true
  historyListError.value = ''
  try {
    const resp = await fetch('/api/history')
    const data = await resp.json().catch(() => ({}))
    if (request !== historyListRequest || ownerId !== user.value?.id) return
    if (resp.status === 401) {
      expireSession()
      return
    }
    if (!resp.ok) throw new Error(data.error || 'Unable to load prediction history.')
    history.value = data.items || []
  } catch (e) {
    if (request === historyListRequest && ownerId === user.value?.id) {
      historyListError.value = e.message || 'Unable to load prediction history.'
    }
  } finally {
    if (request === historyListRequest) historyLoading.value = false
  }
}

function openTab(name) {
  tab.value = name
  clearHistoryDetail()
  if (name === 'history' && user.value) loadHistory()
}

async function openHistoryItem(item) {
  if (!user.value) return
  const ownerId = user.value.id
  const request = ++historyDetailRequest

  selectedHistoryItem.value = item
  historyDetail.value = null
  historyDetailError.value = ''
  historyDetailLoading.value = true

  try {
    const resp = await fetch('/api/history/' + item.id)
    const data = await resp.json().catch(() => ({}))

    if (request !== historyDetailRequest || ownerId !== user.value?.id) return

    if (resp.status === 401) {
      expireSession()
      return
    }

    if (!resp.ok) {
      throw new Error(data.error || 'Request failed (' + resp.status + ')')
    }

    if (!data.result) {
      throw new Error('This saved prediction is unavailable.')
    }

    historyDetail.value = data
  } catch (e) {
    if (request === historyDetailRequest) {
      historyDetailError.value = e.message || 'Unable to open prediction.'
    }
  } finally {
    if (request === historyDetailRequest) {
      historyDetailLoading.value = false
    }
  }
}

function openAuth(mode = 'signin') {
  authMode.value = mode
  authOpen.value = true
}

function onAuthenticated(u) {
  authStateVersion += 1
  authError.value = ''
  user.value = u
  authOpen.value = false
  loadHistory()
}

async function signOut() {
  if (authBusy.value || !user.value) return
  authBusy.value = true
  authError.value = ''
  try {
    const resp = await fetch('/api/auth/logout', { method: 'POST' })
    if (!resp.ok) throw new Error('Sign out failed.')
    authStateVersion += 1
    user.value = null
    tab.value = 'predict'
  } catch {
    authError.value = 'Unable to sign out. Please check your connection and try again.'
  } finally {
    authBusy.value = false
  }
}

function fmtWhen(s) {
  const d = new Date(String(s).replace(" ", "T"))
  return isNaN(d) ? s : d.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" })
}

onMounted(async () => {
  const requestVersion = authStateVersion
  try {
    const resp = await fetch('/api/auth/me')
    const data = await resp.json()
    if (requestVersion !== authStateVersion || !resp.ok) return
    user.value = data.user || null
    if (user.value) loadHistory()
  } catch { /* offline/dev — stay guest */ }
})
</script>

<template>
  <div class="site" :class="{ 'history-view': tab === 'history' }">

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
            <button class="linklike" :disabled="authBusy" @click="signOut">{{ authBusy ? 'Signing out…' : 'Sign out' }}</button>
          </template>
          <button v-else class="pill" @click="openAuth('signin')">Sign in / Join</button>
        </div>
      </div>
    </header>

    <p v-if="authError" class="auth-error" role="alert">{{ authError }}</p>

    <section v-if="tab === 'predict'" class="hero">
      <div class="hero-inner">
        <h1>Know your delay before you fly.</h1>
        <p>Departure delay probabilities for U.S. domestic flights.</p>
      </div>
    </section>

    <main class="main" :class="{ 'has-live-result': tab === 'predict' && result }">
      <div class="bookcard">
        <div class="tabs">
          <button :class="{ on: tab === 'predict' }" @click="openTab('predict')">Predict a delay</button>
          <button :class="{ on: tab === 'history' }" @click="openTab('history')">My predictions</button>
        </div>

        <!-- predict tab -->
        <form
  v-if="tab === 'predict'"
  class="booking"
  novalidate
  @submit.prevent="submit"
>
  <div class="form-heading">
    <div>
      <p class="eyebrow">PLAN YOUR JOURNEY</p>
      <h2>Flight details</h2>
    </div>

    <p class="form-note">
      Enter your route and schedule.<br />
      All times are local to each airport.
    </p>
  </div>

  <div class="grid">
    <SearchSelect
      class="span2"
      label="From"
      placeholder="Airport code or city, e.g. ATL"
      search-url="/api/airports"
      value-key="code"
      v-model="form.ORIGIN"
      required
    >
      <template #default="{ opt }">
        <span class="opt-code">{{ opt.code }}</span>
        <span class="opt-name">{{ opt.city }}, {{ opt.state }}</span>
      </template>
    </SearchSelect>

    <SearchSelect
      class="span2"
      label="To"
      placeholder="Airport code or city, e.g. DEN"
      search-url="/api/airports"
      value-key="code"
      v-model="form.DEST"
      required
    >
      <template #default="{ opt }">
        <span class="opt-code">{{ opt.code }}</span>
        <span class="opt-name">{{ opt.city }}, {{ opt.state }}</span>
      </template>
    </SearchSelect>

    <div class="field">
      <label for="flight-date">
        Flight date<span class="req">*</span>
      </label>
      <DateTimeInput
        id="flight-date"
        v-model="form.FL_DATE"
        mode="date"
        required
      />
    </div>

    <SearchSelect
      label="Airline"
      placeholder="Code or airline name"
      search-url="/api/airlines"
      value-key="name"
      v-model="form.AIRLINE"
      required
    >
      <template #default="{ opt }">
        <span class="opt-code">{{ opt.code }}</span>
        <span class="opt-name">{{ opt.name }}</span>
      </template>
    </SearchSelect>

    <div class="field">
      <label for="departure-time">
        Departure time<span class="req">*</span>
      </label>
      <DateTimeInput
        id="departure-time"
        v-model="form.CRS_DEP_TIME"
        mode="time"
        required
      />
    </div>

    <div class="field">
      <label for="arrival-time">
        Arrival time<span class="req">*</span>
      </label>
      <DateTimeInput
        id="arrival-time"
        v-model="form.CRS_ARR_TIME"
        mode="time"
        required
      />
    </div>
  </div>

  <details class="optional-details">
    <summary>Optional details</summary>

    <div class="optional-content">
      <div class="field">
        <label for="flight-distance">Distance in miles</label>
        <input
          id="flight-distance"
          v-model="form.DISTANCE"
          type="number"
          min="0"
          step="any"
          placeholder="Leave blank for automatic lookup"
        />
      </div>

      <p class="field-hint">
        We use the route's historical distance unless you enter a value.
      </p>
    </div>
  </details>

  <div class="actions">
    <p v-if="error" class="error" role="alert">
      {{ error }}
    </p>

    <p class="form-help" role="status">
      {{
        loading
          ? 'Preparing your departure forecast…'
          : 'No account needed to get a forecast.'
      }}
    </p>

    <button class="go" type="submit" :disabled="loading">
      {{ loading ? 'Predicting…' : 'Predict departure delay' }}
    </button>
  </div>
</form>

        <!-- history tab -->
        <div v-else class="historypane split-history">
          <div v-if="!user" class="signin-prompt">
            <p>
              <b>Save and revisit your flight forecasts.</b><br />
              Sign in to view your predictions.
            </p>
            <button class="go" @click="openAuth('signin')">
              Sign in / Join
            </button>
          </div>

          <template v-else>
            <div class="history-heading">
              <h2>Your predictions</h2>
              <p>Select a flight. Its forecast appears on the right.</p>
            </div>

            <div class="history-workspace">
              <section class="history-master" aria-label="Saved flights">
                <div v-if="historyLoading" class="empty" role="status">
                  Loading your predictions…
                </div>

                <div v-else-if="historyListError" class="empty history-list-error" role="alert">
                  <p>{{ historyListError }}</p>
                  <button type="button" class="linklike" @click="loadHistory">Try again</button>
                </div>

                <div v-else-if="!history.length" class="empty">
                  No saved predictions yet.
                </div>

                <ul v-else class="prediction-list">
                  <li v-for="it in pagedHistory" :key="it.id">
                    <button
                      type="button"
                      class="prediction-row"
                      :class="{ selected: selectedHistoryItem?.id === it.id }"
                      :aria-pressed="selectedHistoryItem?.id === it.id"
                      aria-controls="saved-prediction-detail"
                      @click="openHistoryItem(it)"
                    >
                      <span class="prediction-route">
                        <b>✈️ {{ it.origin }} → {{ it.dest }}</b>
                        <small>{{ it.airline }} · {{ it.fl_date }}</small>
                      </span>
                      <span class="prediction-arrow" aria-hidden="true">→</span>
                    </button>
                  </li>
                </ul>

                <nav
                  class="history-pagination"
                  aria-label="Prediction history pages"
                >
                  <span>
                    {{ historyRangeStart }}–{{ historyRangeEnd }}
                    of {{ history.length }} loaded predictions
                  </span>

                  <div class="page-controls">
                    <button
                      type="button"
                      :disabled="historyPage === 1 || historyLoading"
                      @click="changeHistoryPage(historyPage - 1)"
                    >
                      Previous
                    </button>

                    <span role="status">
                      Page {{ historyPage }} of {{ historyPageCount }}
                    </span>

                    <button
                      type="button"
                      :disabled="historyPage === historyPageCount || historyLoading"
                      @click="changeHistoryPage(historyPage + 1)"
                    >
                      Next
                    </button>
                  </div>
                </nav>
              </section>

              <aside
                id="saved-prediction-detail"
                class="history-preview"
                aria-label="Selected flight prediction"
                :aria-busy="historyDetailLoading"
              >
                <template v-if="selectedHistoryItem">
                  <header class="preview-heading">
                    <div>
                      <h3>
                        {{ selectedHistoryItem.origin }}
                        → {{ selectedHistoryItem.dest }}
                      </h3>
                      <p>
                        {{ selectedHistoryItem.airline }}
                        · {{ selectedHistoryItem.fl_date }}
                      </p>
                    </div>

                    <button
                      type="button"
                      class="preview-close"
                      aria-label="Close prediction details"
                      @click="clearHistoryDetail"
                    >
                      ×
                    </button>
                  </header>

                  <div :key="selectedHistoryItem.id" class="preview-body">
                    <p
                      v-if="historyDetailLoading"
                      class="preview-loading"
                      role="status"
                    >
                      Loading this flight's forecast…
                    </p>

                    <div
                      v-else-if="historyDetailError"
                      class="preview-error"
                      role="alert"
                    >
                      <p>{{ historyDetailError }}</p>
                      <button
                        type="button"
                        class="preview-retry"
                        @click="openHistoryItem(selectedHistoryItem)"
                      >
                        Try again
                      </button>
                    </div>

                    <template v-else-if="historyDetail?.result">
                      <DelayPanel
                        :key="historyDetail.id"
                        title="Departure"
                        :verdict="historyDetail.result.verdicts?.dep_delay"
                        :scenario="historyDetail.result.scenarios?.dep_delay"
                      />
                      <p class="preview-saved">
                        Saved {{ fmtWhen(historyDetail.created_at) }}
                      </p>
                    </template>
                  </div>
                </template>

                <div v-else class="preview-placeholder">
                  <span aria-hidden="true">✈️</span>
                  <h3>Select a flight</h3>
                  <p>Your departure forecast will appear here.</p>
                </div>
              </aside>
            </div>
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
      </section>

      <p v-if="tab !== 'predict' && error" class="error center">{{ error }}</p>
    </main>

    <AuthModal v-if="authOpen" :initial-mode="authMode"
               @close="authOpen = false" @authenticated="onAuthenticated" />
  </div>
</template>

<style scoped>
.site {
  --content-width: 1160px;
  --control-height: 44px;
  display: flex;
  flex-direction: column;
  height: 100dvh;
  min-height: 0;
  overflow: hidden;
}

.site:not(.history-view) { --control-height: 48px; }
.site:not(.history-view) .main { margin-top: 18px; padding-bottom: 18px; }
.site:not(.history-view) .tabs { padding: 12px 20px; }
.site:not(.history-view) .tabs button { min-height: 40px; padding: 8px 18px; font-size: 14px; }

.nav { flex-shrink: 0; background: #fff; border-bottom: 1px solid var(--hairline); z-index: 50; }
.nav-inner, .hero-inner, .main { width: 100%; max-width: var(--content-width); margin-left: auto; margin-right: auto; padding-left: 24px; padding-right: 24px; }
.nav-inner { display: flex; align-items: center; gap: 30px; min-height: 64px; padding-top: 10px; padding-bottom: 10px; }
.brand { display: flex; align-items: center; gap: 9px; cursor: pointer; }
.logo { width: 32px; height: 32px; }
.logo rect { fill: var(--green-ink); }
.word { color: var(--green-ink); font-size: 20px; font-weight: 700; letter-spacing: -.02em; }
.links { display: flex; flex: 1; gap: 24px; }
.links a { padding: 5px 0; border-bottom: 2px solid transparent; color: var(--ink-body); font-size: 13px; text-decoration: none; cursor: pointer; }
.links a.active { border-bottom-color: var(--green-brand); color: var(--green-ink); font-weight: 600; }
.auth { display: flex; align-items: center; gap: 12px; }
.auth-error { width: min(1180px, calc(100% - 40px)); box-sizing: border-box; margin: 12px auto 0; padding: 10px 14px; border-radius: 10px; background: #fff5f5; color: #a52b35; font-size: 14px; }
.history-list-error p { margin: 0 0 10px; }
.hello { color: var(--ink-body); font-size: 12px; }
.linklike { padding: 0; border: 0; background: none; color: var(--green-brand); font: inherit; font-size: 12px; cursor: pointer; }
.linklike:hover { text-decoration: underline; }
.pill { padding: 7px 16px; border: 1px solid var(--green-ink); border-radius: 999px; background: #fff; color: var(--green-ink); font-size: 12px; font-weight: 600; cursor: pointer; }

.hero { flex-shrink: 0; background: var(--panel); }
.hero-inner { display: flex; align-items: center; justify-content: space-between; gap: 28px; padding-top: 26px; padding-bottom: 26px; }
.hero h1 { margin: 0; color: var(--green-ink); font-size: clamp(27px, 2.5vw, 34px); line-height: 1.25; letter-spacing: -.035em; }
.hero p { max-width: 350px; margin: 0; color: var(--ink-body); font-size: 13px; line-height: 1.7; text-align: right; }

.main { display: flex; flex: 1; flex-direction: column; min-height: 0; gap: 14px; margin-top: 16px; padding-bottom: 16px; }
.bookcard { display: flex; flex-direction: column; min-width: 0; min-height: 0; max-height: 100%; border: 1px solid var(--hairline); border-radius: 16px; background: #fff; overflow: hidden; }
.history-view .bookcard { flex: 1; }
.tabs { display: flex; flex-shrink: 0; gap: 8px; padding: 10px 18px; border-bottom: 1px solid var(--hairline); }
.tabs button { min-height: 36px; padding: 7px 16px; border: 0; border-radius: 9px; background: var(--panel-2); color: var(--ink-body); font-size: 13px; cursor: pointer; }
.tabs button.on { background: var(--green-ink); color: #fff; font-weight: 600; }

.booking { flex: 1; min-height: 0; padding: 24px 26px; overflow-y: auto; }
.form-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; margin-bottom: 20px; }
.eyebrow { margin: 0 0 5px; color: var(--green-brand); font-size: 11px; font-weight: 700; letter-spacing: .12em; }
.form-heading h2 { margin: 0; color: var(--green-ink); font-size: 23px; line-height: 1.3; letter-spacing: -.02em; }
.form-note { margin: 0; color: var(--ink-soft); font-size: 13px; line-height: 1.6; text-align: right; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); align-items: start; gap: 20px; }
.span2 { grid-column: span 2; }
.req { margin-left: 3px; color: #b23a3a; }
.booking :deep(.field) { display: flex; flex-direction: column; min-width: 0; gap: 8px; }
.booking :deep(.field > label) { color: var(--ink-body); font-size: 13px; font-weight: 600; line-height: 20px; }
.booking :deep(.field input) { width: 100%; min-width: 0; height: var(--control-height); padding: 0 12px; border: 1px solid var(--hairline); border-radius: 10px; background: #fff; color: var(--ink); font: inherit; font-size: 15px; color-scheme: light; }
.booking :deep(.field input:hover) { border-color: #9fb3cc; }
.booking :deep(.field input:focus) { outline: none; border-color: var(--green-brand); box-shadow: 0 0 0 3px rgba(36,90,153,.1); }
.booking :deep(.field input::placeholder) { color: var(--ink-soft); font-size: 13px; }
.opt-code { flex: 0 0 auto; font-weight: 600; white-space: nowrap; }
.opt-name { flex: 1; min-width: 0; overflow: hidden; color: var(--ink-soft); font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }

.optional-details { margin-top: 20px; padding-top: 14px; border-top: 1px solid var(--hairline); }
.optional-details summary { width: fit-content; color: var(--green-brand); font-size: 13px; font-weight: 600; cursor: pointer; }
.optional-content { max-width: 400px; margin-top: 12px; }
.field-hint { margin: 8px 0 0; color: var(--ink-soft); font-size: 12px; line-height: 1.6; }
.actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; margin-top: 20px; padding-top: 18px; border-top: 1px solid var(--hairline); }
.form-help { margin: 0; color: var(--ink-soft); font-size: 13px; }
.booking .go { min-height: 46px; padding: 11px 22px; font-size: 14px; }
.go { min-height: 42px; padding: 10px 20px; border: 0; border-radius: 10px; background: var(--green-btn); color: #fff; font-size: 13px; font-weight: 600; cursor: pointer; }
.go:hover { background: var(--green-btn-hover); }
.go:disabled { opacity: .6; cursor: wait; }
.error { margin: 0; color: #b23a3a; font-size: 12px; }
.actions .error { flex-basis: 100%; padding: 9px 12px; border: 1px solid #f1c9c9; border-radius: 8px; background: #fff5f5; }
.error.center { text-align: center; }

.results { display: flex; flex-direction: column; min-width: 0; min-height: 0; gap: 10px; margin: 0; overflow-y: auto; }
.meta { padding: 10px 12px; border: 1px solid var(--hairline); border-radius: 10px; background: #fff; color: var(--ink-body); font-size: 11px; line-height: 1.6; }
.panels { display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; }

.historypane { display: flex; flex: 1; flex-direction: column; min-height: 0; padding: 18px; }
.history-heading { flex-shrink: 0; margin-bottom: 14px; }
.history-heading h2 { margin: 0; color: var(--green-ink); font-size: 20px; line-height: 1.3; }
.history-heading p { margin: 4px 0 0; color: var(--ink-soft); font-size: 12px; }
.signin-prompt { padding: 32px 12px; color: var(--ink-body); text-align: center; }
.signin-prompt .go { margin-top: 12px; }
.history-workspace { display: grid; flex: 1; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); min-height: 0; gap: 18px; }
.history-master { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.prediction-list { display: flex; flex: 1; flex-direction: column; min-height: 0; gap: 6px; margin: 0; padding: 3px; overflow-y: auto; list-style: none; scrollbar-gutter: stable; }
.prediction-list > li { flex: 0 0 auto; }
.prediction-row { display: grid; grid-template-columns: minmax(0, 1fr) 18px; align-items: center; width: 100%; min-height: 54px; gap: 12px; padding: 9px 12px; border: 1px solid var(--hairline); border-radius: 10px; background: #fff; color: var(--ink); font: inherit; text-align: left; cursor: pointer; }
.prediction-row:hover { background: #f5f8fc; border-color: #a9bfdc; }
.prediction-row.selected { border-color: #245a99; background: #eaf2fd; }
.prediction-route { display: flex; flex-direction: column; min-width: 0; gap: 2px; }
.prediction-route b { color: var(--green-ink); font-size: 13px; }
.prediction-route small { overflow: hidden; color: var(--ink-soft); font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
.history-pagination { display: flex; flex-shrink: 0; flex-wrap: wrap; gap: 8px; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--hairline); color: var(--ink-soft); font-size: 11px; }
.page-controls { display: flex; align-items: center; justify-content: space-between; width: 100%; gap: 8px; }
.page-controls button { min-height: 30px; padding: 5px 10px; border: 1px solid var(--hairline); border-radius: 8px; background: #fff; color: var(--green-ink); font: inherit; cursor: pointer; }
.page-controls button:hover:not(:disabled) { background: var(--panel); }
.page-controls button:disabled { opacity: .4; cursor: default; }
.history-master > .empty { display: grid; flex: 1; place-content: center; margin: 0; color: var(--ink-soft); font-size: 13px; text-align: center; }

.history-preview { display: flex; flex-direction: column; min-width: 0; min-height: 0; overflow: hidden; border: 1px solid var(--hairline); border-radius: 12px; background: #f7f9fc; }
.preview-heading { display: flex; flex-shrink: 0; align-items: flex-start; justify-content: space-between; gap: 12px; padding: 12px 14px; border-bottom: 1px solid var(--hairline); background: #fff; }
.preview-heading h3 { margin: 0; color: var(--green-ink); font-size: 15px; }
.preview-heading p { margin: 3px 0 0; color: var(--ink-soft); font-size: 10px; }
.preview-close { display: grid; flex: 0 0 26px; place-items: center; width: 26px; height: 26px; padding: 0; border: 0; border-radius: 8px; background: #edf2f8; color: var(--green-ink); font-size: 21px; cursor: pointer; }
.preview-close:hover { background: #dfe9f6; }
.preview-body { flex: 1; min-height: 0; padding: 10px; overflow-y: auto; }
.preview-loading, .preview-error { padding: 18px 8px; color: var(--ink-body); font-size: 12px; }
.preview-error p { margin-top: 0; overflow-wrap: anywhere; }
.preview-retry { padding: 8px 14px; border: 1px solid var(--hairline); border-radius: 8px; background: #fff; color: var(--green-ink); font: inherit; cursor: pointer; }
.preview-placeholder { display: flex; flex: 1; flex-direction: column; align-items: center; justify-content: center; padding: 24px; color: var(--ink-soft); text-align: center; }
.preview-placeholder > span { font-size: 26px; }
.preview-placeholder h3 { margin: 10px 0 4px; color: var(--green-ink); font-size: 15px; }
.preview-placeholder p { margin: 0; font-size: 12px; }
.preview-saved { margin: 8px 0 0; color: var(--ink-soft); font-size: 10px; }
button:focus-visible, .links a:focus-visible { outline: 2px solid var(--green-brand); outline-offset: 3px; }

@media (min-width: 1000px) {
  .main.has-live-result { display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(0, 1fr); grid-template-rows: minmax(0, 1fr); }
  .has-live-result .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .has-live-result .span2 { grid-column: span 1; }
  .has-live-result .form-heading { flex-direction: column; gap: 6px; }
  .has-live-result .form-note { text-align: left; }
}

@media (min-width: 721px) and (max-width: 999px) {
  .hero-inner { gap: 16px; }
  .hero h1 { font-size: 24px; }
  .hero p { max-width: 260px; font-size: 11px; }
  .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .span2 { grid-column: span 1; }
  .main.has-live-result { overflow-y: auto; }
  .has-live-result .bookcard, .has-live-result .results { flex: 0 0 auto; max-height: none; overflow: visible; }
}

@media (max-width: 720px) {
  .site { height: auto; min-height: 100dvh; overflow: visible; }
  .nav { position: sticky; top: 0; }
  .nav-inner, .hero-inner, .main { padding-left: 16px; padding-right: 16px; }
  .nav-inner { flex-wrap: wrap; gap: 10px 18px; padding-top: 10px; padding-bottom: 10px; }
  .auth { margin-left: auto; }
  .links { order: 3; flex-basis: 100%; gap: 24px; }
  .word { font-size: 18px; }
  .hero-inner { align-items: flex-start; flex-direction: column; gap: 8px; padding-top: 16px; padding-bottom: 16px; }
  .hero h1 { font-size: 25px; }
  .hero p { max-width: none; text-align: left; }
  .main { flex: initial; margin-top: 12px; }
  .bookcard { max-height: none; }
  .booking { padding: 16px; overflow: visible; }
  .tabs { padding: 10px 12px; }
  .tabs button { flex: 1; padding: 7px 10px; font-size: 12px; }
  .form-heading { flex-direction: column; gap: 8px; }
  .form-note { text-align: left; }
  .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
  .span2 { grid-column: span 1; }
  .actions .go { width: 100%; }
  .historypane { padding: 14px; }
  .history-workspace { grid-template-columns: minmax(0, 1fr); gap: 14px; }
  .history-preview { grid-row: 1; position: sticky; top: 116px; z-index: 5; max-height: 48dvh; }
  .preview-body { flex: 0 1 auto; }
  .preview-placeholder { min-height: 100px; }
  .history-master { height: 360px; }
  .results { flex-shrink: 0; overflow: visible; }
}

@media (max-width: 380px) {
  .grid { grid-template-columns: minmax(0, 1fr); }
  .hello { max-width: 80px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
}
</style>
