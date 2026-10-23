/** Execute the actual compiled App.vue setup logic with controlled HTTP timing. */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import { after, test } from 'node:test'
import { localToday, isSelectableFlightDate } from '../src/flightDate.js'

const require = createRequire(new URL('../package.json', import.meta.url))
const vue = require('vue')
const { parse, compileScript } = require('@vue/compiler-sfc')
const frontend = fileURLToPath(new URL('..', import.meta.url))
const source = fs.readFileSync(path.join(frontend, 'src/App.vue'), 'utf8')
const descriptor = parse(source).descriptor
const compiled = compileScript(descriptor, { id: 'auth-history-regression' }).content
  .replace(/^import\s+[\s\S]*?\s+from\s+['"][^'"]+['"];?\s*$/gm, '')
  .replace('export default', 'const component =')
const records = []
const A = { id: 101, username: 'test_a' }
const B = { id: 202, username: 'test_b' }
const A_HISTORY = { id: 11, origin: 'AAA', dest: 'AAB' }
const B_HISTORY = { id: 22, origin: 'BBB', dest: 'BBC' }

function harness() {
  const requests = [], mounted = []
  const context = {
    ...vue,
    onMounted: callback => mounted.push(callback),
    SearchSelect: {}, DelayPanel: {}, AuthModal: {}, DateTimeInput: {},
    fetch: (url, options) => new Promise((resolve, reject) => requests.push({ url, options, resolve, reject })),
    console, Intl, isSelectableFlightDate,
  }
  const ui = vm.runInNewContext(compiled + '\ncomponent.setup({}, { expose() {} })', context)
  function reply(index, data, status = 200) {
    assert.ok(requests[index], `Missing request ${index}`)
    requests[index].resolve({ ok: status >= 200 && status < 300, status, json: async () => data })
  }
  return { ui, requests, mounted, reply }
}

async function flush() {
  for (let index = 0; index < 5; index++) await Promise.resolve()
  await vue.nextTick()
}

async function check(name, fn) {
  await test(name, async () => {
    try {
      await fn()
      records.push({ test: name, status: 'passed' })
    } catch (error) {
      records.push({ test: name, status: 'failed', detail: String(error) })
      throw error
    }
  })
}

await check('Normal history request displays current user records', async () => {
  const { ui, reply } = harness()
  ui.user.value = A
  await flush()
  const pending = ui.loadHistory()
  reply(0, { items: [A_HISTORY] })
  await pending
  assert.equal(ui.history.value[0].id, A_HISTORY.id)
  assert.equal(ui.historyLoading.value, false)
})

await check('Late A history cannot overwrite B after account switch', async () => {
  const { ui, reply } = harness()
  ui.user.value = A
  await flush()
  const oldRequest = ui.loadHistory()
  ui.onAuthenticated(B)
  await flush()
  reply(1, { items: [B_HISTORY] })
  await flush()
  reply(0, { items: [A_HISTORY] })
  await oldRequest
  assert.equal(ui.user.value.id, B.id)
  assert.equal(ui.history.value[0].id, B_HISTORY.id)
})

await check('Late A history cannot repopulate state after logout', async () => {
  const { ui, reply } = harness()
  ui.user.value = A
  await flush()
  const oldRequest = ui.loadHistory()
  const logout = ui.signOut()
  reply(1, { ok: true })
  await logout
  await flush()
  reply(0, { items: [A_HISTORY] })
  await oldRequest
  assert.equal(ui.user.value, null)
  assert.equal(ui.history.value.length, 0)
})

await check('Old A request does not stop B loading indicator', async () => {
  const { ui, reply } = harness()
  ui.user.value = A
  await flush()
  const oldRequest = ui.loadHistory()
  ui.onAuthenticated(B)
  await flush()
  reply(0, { items: [A_HISTORY] })
  await oldRequest
  assert.equal(ui.historyLoading.value, true)
  reply(1, { items: [B_HISTORY] })
  await flush()
})

await check('Expired session clears cached private data on 401', async () => {
  const { ui, reply } = harness()
  ui.user.value = A
  await flush()
  ui.history.value = [A_HISTORY]
  const pending = ui.loadHistory()
  reply(0, { error: 'Sign in required.' }, 401)
  await pending
  assert.equal(ui.user.value, null)
  assert.equal(ui.history.value.length, 0)
})

await check('Network failure during logout does not pretend logout succeeded', async () => {
  const { ui, requests } = harness()
  ui.user.value = A
  await flush()
  const logout = ui.signOut()
  requests[0].reject(new Error('Offline'))
  await logout
  assert.equal(ui.user.value?.id, A.id)
  assert.ok(ui.authError?.value || ui.error.value)
})

await check('Server failure during logout keeps identity and reports error', async () => {
  const { ui, reply } = harness()
  ui.user.value = A
  await flush()
  const logout = ui.signOut()
  reply(0, { error: 'Unavailable' }, 500)
  await logout
  assert.equal(ui.user.value?.id, A.id)
  assert.ok(ui.authError?.value || ui.error.value)
})

await check('Late A detail cannot appear after switching to B', async () => {
  const { ui, reply } = harness()
  ui.user.value = A
  await flush()
  const pending = ui.openHistoryItem(A_HISTORY)
  ui.onAuthenticated(B)
  await flush()
  reply(1, { items: [B_HISTORY] })
  reply(0, { id: A_HISTORY.id, result: { private: 'A' } })
  await pending
  assert.equal(ui.historyDetail.value, null)
  assert.equal(ui.selectedHistoryItem.value, null)
})

await check('Late prediction cannot appear after logout', async () => {
  const { ui, reply } = harness()
  ui.user.value = A
  await flush()
  Object.assign(ui.form, { FL_DATE: localToday(), AIRLINE: 'WN', ORIGIN: 'DEN', DEST: 'SMF',
    CRS_DEP_TIME: '22:45', CRS_ARR_TIME: '00:25', DISTANCE: '909' })
  const prediction = ui.submit()
  const logout = ui.signOut()
  reply(1, { ok: true })
  await logout
  await flush()
  reply(0, { input: { private: 'A' }, scenarios: {} })
  await prediction
  assert.equal(ui.result.value, null)
})

await check('Late initial session lookup cannot overwrite a newer login', async () => {
  const { ui, reply, mounted } = harness()
  const initialLookup = mounted[0]()
  ui.onAuthenticated(B)
  await flush()
  reply(1, { items: [B_HISTORY] })
  reply(0, { user: A })
  await initialLookup
  assert.equal(ui.user.value.id, B.id)
})

await check('History network errors are caught and loading ends', async () => {
  const { ui, requests } = harness()
  ui.user.value = A
  await flush()
  const pending = ui.loadHistory()
  requests[0].reject(new Error('Offline'))
  await assert.doesNotReject(pending)
  assert.equal(ui.historyLoading.value, false)
  assert.ok(ui.historyListError?.value || ui.error.value)
})

function fillFlight(ui, day = localToday()) {
  Object.assign(ui.form, { FL_DATE: day, AIRLINE: 'WN', ORIGIN: 'DEN', DEST: 'SMF',
    CRS_DEP_TIME: '22:45', CRS_ARR_TIME: '00:25', DISTANCE: '909' })
}

await check('Guest result is cleared at login without posting or backfilling it', async () => {
  const { ui, requests, reply } = harness()
  fillFlight(ui)
  const pending = ui.submit()
  reply(0, { scenarios: {}, guest: true })
  await pending
  assert.ok(ui.result.value)
  ui.onAuthenticated(A)
  reply(1, { items: [] })
  await flush()
  assert.equal(ui.result.value, null)
  assert.equal(ui.history.value.length, 0)
  assert.equal(requests.filter(r => r.options?.method === 'POST').length, 1)
  assert.equal(requests[1].url, '/api/history')
})

await check('Guest prediction arriving after login cannot populate account state', async () => {
  const { ui, requests, reply } = harness()
  fillFlight(ui)
  const pending = ui.submit()
  ui.onAuthenticated(A)
  reply(1, { items: [] })
  await flush()
  reply(0, { scenarios: {}, guest: true })
  await pending
  assert.equal(ui.result.value, null)
  assert.equal(ui.history.value.length, 0)
  assert.equal(requests.length, 2)
})

await check('Manually supplied past flight date is blocked before an HTTP request', async () => {
  const { ui, requests } = harness()
  fillFlight(ui, '2021-12-30')
  await ui.submit()
  assert.equal(requests.length, 0)
  assert.match(ui.error.value, /today or a future/)
})

await check('Invalid calendar date is blocked before an HTTP request', async () => {
  const { ui, requests } = harness()
  fillFlight(ui, '2099-02-30')
  await ui.submit()
  assert.equal(requests.length, 0)
})

after(() => {
  const destination = path.join(frontend, '../../.local/auth-history-frontend-test-results.json')
  fs.mkdirSync(path.dirname(destination), { recursive: true })
  fs.writeFileSync(destination, JSON.stringify({ tests_run: records.length,
    successful: records.every(record => record.status === 'passed'),
    scope: 'Actual compiled App.vue logic and Vue reactivity, controlled HTTP timing; no browser or production database.',
    records }, null, 2))
})
