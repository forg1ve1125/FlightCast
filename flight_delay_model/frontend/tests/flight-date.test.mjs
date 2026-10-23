/** Real compiled picker logic, with a controlled local clock and picker adapter.
 * Actual disabled-day interaction is additionally checked in the browser.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import vm from 'node:vm'
import { createRequire } from 'node:module'
import { test } from 'node:test'
import { localToday, isSelectableFlightDate } from '../src/flightDate.js'

const require = createRequire(new URL('../package.json', import.meta.url))
const vue = require('vue')
const { parse, compileScript } = require('@vue/compiler-sfc')
const descriptor = parse(fs.readFileSync(new URL('../src/components/DateTimeInput.vue', import.meta.url), 'utf8')).descriptor
const compiled = compileScript(descriptor, { id: 'flight-date-regression' }).content
  .replace(/^import\s+[\s\S]*?\s+from\s+['"][^'"]+['"];?\s*$/gm, '')
  .replace('export default', 'const component =')

function harness(mode = 'date', initial = '') {
  let now = new Date(2026, 9, 23, 23, 59, 59)
  const mounted = [], unmounted = [], events = new Map()
  let options, instance, interval, destroyed = false, cleared = false
  const input = { value: initial }
  const context = {
    ...vue, _mergeModels: vue.mergeModels, _useModel: props => vue.ref(props.modelValue),
    onMounted: fn => mounted.push(fn), onBeforeUnmount: fn => unmounted.push(fn),
    localToday: () => localToday(now),
    isSelectableFlightDate: value => isSelectableFlightDate(value, now),
    window: {
      addEventListener: (key, fn) => events.set(`window:${key}`, fn),
      removeEventListener: key => events.delete(`window:${key}`),
      setInterval: fn => { interval = fn; return 1 },
      clearInterval: () => { cleared = true },
    },
    document: {
      addEventListener: (key, fn) => events.set(`document:${key}`, fn),
      removeEventListener: key => events.delete(`document:${key}`),
    },
    flatpickr: (_, config) => {
      options = config
      instance = {
        input, config: { ...config }, currentYearElement: { setAttribute() {} },
        calendarContainer: { classList: { add() {} }, setAttribute() {} },
        set: (key, value) => { instance.config[key] = value },
        clear: () => { input.value = '' },
        setDate: value => { input.value = value },
        destroy: () => { destroyed = true },
      }
      config.onReady([], '', instance)
      return instance
    },
  }
  const ui = vm.runInNewContext(compiled + '\ncomponent.setup(__props, { expose() {} })',
    { ...context, __props: { id: 'test-date', mode, modelValue: initial } })
  mounted.forEach(fn => fn())
  return { ui, options, instance, events, tick: () => interval(),
    setNow: date => { now = date },
    close: () => unmounted.forEach(fn => fn()),
    status: () => ({ destroyed, cleared }) }
}

test('Local date formatting uses local midnight rather than UTC', () => {
  const previous = process.env.TZ
  try {
    process.env.TZ = 'Asia/Shanghai'
    assert.equal(localToday(new Date('2026-10-22T16:01:00Z')), '2026-10-23')
  } finally {
    if (previous === undefined) delete process.env.TZ
    else process.env.TZ = previous
  }
})

test('Today and future are accepted, yesterday is rejected', () => {
  const now = new Date(2026, 9, 23, 0, 1)
  assert.ok(isSelectableFlightDate('2026-10-23', now))
  assert.ok(isSelectableFlightDate('2027-05-12', now))
  assert.equal(isSelectableFlightDate('2026-10-22', now), false)
})

test('Impossible dates, non-ISO formats, and non-text values are rejected', () => {
  const now = new Date(2026, 9, 23)
  for (const value of ['2027-02-29', '2028-02-30', '2027-13-01', '2027-1-01', '', null, 20270101]) {
    assert.equal(isSelectableFlightDate(value, now), false)
  }
  assert.ok(isSelectableFlightDate('2028-02-29', now))
})

test('Date picker receives today as its minimum and disallows text entry', () => {
  const h = harness()
  assert.equal(h.options.minDate, '2026-10-23')
  assert.equal(h.options.allowInput, false)
  assert.equal(h.instance.currentYearElement.readOnly, true)
  h.close()
})

test('Opening the picker refreshes its minimum after midnight and clears yesterday', () => {
  const h = harness('date', '2026-10-23')
  h.setNow(new Date(2026, 9, 24, 0, 0, 1))
  h.options.onOpen([], '2026-10-23', h.instance)
  assert.equal(h.instance.config.minDate, '2026-10-24')
  assert.equal(h.ui.value.value, '')
  h.close()
})

test('An already-open picker refreshes across midnight without reopening', () => {
  const h = harness('date', '2026-10-23')
  h.setNow(new Date(2026, 9, 24, 0, 0, 0))
  h.tick()
  assert.equal(h.instance.config.minDate, '2026-10-24')
  assert.equal(h.ui.value.value, '')
  h.close()
})

test('Returning from sleep or a system clock change refreshes the date limit', () => {
  const h = harness()
  h.setNow(new Date(2026, 10, 1))
  h.events.get('window:focus')()
  assert.equal(h.instance.config.minDate, '2026-11-01')
  h.setNow(new Date(2026, 9, 23))
  h.events.get('document:visibilitychange')()
  assert.equal(h.instance.config.minDate, '2026-10-23')
  h.close()
})

test('Past initial or programmatically supplied values are removed', async () => {
  const h = harness('date', '2021-12-30')
  assert.equal(h.options.defaultDate, undefined)
  assert.equal(h.ui.value.value, '')
  h.ui.value.value = '2020-01-01'
  await vue.nextTick()
  assert.equal(h.ui.value.value, '')
  h.close()
})

test('Time inputs keep their time-only configuration and value', () => {
  const h = harness('time', '22:45')
  assert.equal(h.options.minDate, undefined)
  assert.equal(h.options.noCalendar, true)
  assert.equal(h.ui.value.value, '22:45')
  assert.equal(h.events.size, 0)
  h.close()
})

test('Unmounting removes timers, listeners and the picker', () => {
  const h = harness()
  h.close()
  assert.equal(h.events.size, 0)
  assert.deepEqual(h.status(), { destroyed: true, cleared: true })
})
