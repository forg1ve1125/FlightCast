<script setup>
/**
 * One result panel (departure / arrival): verdict badge + scenario probability
 * bars + expected delay. Data comes from POST /api/predict (or a stored history
 * item): scenarios.<seg> with items {bucket, label, probability, typical_minutes}.
 */
const props = defineProps({
  title: String,          // "Departure" / "Arrival"
  verdict: Object,        // {level, phrase, p_delay, expected_delay_min}
  scenario: Object,       // {scenarios: [...], expected_delay_min_from_buckets}
})

const LEVEL_CLASS = { good: 'lv-good', notice: 'lv-notice', warn: 'lv-warn', alert: 'lv-alert' }
// by bucket severity: on-time green -> delay reds
const BAR_COLOR = ['#00645a', '#6fa287', '#c9a227', '#c0763c', '#b23a3a']

function pct(p) { return (p * 100).toFixed(1) + '%' }
function barWidth(p) { return Math.max(p * 100, 2) + '%' }
function minutes(s) {
  return s.bucket === 0 ? ''
    : (s.typical_minutes != null ? `~${Math.round(s.typical_minutes)} min` : '')
}
</script>

<template>
  <div class="panel" v-if="scenario">
    <div class="head">
      <h3>{{ title }}</h3>
      <span class="badge" :class="LEVEL_CLASS[verdict?.level] || ''">{{ verdict?.phrase || '' }}</span>
    </div>
    <div class="bars">
      <div v-for="s in scenario.scenarios" :key="s.bucket" class="row">
        <div class="label">{{ s.label }}<span v-if="minutes(s)" class="min">{{ minutes(s) }}</span></div>
        <div class="track"><div class="fill" :style="{ width: barWidth(s.probability), background: BAR_COLOR[s.bucket] || BAR_COLOR[4] }"></div></div>
        <div class="pct">{{ pct(s.probability) }}</div>
      </div>
    </div>
    <div class="foot">
      Expected delay <b>{{ scenario.expected_delay_min_from_buckets }}</b> min
      <span v-if="verdict?.p_delay != null" class="muted">· chance of any delay {{ pct(verdict.p_delay) }}</span>
    </div>
  </div>
</template>

<style scoped>
.panel {
  background: #fff; border: 1px solid var(--hairline); border-radius: var(--radius);
  padding: 18px 20px;
}
.head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 14px; }
h3 { margin: 0; font-size: 16px; color: var(--green-ink); letter-spacing: 0.01em; }
.badge { font-size: 13px; font-weight: 600; padding: 5px 12px; border-radius: 999px; white-space: nowrap; }
.lv-good { background: #e2efeb; color: #00584e; }
.lv-notice { background: #f6efd9; color: #7c6210; }
.lv-warn { background: #f7e8d8; color: #93521c; }
.lv-alert { background: #f6dfdf; color: #8f2f2f; }
.bars { display: flex; flex-direction: column; gap: 9px; }
.row { display: grid; grid-template-columns: 190px 1fr 54px; align-items: center; gap: 10px; }
.label { font-size: 13px; color: var(--ink); text-align: right; }
.min { color: var(--ink-soft); font-size: 12px; margin-left: 4px; }
.track { background: var(--panel-2); border-radius: 4px; height: 14px; overflow: hidden; }
.fill { height: 100%; border-radius: 4px; transition: width .4s; }
.pct { font-size: 13px; font-variant-numeric: tabular-nums; color: var(--ink-body); }
.foot { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--hairline); font-size: 13px; color: var(--ink-body); }
.muted { color: var(--ink-soft); }
</style>
