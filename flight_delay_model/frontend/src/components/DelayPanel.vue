<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: { type: String, default: 'Departure' },
  verdict: Object,
  scenario: Object,
})

const OUTCOMES = [
  {
    label: 'On time or early',
    value: 'On time',
    unit: '',
    icon: '✅',
    description: 'Departure is on schedule or earlier than scheduled.',
  },
  {
    label: '1–15 min late',
    value: '1–15',
    unit: 'min',
    icon: '🕒',
    description: 'Departure is between 1 and 15 minutes later than scheduled.',
  },
  {
    label: '16–30 min late',
    value: '16–30',
    unit: 'min',
    icon: '⏳',
    description: 'Departure is between 16 and 30 minutes later than scheduled.',
  },
  {
    label: '31–60 min late',
    value: '31–60',
    unit: 'min',
    icon: '🕓',
    description: 'Departure is between 31 and 60 minutes later than scheduled.',
  },
  {
    label: 'Over 60 min late',
    value: 'Over 60',
    unit: 'min',
    icon: '⚠️',
    description: 'Departure is more than 60 minutes later than scheduled.',
  },
]

const outcomes = computed(() =>
  (props.scenario?.scenarios || [])
    .map(item => ({
      ...item,
      bucket: Number(item.bucket),
      ...OUTCOMES[Number(item.bucket)],
    }))
    .sort((a, b) => a.bucket - b.bucket),
)

const leading = computed(() =>
  outcomes.value.reduce(
    (best, item) =>
      !best || item.probability > best.probability ? item : best,
    null,
  ),
)

const delayProbability = computed(() => {
  const onTime = outcomes.value.find(item => item.bucket === 0)

  return onTime
    ? 1 - onTime.probability
    : props.verdict?.p_delay ?? null
})

const expectedDelay = computed(() =>
  props.scenario?.expected_delay_min_from_buckets ?? null,
)

function percent(value) {
  return value == null
    ? '—'
    : (Number(value) * 100).toFixed(1) + '%'
}

function minutes(value) {
  if (value == null) return '—'

  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 1,
  }).format(Number(value)) + ' min'
}
</script>

<template>
  <article v-if="leading" class="departure-panel">
    <header class="forecast-hero">

      <h3>{{ title }}</h3>

      <div class="hero-main">
        <div class="main-outcome">
          <p class="hero-label">Most likely outcome</p>

          <div class="outcome-value">
            <strong>{{ leading.value }}</strong>
            <span v-if="leading.unit">{{ leading.unit }} late</span>
          </div>

        </div>

        <div class="hero-probability">
          <strong>{{ percent(leading.probability) }}</strong>
          <span>Probability of this outcome</span>
        </div>
      </div>

    </header>

    <div class="metrics">
      <div class="metric">
        <span>Chance of any delay</span>
        <strong>{{ percent(delayProbability) }}</strong>
        <small>Includes all late departures</small>
      </div>

      <div class="metric">
        <span>Expected delay</span>
        <strong>{{ minutes(expectedDelay) }}</strong>
        <small>Weighted estimate across all outcomes</small>
      </div>
    </div>

    <section class="outcome-section">
      <div class="section-heading">
        <h4>All possible outcomes</h4>
        <p>Select a row for details.</p>
      </div>

      <div class="outcome-list">
        <details
          v-for="item in outcomes"
          :key="item.bucket"
          class="outcome-row"
          :class="{ featured: item.bucket === leading.bucket }"
        >
          <summary>
            <span class="outcome-icon" aria-hidden="true">
              {{ item.icon }}
            </span>

            <span class="outcome-name">
              <span>{{ item.label }}</span>

              <small v-if="item.bucket === leading.bucket">
                Most likely
              </small>
            </span>

            <strong class="row-probability">
              {{ percent(item.probability) }}
            </strong>

            <span class="expand-icon" aria-hidden="true">+</span>
          </summary>

          <div class="outcome-description">
            <p>{{ item.description }}</p>

            <p v-if="item.bucket !== 0 && item.typical_minutes != null">
              Typical delay within this range:
              <strong>{{ minutes(item.typical_minutes) }}</strong>.
            </p>
          </div>
        </details>
      </div>

      <p class="distribution-note">
        Probabilities cover all five possible outcomes.
      </p>
    </section>
  </article>
</template>

<style scoped>
.departure-panel {
  min-width: 0;
  overflow: hidden;
  border: 1px solid #d9e2ef;
  border-radius: 12px;
  background: #fff;
  color: #172b45;
  font-variant-numeric: tabular-nums;
}

.forecast-hero { padding: 14px 16px; background: #142d52; color: #fff; }
.forecast-hero h3 { margin: 0 0 8px; color: #fff; font-size: 16px; line-height: 1.3; font-weight: 650; }
.hero-main { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.main-outcome { min-width: 0; }
.hero-label { margin: 0 0 4px; color: #c5d5ed; font-size: 10px; line-height: 1.3; }
.outcome-value { display: flex; align-items: baseline; flex-wrap: wrap; gap: 5px; line-height: 1.15; }
.outcome-value strong { font-size: 27px; font-weight: 700; letter-spacing: -.035em; }
.outcome-value > span { color: #dce6f4; font-size: 11px; }

.hero-probability {
  display: flex;
  flex: 0 0 auto;
  flex-direction: column;
  max-width: 135px;
  gap: 4px;
  padding-left: 14px;
  border-left: 1px solid #476083;
}

.hero-probability strong { color: #ffda85; font-size: 27px; font-weight: 700; line-height: 1.1; letter-spacing: -.03em; }
.hero-probability > span { color: #dce6f4; font-size: 9px; line-height: 1.4; }
.metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 10px 10px 0; }
.metric { display: flex; flex-direction: column; gap: 2px; padding: 9px 10px; border-radius: 9px; background: #f1f5fa; }
.metric > span { color: #485b73; font-size: 10px; }
.metric strong { color: #142d52; font-size: 19px; font-weight: 650; line-height: 1.25; }
.metric small { display: none; }
.outcome-section { padding: 12px 10px; }
.section-heading { margin-bottom: 8px; }
.section-heading h4 { margin: 0; font-size: 13px; font-weight: 650; line-height: 1.4; }
.section-heading p { margin: 3px 0 0; color: #5d6f86; font-size: 10px; }
.outcome-list { display: flex; flex-direction: column; gap: 4px; }
.outcome-row { border: 1px solid #dce4ef; border-radius: 9px; background: #fff; }
.outcome-row.featured { border-color: #9bb9df; background: #f0f5fc; }

.outcome-row summary {
  display: flex;
  align-items: center;
  min-height: 36px;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 8px;
  list-style: none;
  cursor: pointer;
}

.outcome-row summary::-webkit-details-marker { display: none; }
.outcome-row summary::marker { content: ''; }
.outcome-row summary:hover { background: #eaf0f8; }
.outcome-row summary:focus-visible { outline: 2px solid #245a99; outline-offset: 2px; }
.outcome-icon { flex: 0 0 18px; font-size: 15px; text-align: center; }
.outcome-name { display: flex; flex: 1; flex-direction: column; min-width: 0; gap: 1px; font-size: 11px; font-weight: 600; line-height: 1.4; }
.outcome-name small { color: #245a99; font-size: 9px; font-weight: 600; }
.row-probability { flex: 0 0 auto; color: #173f73; font-size: 12px; }
.expand-icon { flex: 0 0 12px; color: #5d6f86; font-size: 18px; line-height: 1; text-align: center; }
.outcome-row[open] .expand-icon { transform: rotate(45deg); }
.outcome-description { padding: 0 12px 10px 36px; color: #485b73; font-size: 11px; }
.outcome-description p { margin: 0 0 5px; }
.outcome-description p:last-child { margin-bottom: 0; }
.distribution-note { margin: 9px 0 0; color: #5d6f86; font-size: 10px; line-height: 1.4; }

@media (max-width: 380px) {
  .outcome-value strong { font-size: 24px; }
  .hero-probability { max-width: 115px; padding-left: 10px; }
  .hero-probability strong { font-size: 24px; }
}
</style>
