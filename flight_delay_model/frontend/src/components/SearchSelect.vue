<script setup>
/**
 * Searchable dropdown: type a query -> GET searchUrl -> pick from candidates.
 * Airlines: IATA code (DL/dl) or partial name (Delta/DELTA).
 * Airports: code or city name (new york -> JFK/LGA).
 * Raw exact input also works (chosen on blur/Enter when it matches a candidate).
 */
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'

const props = defineProps({
  label: String,
  placeholder: String,
  searchUrl: String,          // e.g. '/api/airlines'
  valueKey: { type: String, default: 'name' },   // field submitted to the backend
  extraKeys: { type: Array, default: () => [] }, // extra display fields
  modelValue: String,
  required: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'select'])
const query = ref(props.modelValue || '')
const options = ref([])
const open = ref(false)
const highlight = ref(-1)
const loading = ref(false)
const root = ref(null)
const exactMatch = computed(() =>
  options.value.some(o => String(o[props.valueKey]).toLowerCase() === query.value.trim().toLowerCase()))

let debounceTimer = null

async function search() {
  loading.value = true
  try {
    const q = encodeURIComponent(query.value.trim())
    const resp = await fetch(`${props.searchUrl}?q=${q}`)
    options.value = await resp.json()
    highlight.value = options.value.length ? 0 : -1
  } finally {
    loading.value = false
  }
}

function onInput() {
  emit('update:modelValue', query.value)
  open.value = true
  clearTimeout(debounceTimer)
  // empty input: load the default list (the airlines endpoint returns all without q)
  if (query.value.trim().length === 0) debounceTimer = setTimeout(search, 120)
  else debounceTimer = setTimeout(search, 200)
}

function choose(opt) {
  query.value = opt[props.valueKey]
  emit('update:modelValue', query.value)
  emit('select', opt)
  open.value = false
}

function onKeydown(e) {
  if (!open.value) {
    if (['ArrowDown', 'ArrowUp'].includes(e.key)) { open.value = true; search() }
    return
  }
  if (e.key === 'ArrowDown') { e.preventDefault(); highlight.value = Math.min(highlight.value + 1, options.value.length - 1) }
  else if (e.key === 'ArrowUp') { e.preventDefault(); highlight.value = Math.max(highlight.value - 1, 0) }
  else if (e.key === 'Enter') {
    e.preventDefault()
    if (highlight.value >= 0 && options.value[highlight.value]) choose(options.value[highlight.value])
    else if (exactMatch.value) choose(options.value.find(o => String(o[props.valueKey]).toLowerCase() === query.value.trim().toLowerCase()))
  } else if (e.key === 'Escape') open.value = false
}

function onDocClick(e) {
  if (root.value && !root.value.contains(e.target)) open.value = false
}
onMounted(() => {
  document.addEventListener('click', onDocClick)
  if (props.searchUrl.includes('airlines')) search() // airlines load in full so the list opens ready
})
onBeforeUnmount(() => document.removeEventListener('click', onDocClick))
</script>

<template>
  <div class="field" ref="root">
    <label>{{ label }}<span v-if="required" class="req">*</span></label>
    <input v-model="query" :placeholder="placeholder" @input="onInput" @focus="open = true"
           @keydown="onKeydown" autocomplete="off" />
    <ul v-if="open && options.length" class="opts">
      <li v-for="(opt, i) in options" :key="i" :class="{ hl: i === highlight }"
          @mousedown.prevent="choose(opt)" @mousemove="highlight = i">
        <slot :opt="opt">
          <span class="opt-code">{{ opt[valueKey] }}</span>
          <span v-for="k in extraKeys" :key="k" class="opt-name">{{ opt[k] }}</span>
        </slot>
      </li>
    </ul>
    <div v-else-if="open && loading" class="opts hint">Searching…</div>
  </div>
</template>

<style scoped>
.field { position: relative; display: flex; flex-direction: column; gap: 2px; min-width: 0; }
label { font-size: 12px; color: var(--ink-body); letter-spacing: 0.02em; }
.req { color: #b23a3a; margin-left: 2px; }
input {
  border: 0; border-bottom: 1.5px solid #b9b7af; border-radius: 0; background: transparent;
  padding: 8px 2px; font-size: 15px; color: var(--ink); font-family: inherit;
}
input:focus { outline: none; border-bottom-color: var(--green-brand); }
input::placeholder { color: #a8a7a0; }
.opts {
  position: absolute; top: 100%; left: 0; right: 0; z-index: 30; background: #fff;
  border: 1px solid var(--hairline); border-radius: 10px; max-height: 248px; overflow-y: auto;
  margin: 2px 0 0; padding: 6px 0; list-style: none; box-shadow: var(--card-shadow);
}
.opts li {
  padding: 8px 14px; display: flex; align-items: baseline; gap: 10px;
  cursor: pointer; font-size: 14px; white-space: nowrap;
}
.opts li.hl, .opts li:hover { background: var(--panel); }
.opt-code { font-weight: 600; flex: 0 0 auto; }
.opt-name { color: var(--ink-soft); font-size: 13px; min-width: 0; flex: 1 1 auto;
            overflow: hidden; text-overflow: ellipsis; }
.hint { padding: 10px 14px; color: var(--ink-soft); }
</style>
