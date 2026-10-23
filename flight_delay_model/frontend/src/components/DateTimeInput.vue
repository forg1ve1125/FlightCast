<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import flatpickr from 'flatpickr'
import { localToday, isSelectableFlightDate } from '../flightDate.js'

const props = defineProps({
  id: { type: String, required: true },
  mode: { type: String, default: 'date' },
  required: Boolean,
})

const value = defineModel({ type: String, default: '' })

const input = ref(null)
const opened = ref(false)
let picker = null
let dateTimer = null
let minimumDay = ''

function refreshDateLimit(instance = picker) {
  if (props.mode === 'time' || !instance) return
  const today = localToday()
  if (minimumDay !== today) {
    minimumDay = today
    instance.set('minDate', today)
  }
  if (value.value && !isSelectableFlightDate(value.value)) {
    instance.clear(false)
    value.value = ''
  }
}

function refreshOnFocus() {
  refreshDateLimit()
}

function positionDatePicker(instance, positionElement) {
  const popup = instance.calendarContainer
  const anchor = (positionElement || instance.input).getBoundingClientRect()
  const margin = 12
  const gap = 6
  const viewportWidth = document.documentElement.clientWidth
  const viewportHeight = document.documentElement.clientHeight
  const popupWidth = popup.offsetWidth
  const popupHeight = popup.offsetHeight
  const below = anchor.bottom + gap
  const preferredTop = below + popupHeight <= viewportHeight - margin
    ? below
    : anchor.top - popupHeight - gap

  // Include the popup's padding and keep it outside the page's scroll layout.
  popup.style.position = 'fixed'
  popup.style.top = Math.max(
    margin,
    Math.min(preferredTop, viewportHeight - popupHeight - margin),
  ) + 'px'
  popup.style.left = Math.max(
    margin,
    Math.min(anchor.left, viewportWidth - popupWidth - margin),
  ) + 'px'
  popup.style.right = 'auto'
}

function repositionDatePicker() {
  if (props.mode !== 'time' && picker?.isOpen) {
    positionDatePicker(picker)
  }
}

onMounted(() => {
  const isTime = props.mode === 'time'

  picker = flatpickr(input.value, {
    locale: 'en',
    disableMobile: true,
    allowInput: false,
    minDate: isTime ? undefined : localToday(),
    monthSelectorType: 'static',
    position: isTime ? 'auto' : positionDatePicker,

    enableTime: isTime,
    noCalendar: isTime,
    time_24hr: true,
    minuteIncrement: 1,

    dateFormat: isTime ? 'H:i' : 'Y-m-d',
    defaultDate: (isTime || isSelectableFlightDate(value.value)) ? value.value || undefined : undefined,

    onReady: (_, __, instance) => {
      const popup = instance.calendarContainer

      popup.classList.add('flight-picker')
      if (!isTime) popup.classList.add('flight-date-picker')
      popup.id = `${props.id}-popup`
      popup.lang = 'en'
      popup.setAttribute('role', 'dialog')
      popup.setAttribute(
        'aria-label',
        isTime ? 'Choose a time' : 'Choose a date',
      )
      if (!isTime) {
        instance.currentYearElement.readOnly = true
        instance.currentYearElement.setAttribute('inputmode', 'none')
        refreshDateLimit(instance)
      }
    },

    onChange: (_, selectedValue) => {
      value.value = selectedValue
    },

    onOpen: (_, __, instance) => {
      refreshDateLimit(instance)
      opened.value = true
    },

    onClose: (_, selectedValue) => {
      opened.value = false
      value.value = selectedValue
    },
  })

  if (!isTime) {
    // Refresh an already-open calendar across midnight, and after sleep/clock changes.
    dateTimer = window.setInterval(refreshOnFocus, 1000)
    window.addEventListener('focus', refreshOnFocus)
    document.addEventListener('visibilitychange', refreshOnFocus)
    window.addEventListener('scroll', repositionDatePicker, {
      capture: true,
      passive: true,
    })
  }
})

watch(value, (nextValue) => {
  if (props.mode !== 'time' && nextValue && !isSelectableFlightDate(nextValue)) {
    picker?.clear(false)
    value.value = ''
    return
  }
  if (!picker || nextValue === picker.input.value) return

  if (nextValue) {
    picker.setDate(
      nextValue,
      false,
      props.mode === 'time' ? 'H:i' : 'Y-m-d',
    )
  } else {
    picker.clear(false)
  }
})

function openPicker() {
  picker?.open()
}

onBeforeUnmount(() => {
  window.clearInterval(dateTimer)
  window.removeEventListener('focus', refreshOnFocus)
  document.removeEventListener('visibilitychange', refreshOnFocus)
  window.removeEventListener('scroll', repositionDatePicker, true)
  picker?.destroy()
})
</script>

<template>
  <input
    :id="id"
    ref="input"
    class="flight-picker-input"
    type="text"
    :value="value"
    :required="required"
    :placeholder="mode === 'time' ? 'Select time' : 'Select date'"
    :aria-controls="`${id}-popup`"
    :aria-expanded="opened"
    aria-haspopup="dialog"
    autocomplete="off"
    readonly
    inputmode="none"
    @beforeinput.prevent
    @paste.prevent
    @drop.prevent
    @keydown.enter.prevent="openPicker"
    @keydown.space.prevent="openPicker"
  />
</template>
