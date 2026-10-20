<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import flatpickr from 'flatpickr'

const props = defineProps({
  id: { type: String, required: true },
  mode: { type: String, default: 'date' },
  required: Boolean,
})

const value = defineModel({ type: String, default: '' })

const input = ref(null)
const opened = ref(false)
let picker = null

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
    monthSelectorType: 'static',
    position: isTime ? 'auto' : positionDatePicker,

    enableTime: isTime,
    noCalendar: isTime,
    time_24hr: true,
    minuteIncrement: 1,

    dateFormat: isTime ? 'H:i' : 'Y-m-d',
    defaultDate: value.value || undefined,

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
    },

    onChange: (_, selectedValue) => {
      value.value = selectedValue
    },

    onOpen: () => {
      opened.value = true
    },

    onClose: (_, selectedValue) => {
      opened.value = false
      value.value = selectedValue
    },
  })

  if (!isTime) {
    window.addEventListener('scroll', repositionDatePicker, {
      capture: true,
      passive: true,
    })
  }
})

watch(value, (nextValue) => {
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
    @keydown.enter.prevent="openPicker"
    @keydown.space.prevent="openPicker"
  />
</template>
