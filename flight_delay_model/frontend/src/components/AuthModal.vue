<script setup>
/**
 * Sign-in / create-account modal. Registering also signs the user in.
 * Accounts are optional — every prediction feature works signed out;
 * an account only keeps the prediction history.
 */
import { ref, watch } from 'vue'

const props = defineProps({ initialMode: { type: String, default: 'signin' } })
const emit = defineEmits(['close', 'authenticated'])

const mode = ref(props.initialMode)
const username = ref('')
const password = ref('')
const error = ref('')
const busy = ref(false)

watch(mode, () => { error.value = '' })

async function submit() {
  error.value = ''
  if (!username.value.trim() || !password.value) {
    error.value = 'Please enter both a username and a password.'
    return
  }
  busy.value = true
  try {
    const url = mode.value === 'signin' ? '/api/auth/login' : '/api/auth/register'
    const resp = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: username.value.trim(), password: password.value }),
    })
    const data = await resp.json().catch(() => ({}))
    if (!resp.ok) { error.value = data.error || `Request failed (${resp.status})`; return }
    emit('authenticated', data.user)
  } catch (e) {
    error.value = 'Network error: ' + e.message
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="overlay" @click.self="emit('close')">
    <div class="modal">
      <button class="close" aria-label="Close" @click="emit('close')">×</button>
      <div class="tabs">
        <button :class="{ on: mode === 'signin' }" @click="mode = 'signin'">Sign in</button>
        <button :class="{ on: mode === 'register' }" @click="mode = 'register'">Create an account</button>
      </div>

      <form v-if="mode === 'signin'" @submit.prevent="submit">
        <div class="field"><label>Username</label>
          <input v-model="username" autocomplete="username" /></div>
        <div class="field"><label>Password</label>
          <input v-model="password" type="password" autocomplete="current-password" /></div>
        <p v-if="error" class="error">{{ error }}</p>
        <button class="go" :disabled="busy">{{ busy ? 'Signing in…' : 'Sign in' }}</button>
        <p class="note">No account yet? Creating one only keeps your prediction history —
          every feature works without signing in.</p>
      </form>

      <form v-else @submit.prevent="submit">
        <div class="field"><label>Username</label>
          <input v-model="username" autocomplete="username" placeholder="3–32 characters" /></div>
        <div class="field"><label>Password</label>
          <input v-model="password" type="password" autocomplete="new-password" placeholder="At least 6 characters" /></div>
        <p v-if="error" class="error">{{ error }}</p>
        <button class="go" :disabled="busy">{{ busy ? 'Creating account…' : 'Create account' }}</button>
        <p class="note">Stored in a local SQLite database (password hashed with PBKDF2).</p>
      </form>
    </div>
  </div>
</template>

<style scoped>
.overlay {
  position: fixed; inset: 0; background: rgba(16, 24, 32, 0.45);
  display: flex; align-items: center; justify-content: center; z-index: 100; padding: 16px;
}
.modal {
  position: relative; width: 400px; max-width: 100%; background: #fff;
  border-radius: var(--radius); padding: 26px 28px 24px; box-shadow: var(--card-shadow);
}
.close {
  position: absolute; top: 10px; right: 12px; border: 0; background: none;
  font-size: 24px; line-height: 1; color: var(--ink-soft); cursor: pointer;
}
.tabs { display: flex; gap: 22px; border-bottom: 1px solid var(--hairline); margin-bottom: 18px; }
.tabs button {
  border: 0; background: none; padding: 8px 2px 10px; font-size: 15px; cursor: pointer;
  color: var(--ink-soft); border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tabs button.on { color: var(--green-brand); border-bottom-color: var(--green-brand); font-weight: 600; }
.field { display: flex; flex-direction: column; gap: 2px; margin-bottom: 14px; }
label { font-size: 12px; color: var(--ink-body); }
input {
  border: 0; border-bottom: 1.5px solid #b9b7af; border-radius: 0; background: transparent;
  padding: 8px 2px; font-size: 15px; color: var(--ink); font-family: inherit;
}
input:focus { outline: none; border-bottom-color: var(--green-brand); }
.go {
  width: 100%; margin-top: 6px; padding: 12px; background: var(--green-btn); color: #fff;
  border: 0; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer;
}
.go:hover { background: var(--green-btn-hover); }
.go:disabled { opacity: 0.6; cursor: wait; }
.error { color: #b23a3a; font-size: 13px; margin: 0 0 8px; }
.note { color: var(--ink-soft); font-size: 12.5px; margin: 14px 0 0; line-height: 1.5; }
</style>
