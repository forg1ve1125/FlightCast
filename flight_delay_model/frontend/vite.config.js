import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发模式: npm run dev (5173) 代理到 Flask(5000)；生产模式: npm run build 后由 Flask 托管 dist
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:5000' },
  },
})
