import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // Dev-only: same-origin /api calls forward to the deployed stack, so local dev needs no
  // CORS story and no env file. Production builds set VITE_API_URL instead (see src/api.ts).
  // To develop against a local API (scripts/dev_api.py), put
  // VITE_API_PROXY=http://localhost:8000 in frontend/.env.local (gitignored).
  const env = loadEnv(mode, ".", "VITE_");
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": {
          target: env.VITE_API_PROXY || "https://99a0zbyk70.execute-api.us-east-1.amazonaws.com",
          changeOrigin: true,
        },
      },
    },
  };
});
