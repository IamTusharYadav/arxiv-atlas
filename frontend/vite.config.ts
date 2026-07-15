import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Dev-only: same-origin /api calls forward to the deployed stack, so local dev needs no
    // CORS story and no env file. Production builds set VITE_API_URL instead (see src/api.ts).
    proxy: {
      "/api": {
        target: "https://99a0zbyk70.execute-api.us-east-1.amazonaws.com",
        changeOrigin: true,
      },
    },
  },
});
