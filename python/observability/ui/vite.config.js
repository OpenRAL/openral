import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the robot API to a running `openral dashboard`.
// ROBOT=http://<host>:4318 npm run dev points it at another robot.
const robot = process.env.ROBOT || "http://127.0.0.1:4318";

export default defineConfig({
  // Relative asset URLs: the same build is served at the robot's `/` and at
  // www.openral.com/robot-ui/.
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false },
  server: {
    proxy: {
      "/api": robot,
      "/viz": robot,
      "/ws": { target: robot.replace(/^http/, "ws"), ws: true },
    },
  },
});
