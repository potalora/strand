import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produce a self-contained server bundle (.next/standalone/server.js) for
  // the Docker runtime image. See frontend/Dockerfile.
  output: "standalone",
};

export default nextConfig;
