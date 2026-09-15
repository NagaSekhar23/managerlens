import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produces a minimal .next/standalone server bundle for the Docker image —
  // no need to ship node_modules or the full source tree into the runtime image.
  output: "standalone",
};

export default nextConfig;
