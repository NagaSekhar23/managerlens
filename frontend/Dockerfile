# syntax=docker/dockerfile:1

FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Baked into the client bundle at build time — this must be the URL the *browser*
# can reach, not an internal Docker service name. It is intentionally public: it's
# just the backend's address, not a secret.
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL

RUN npm run build

FROM node:22-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
# Next's standalone server.js binds to `process.env.HOSTNAME || '0.0.0.0'`. Docker
# auto-injects HOSTNAME=<container id> into every container, which the server then
# resolves and binds to instead of falling through to 0.0.0.0 — so the app is only
# reachable via the container's bridge IP, never via 127.0.0.1. That breaks the
# HEALTHCHECK below (it curls localhost from inside the container) even though the
# app is perfectly reachable from outside via the published port. Overriding
# HOSTNAME here restores the intended "listen on all interfaces" behavior.
ENV HOSTNAME=0.0.0.0

# node:22-slim already ships a non-root "node" user — reuse it instead of colliding
# with its UID 1000.
COPY --from=builder /app/public ./public
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
USER node

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/', r => process.exit(r.statusCode < 500 ? 0 : 1)).on('error', () => process.exit(1))"

CMD ["node", "server.js"]
