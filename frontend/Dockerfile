# Stage 1: Build
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
# Empty API URL → frontend uses relative paths, nginx proxies /api/ to backend
ENV VITE_AGENTBOOK_API_URL=""
RUN npm run build

# Stage 2: Serve
FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
# 127.0.0.1 (not localhost) — Alpine wget resolves localhost to ::1, nginx listens IPv4 only
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q --spider http://127.0.0.1/ || exit 1
