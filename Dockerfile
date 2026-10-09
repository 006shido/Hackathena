# Production Dockerfile for DeepTrace
FROM node:20-alpine AS builder

WORKDIR /app

# Copy root package.json and manifests
COPY package.json package-lock.json* ./
COPY server/package.json ./server/
COPY server/package-lock.json ./server/
COPY client/package.json ./client/
COPY client/package-lock.json ./client/

# Install dependencies
RUN npm ci && npm ci --prefix server && npm ci --prefix client

# Copy source files
COPY . .

# Build client and server
RUN npm run build
RUN npm prune --omit=dev --prefix server

# Production runtime stage
FROM node:20-alpine AS runner

WORKDIR /app
ENV NODE_ENV=production
ENV PORT=5001

COPY package.json ./
COPY --from=builder /app/server ./server
COPY --from=builder /app/client/dist ./client/dist

EXPOSE 5001

CMD ["npm", "start"]
