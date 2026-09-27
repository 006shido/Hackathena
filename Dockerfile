# Production Dockerfile for DeepTrace
FROM node:20-alpine AS builder

WORKDIR /app

# Copy root package.json and manifests
COPY package.json package-lock.json* ./
COPY server/package.json ./server/
COPY client/package.json ./client/

# Install dependencies
RUN npm run install:all

# Copy source files
COPY . .

# Build client and server
RUN npm run build

# Production runtime stage
FROM node:20-alpine AS runner

WORKDIR /app
ENV NODE_ENV=production
ENV PORT=5001

COPY package.json ./
COPY --from=builder /app/server ./server
COPY --from=builder /app/client/dist ./client/dist
COPY --from=builder /app/node_modules ./node_modules

EXPOSE 5001

CMD ["npm", "start"]
