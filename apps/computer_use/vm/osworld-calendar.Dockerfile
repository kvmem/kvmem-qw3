# Upstream calendar image with its missing lib/ Tailwind scan repaired.
# The fixture stores event colors in src/lib/calendar-data.ts. Without this
# scan, green events render white text on a white background in production.
FROM node:22-alpine AS base
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --fund=false

FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN node -e 'const fs=require("node:fs"); const p="tailwind.config.ts"; const s=fs.readFileSync(p,"utf8"); const marker="  content: ["; if(s.split(marker).length!==2 || s.includes("./src/lib/")) throw Error("Unexpected upstream Tailwind configuration"); fs.writeFileSync(p,s.replace(marker,marker+"\n    \"./src/lib/**/*.{js,ts,jsx,tsx,mdx}\","));'
# Day is already exposed by Header and rendered by WeekView, but the upstream
# action endpoint rejects it, making the UI immediately revert to Month.
RUN node -e 'const fs=require("node:fs"); const p="src/app/api/calendar/actions/route.ts"; const s=fs.readFileSync(p,"utf8"); const marker="[\"month\", \"week\", \"agenda\"].includes(input.payload as string)"; if(s.split(marker).length!==2) throw Error("Unexpected upstream calendar view validation"); fs.writeFileSync(p,s.replace(marker,"[\"month\", \"week\", \"day\", \"agenda\"].includes(input.payload as string)"));'
RUN npm run build

FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs
COPY --from=builder /app/public ./public
RUN mkdir .next && chown nextjs:nodejs .next
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
RUN mkdir -p uploads && chown nextjs:nodejs uploads
USER nextjs
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
CMD ["node", "server.js"]
