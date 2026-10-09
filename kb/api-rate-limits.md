---
doc_id: api-rate-limits
title: API Rate Limits
effective_date: 2026-01-10
status: current
---
# API Rate Limits

The Orbitask REST API (https://api.orbitask.example/v2) is available on all plans. Limits apply per workspace, not per user or per API key.

## Limits by plan

| Plan | Requests per minute | Burst (requests per second) | Requests per day | Webhooks per workspace |
|---|---|---|---|---|
| Starter | 60 | 5 | 10,000 | 5 |
| Team | 300 | 20 | 100,000 | 25 |
| Business | 1,000 | 50 | 500,000 | 100 |
| Enterprise | 5,000 | 200 | No daily limit | 500 |

## When you hit a limit

The API answers with HTTP 429 Too Many Requests and a Retry-After header that gives the number of seconds to wait. Requests rejected for rate limiting do not count toward the daily limit.

## Other limits

- Maximum request body size: 10 MB.
- List endpoints return at most 200 items per page; use the cursor parameter to get the next page.
- API keys expire after 365 days and can be rotated at any time under Settings > API.
- Webhook deliveries are retried up to 5 times over 24 hours if your endpoint does not reply with a 2xx status.
