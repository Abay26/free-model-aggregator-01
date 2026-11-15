# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Abay26/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Abay26/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten simulated free-tier model providers, routing each request to the best free model able to serve it, failing over when a provider refuses, and reporting the catalog it read.

v2: rating-driven routing from the brief's sealed quality table (v1 routed blind and scored 10), failover in rating order with per-provider refusal accounting, and all three upstream wires — OpenAI, Anthropic and Gemini — so the best model is reachable whatever wire it speaks. A streaming OpenAI request is sent upstream with `stream_options.include_usage`, or the provider sends no usage chunk and the client's usage check fails.

## Run it

```sh
node main.ts --list
MAKEFUN_ENV_URL=http://127.0.0.1:18081 node main.ts
```

Node 22 standard library only; nothing is installed.

## Layout

- `main.ts` — the server: brief, catalog probe, rating-based routing, failover, register and poll.
- `lib/providers.ts` — the brief's three documents and the catalog probe.
- `lib/wire.ts` — the client wires and the OpenAI model listing.
- `lib/upstream.ts` — the Gemini and Anthropic upstream wires, and turning any provider's answer into text plus usage.

---

<sub>Entered on [make.fun](https://make.fun/?ref=qw56w9dq&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
