# ollama-text-improving-agent

A small self-hosted service that rewrites text. Send it a paragraph with `curl`, get back a cleaner version: grammar fixed, padding cut, tone adjusted. A FastAPI app in front of a CPU-only Ollama container.

This is a proof of concept. It works and it is useful, but the point of building it was to get hands-on with the Ollama Python SDK, with running a local model in Docker, and with how much of the output quality comes down to prompt design.

> [!NOTE]
> "Agent" is the name of the service, not a claim that this is agentic. Each request is a single stateless model call with a fixed system prompt: no tool calling, no loop, no memory, no autonomy.

___
## Run it

```bash
docker compose up --build
```

The first start pulls a few gigabytes of model and blocks for several minutes. Later starts reuse the volume.

```bash
export AGENT="http://192.168.120.252:8000"
export AGENT_API_KEY="YOUR_API_KEY_HERE" # Get the key from the first deploy or from /opt/ollama-agent/.env on the server

curl -s "${AGENT}/improve" \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${AGENT_API_KEY}" \
  -d '{"text":"we was hoping to get this done friday","mode":"professional"}' | jq -r .improved
```

Pipe a file through it. `jq -Rs .` turns the file into a valid JSON string:

```bash
jq -Rs '{text: ., mode: "concise"}' < draft.md \
  | curl -s "${AGENT}/improve" -H 'Content-Type: application/json' \
      -H "X-API-Key: ${AGENT_API_KEY}" -d @- | jq -r .improved
```

Endpoints: `POST /improve`, `GET /health`, `GET /ready`, `GET /docs`. Only `/improve` needs the key, and only when `AGENT_API_KEY` is set. Errors: `401` bad key, `413` over `MAX_INPUT_CHARS`, `422` bad input, `502` Ollama unreachable, `504` too slow.

> [!TIP]
> This is a FastAPI service, therefore you can visit `/docs` in your browser to explore the API interactively.

___
## Modes

All four share one rule: clear and smooth to read, cut unnecessary words, keep it natural. `temperature` defaults to `0.2`.

| Mode | Does |
|---|---|
| `professional` | Default. Fixes errors, cuts padding, lifts the wording. |
| `grammar` | Corrections only: grammar, spelling, punctuation, accents, capitalisation. |
| `clarity` | Untangles awkward sentences, prefers plain words and the active voice. |
| `concise` | As short as it can be said, every fact kept. |

Input language is preserved and never translated, subject to the model supporting it.

___
## Deploy

```bash
ansible-galaxy collection install -r ansible/requirements.yml
./ansible/deploy.sh
```

Deploys to `server_ip` from `ansible/vars.yml`, which also holds the model and tuning values. The playbook creates `/opt/ollama-agent`, writes a root-only `.env` there, starts the stack, opens the port in `ufw`, and waits for `/ready`. The API key is generated on first deploy and reused on every rerun, so redeploying never invalidates existing clients. The target host needs Docker and the Compose v2 plugin already installed.

___
## Notes

- **Model is an env var**, never a code change. Set `OLLAMA_MODEL` and restart. `./compare.sh` runs one input through several models using the service's own prompts, so switching is a measurement rather than a guess.
- **Latency is model loading, not generation.** Ollama releases a model 5 minutes after the last request, so an occasional request waits for it to load again: measured at 11 s to load plus 4 s for a cold prompt, against 1.4 s fully warm. `REQUEST_TIMEOUT` is 300s to cover that. To keep it resident instead, set `OLLAMA_KEEP_ALIVE` on the `ollama` service, at the cost of holding several gigabytes of RAM. A model pinned by an earlier run stays pinned until the `ollama` container restarts.
- **Every response carries `timings`** with Ollama's own `load_ms`, `prompt_eval_ms` and `eval_ms`, so slowness is diagnosable rather than guessable.
- **Small models are weaker outside English.** The default fixes unambiguous Spanish accents but will not correct `hable` to `hablé`, since telling those apart needs the author's intent. Read non-English output before sending it.
- **Avoid reasoning models.** `qwen3:4b` leaked its reasoning into the reply even with `think=False`.

___
## Author

- [Jhonnathan Manzur](https://jmanzur.com)