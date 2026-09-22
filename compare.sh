#!/bin/bash
# Runs one input through several models using the agent's own prompts, so the
# comparison reflects what the service actually sends. The stack must be up.
#
#   ./compare.sh                                  # defaults, English
#   MODE=grammar TEXT="teniamos pensado..." ./compare.sh qwen3:4b gemma3:4b
#
# Pull a model first: docker compose exec ollama ollama pull <tag>
set -euo pipefail

MODE="${MODE:-professional}"
TEXT="${TEXT:-we was hoping to get this done friday}"

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
  models=(gemma3:4b granite4:7b-a1b-h granite4:3b)
fi

docker compose exec -T agent python - "$MODE" "$TEXT" "${models[@]}" <<'PY'
import asyncio
import os
import sys

from ollama import AsyncClient

from app.agent import _INSTRUCTIONS, _RULES, _strip_preamble, _ms, Mode

mode, text, models = Mode(sys.argv[1]), sys.argv[2], sys.argv[3:]
system = f"{_INSTRUCTIONS[mode]} {_RULES}"

# Every model but the one the service runs is unloaded again afterwards, so a
# comparison does not leave gigabytes resident.
serving = os.environ.get("OLLAMA_MODEL", "")


async def run(client, model, **options):
    return await client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0.2, "num_predict": 1024, "num_ctx": 4096, **options},
    )


async def main():
    client = AsyncClient(host="http://ollama:11434", timeout=900)
    print(f"mode: {mode.value}\ninput: {text}\n")
    for model in models:
        try:
            await run(client, model, num_predict=1)  # prime the prompt prefix
            response = await run(client, model)
        except Exception as exc:
            print(f"{model}\n  FAILED: {exc}\n")
            continue

        if model != serving:
            await client.generate(model=model, prompt="", keep_alive=0)

        eval_ms = _ms(response.eval_duration)
        tokens = response.eval_count or 0
        print(f"{model}")
        print(f"  {_strip_preamble(response['message']['content'])}")
        print(
            f"  [{_ms(response.total_duration)} ms total, "
            f"{round(tokens / (eval_ms / 1000), 1) if eval_ms else 0} tok/s]\n"
        )


asyncio.run(main())
PY
