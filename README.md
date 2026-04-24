# Home Assistant OpenAI Codex

Custom Home Assistant integration for experimenting with the ChatGPT Codex
backend inside a local Home Assistant development environment.

## Features

- Device-code authorization against ChatGPT auth.
- Conversation entity backed by the Codex Responses endpoint.
- Home Assistant Assist LLM tool support for device/entity control.
- Runtime model catalog loading from the Codex `/models` API with Home
  Assistant-local caching and fallback.
- Options UI for model selection, reasoning budget, fast mode, and live web
  search when the selected model supports it.
- User-facing Codex API error details when the backend returns a readable
  message.

## Project Layout

- `custom_components/openai_codex/` - integration source code.
- `docker-compose.yml` - local Home Assistant test instance.
- `ha-config/` - local Home Assistant runtime config, ignored by Git.
- `.reference/home-assistant-core/` - upstream Home Assistant Core checkout for
  reference, ignored by Git.
- `.reference/codex-cli/` - upstream Codex CLI checkout for backend behavior
  reference, ignored by Git.
- `specs/` - local implementation notes and specifications.

## Development

Start the local test instance:

```sh
docker compose up -d
```

Open Home Assistant at http://localhost:8123.

Local test credentials:

- Username: `aaa4xu`
- Password: `12345678`
- Long-lived access token:
  `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmMTJjZDA2MDg0MzI0YjU0OTAyYWNhODNmMjIxMGE0MyIsImlhdCI6MTc3Njg0ODc3MCwiZXhwIjoyMDkyMjA4NzcwfQ.2dKo2uofpNE38xnA0lpjBz-BjvZYYeGYgzn0xwTJ23Y`

Stop it when needed:

```sh
docker compose down
```

These credentials are for this local test instance only.

Run quick local checks:

```sh
python3 -m py_compile custom_components/openai_codex/*.py
jq empty custom_components/openai_codex/translations/en.json custom_components/openai_codex/translations/ru.json
git diff --check
```

Run the same Python syntax check inside the Home Assistant container:

```sh
docker compose exec -T homeassistant sh -c 'python -m py_compile /config/custom_components/openai_codex/*.py'
```

Restart the test instance after integration changes:

```sh
docker compose restart homeassistant
```
