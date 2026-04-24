# OpenAI Codex for Home Assistant

Custom Home Assistant integration that connects Home Assistant Assist to the
ChatGPT Codex backend.

The integration adds a conversation agent powered by Codex models and can expose
Home Assistant LLM tools to the model, allowing Assist to answer questions and
control Home Assistant entities through the standard Assist pipeline.

> This project is experimental and unofficial. It is not affiliated with or
> endorsed by OpenAI or Home Assistant.

## Features

- ChatGPT device-code authorization flow.
- Conversation entity for Home Assistant Assist.
- Codex-backed responses through the OpenAI Python SDK.
- Home Assistant LLM API/tool support for entity and device control.
- Runtime Codex model catalog loading from the Codex `/models` endpoint.
- Local model catalog cache with stale-cache fallback.
- Options flow for model selection, reasoning effort, fast mode, custom
  instructions, Home Assistant LLM APIs, and live web search where supported.
- English and Russian UI translations.
- Reauthentication support when ChatGPT tokens expire or become invalid.

## Requirements

- Home Assistant with custom integrations enabled.
- A ChatGPT account that has access to Codex.
- Network access from Home Assistant to ChatGPT/OpenAI services.

## Installation

### Manual

1. Copy `custom_components/openai_codex` into the `custom_components` directory
   of your Home Assistant configuration.
2. Restart Home Assistant.
3. Go to **Settings** -> **Devices & services** -> **Add integration**.
4. Search for **OpenAI Codex** and start the setup flow.
5. Open the displayed ChatGPT authorization URL, enter the device code, approve
   access, then return to Home Assistant and submit the form.

After setup, select the **OpenAI Codex** conversation agent in Home Assistant
Assist or use it from any Assist pipeline that supports conversation agents.

## Configuration

The integration options are available from the integration's **Configure** menu.

Available options depend on the selected Codex model and may include:

- Model
- Reasoning effort
- Fast mode
- Live web search
- Home Assistant LLM APIs
- Custom instructions

## Notes

Codex backend APIs and model metadata can change. This integration loads the
model catalog at runtime and keeps a local cache so Home Assistant can continue
using the last known usable model list if a temporary catalog request fails.

Because this integration can expose Home Assistant control tools to a remote
model, review the selected Home Assistant LLM APIs and instructions before using
it with sensitive devices or automations.

## Development

Start a local Home Assistant test instance:

```sh
docker compose up -d
```

Open Home Assistant at:

```text
http://localhost:8123
```

Run quick local checks:

```sh
python3 -m py_compile custom_components/openai_codex/*.py
jq empty custom_components/openai_codex/translations/en.json custom_components/openai_codex/translations/ru.json
git diff --check
```

Run the Python syntax check inside the Home Assistant container:

```sh
docker compose exec -T homeassistant sh -c 'python -m py_compile /config/custom_components/openai_codex/*.py'
```

Restart the test instance after integration changes:

```sh
docker compose restart homeassistant
```

Stop the test instance:

```sh
docker compose down
```
