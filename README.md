# duckdb-models

A DuckDB extension for querying your data with LLMs. Local-first via [llama.cpp](https://github.com/ggml-org/llama.cpp), with an optional Claude CLI backend.

## Installation

```sql
INSTALL models FROM community;
LOAD models;
```

## Setup

### Local inference (llama.cpp)

```sql
-- First-time setup: pick models interactively
CALL setup_models();

-- Or specify models directly
CALL setup_models(
    coordinator_model := 'qwen2.5-coder-14b',
    sql_expert_model  := 'qwen2.5-coder-14b'
);
```

Settings are saved to `~/.duckdb/extension_data/models/setup.sql` and auto-loaded on the next `LOAD models`.

To overwrite an existing configuration:

```sql
CALL setup_models(override := true);
```

### Claude CLI backend

```sql
CALL setup_models(claude := true);
```

Requires the [Claude CLI](https://claude.ai/download) to be installed and authenticated.

## Querying

```sql
-- Ask a question about your data
FROM ask_models('Which customers placed the most orders last month?');

-- View conversation history
FROM models_interactions;
```

The extension automatically introspects your schema, runs SQL queries as needed, and returns a natural-language answer.

## Model management

```sql
-- Browse the built-in model catalog
FROM llama_models();

-- Download a model
CALL llama_download_model('qwen2.5-coder-14b');

-- Scan local filesystem for GGUF files
FROM llama_discover_models();
```

## Settings

| Setting | Description |
|---|---|
| `coordinator_model` | Model used to plan and synthesize answers |
| `sql_expert_model` | Model used for SQL generation |
| `llama_gpu_layers` | Number of layers to offload to GPU (default: auto) |
| `llama_context_size` | Context window size (default: 4096) |

```sql
-- View current settings
FROM duckdb_settings() WHERE name ILIKE '%model%';
```

## How it works

By default the extension runs inference locally using llama.cpp — no API keys required. If you configure the Claude CLI backend, queries are forwarded to a `claude` subprocess instead.

> **TODO**: API-based backends (Anthropic API, OpenAI, etc.) are not yet implemented. The current backends are local llama.cpp and the Claude CLI.
