# Claude Extension — Testing Strategy

## Test format
DuckDB sqllogictest (`.test` files), placed in `extension/claude/test/`.

---

## Tier 1: No model required (always run)

These test registration, schema correctness, and structural invariants.
No Claude CLI or GGUF model needed.

| # | What | How |
|---|------|-----|
| 1 | Extension loads cleanly | `LOAD claude` → no error |
| 2 | All expected functions registered | `SELECT function_name FROM duckdb_functions() WHERE function_name IN ('ask_models', 'llama_discover_models', 'llama_import_model', 'setup_models')` returns 4 rows |
| 3 | `models_interactions()` schema | Column names and types match expected: `id BIGINT, question VARCHAR, answer VARCHAR, elapsed_seconds DOUBLE, sql_expert_model VARCHAR, coordinator_model VARCHAR, stats VARCHAR` |
| 4 | `models_interactions()` starts empty | `SELECT count(*) FROM models_interactions()` → 0 |
| 5 | `FROM models_interactions` (no parens) | Replacement scan works, also returns 0 rows |
| 6 | `llama_discover_models()` schema | Columns: `name VARCHAR, path VARCHAR, size_gb DOUBLE, source VARCHAR`; no crash even if no models present |
| 7 | `llama_instructions` non-empty | `SELECT count(*) FROM llama_instructions` > 0 |
| 8 | `llama_instructions` expected tasks | `task IN ('describe-table', 'list-tables', 'profile-data', ...)` all present |

---

## Tier 2: Local llama model required (conditional)

Skip if no GGUF model is configured / present.

| # | What | How |
|---|------|-----|
| 9  | `ask_models()` returns 2 columns | `DESCRIBE SELECT * FROM ask_models('...')` → `answer VARCHAR, stats VARCHAR` |
| 10 | `answer` is non-empty | `SELECT length(answer) > 0 FROM ask_models('say OK')` → true |
| 11 | `stats` is valid JSON with required keys | `json_extract(stats, '$.id') IS NOT NULL`, same for `elapsed_seconds`, `sql_total`, `sql_ok`, `sql_error`, `rounds` |
| 12 | `models_interactions` has a row after call | `SELECT count(*) FROM models_interactions` → 1 |
| 13 | `id` in stats matches `models_interactions.id` | Cross-check `json_extract_string(stats, '$.id')::BIGINT = (SELECT id FROM models_interactions LIMIT 1)` |
| 14 | Second call increments id | id in second call = first id + 1 |

---

## Tier 3: Local llama model + SQL execution logging (blocked on TODO)

Requires adding `DUCKDB_LOG` entries around `ExecuteQuery` call sites
(see TODO comments in `claude_extension.cpp` lines ~461 and ~720).

| # | What | How |
|---|------|-----|
| 15 | SQL is actually executed for a data question | `SET enable_logging=true; SET enabled_log_types='Claude'; CALL ask_models('how many tables are in the database?'); SELECT count(*) > 0 FROM duckdb_logs_parsed('Claude') WHERE type = 'sql_exec'` |
| 16 | SQL result is fed back (not an error) | Log has a `sql_result` entry with value `ok` |
| 17 | `sql_total` in stats matches log entry count | `json_extract(stats, '$.sql_total')::INT = (SELECT count(*) FROM duckdb_logs_parsed('Claude') WHERE type = 'sql_exec')` |

---

## Implementation order

1. Write Tier 1 tests (no dependencies, do now)
2. Write Tier 2 tests with a `skipif` / `require` guard for model presence
3. Land Tier 3 after the TODO logging is implemented
