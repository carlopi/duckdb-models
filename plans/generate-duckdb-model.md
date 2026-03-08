# Plan: Fine-tune a DuckDB-specific text-to-SQL model

Goal: produce a small GGUF model that replaces sqlcoder-7b-2 as the worker
in the coordinator+worker architecture, with accurate DuckDB v1.5 syntax and
knowledge of duckdb_* system functions and this extension's duckdb_summary().

---

## TODO

### Phase 0 — Evaluate existing model first
- [ ] Download `motherduckdb/DuckDB-NSQL-7B-v0.1-GGUF` and plug it in as the
      worker (it is already DuckDB-aware, no fine-tuning needed).
      https://huggingface.co/motherduckdb/DuckDB-NSQL-7B-v0.1-GGUF
- [ ] Run a set of representative questions against the coordinator+NSQL stack.
- [ ] Decide if a fine-tune is actually needed based on failure modes observed.
      (If NSQL already handles v1.5 idioms well, skip to Phase 3 for
      extension-specific additions only.)

### Phase 1 — Assemble the base dataset
- [ ] Download `motherduckdb/duckdb-text2sql-25k` (25k DuckDB pairs, CC-BY-SA).
      https://huggingface.co/datasets/motherduckdb/duckdb-text2sql-25k
      Already in CREATE TABLE + question + SQL format. Use as-is.
- [ ] Download `b-mc2/sql-create-context` (78k Spider+WikiSQL, CC-BY-SA).
      https://huggingface.co/datasets/b-mc2/sql-create-context
      Already in SQLCoder's exact prompt format but SQLite/PostgreSQL dialect.
- [ ] Transpile sql-create-context SQL to DuckDB using Claude (Haiku for cost):
      Prompt: "Rewrite this SQL for DuckDB v1.5. Prefer DuckDB idioms. Reply
      SKIP if the query cannot be cleanly translated."
- [ ] Validate every translated query by executing it against a DuckDB v1.5
      instance with the matching schema (CREATE TABLE + INSERT or empty tables).
      Keep only queries that execute without error.
- [ ] Expect to retain ~60-70% of sql-create-context after filtering (~45-50k).

### Phase 2 — Generate DuckDB-specific synthetic examples
Topics that existing datasets under-represent, needing synthetic generation
via Claude (target ~3-5k examples total across these):

- [ ] System catalog queries:
      duckdb_tables(), duckdb_columns(), duckdb_constraints(),
      duckdb_indexes(), duckdb_functions(), duckdb_extensions()
      (correct column names — e.g. table_name not table_catalog)
- [ ] DuckDB-specific types and syntax:
      LIST / ARRAY types, STRUCT, MAP, UNION
      list_filter(), list_aggregate(), struct_pack(), unnest()
      Array slicing: arr[1:3]
- [ ] Modern DuckDB syntax:
      QUALIFY, PIVOT / UNPIVOT, EXCLUDE columns, REPLACE columns
      SELECT * EXCLUDE (col), SELECT * REPLACE (expr AS col)
- [ ] File reading:
      read_parquet(), read_csv_auto(), read_json_auto(), read_json()
      glob patterns, hive partitioning, union_by_name
- [ ] Extension functions (if loaded):
      spatial, httpfs, json, icu
- [ ] This extension's helpers:
      duckdb_summary('schema'), duckdb_summary('context'), duckdb_summary()
      duckdb_summary() column names: section, category, value, details
- [ ] Error-prone patterns to explicitly train correct behavior:
      WHERE col NOT IN (SELECT ...) with NULLs → use IS NOT DISTINCT FROM
      String concatenation: || not CONCAT (DuckDB supports both but || preferred)
      Date arithmetic: interval syntax, date_diff(), date_trunc()

Synthetic generation script outline:
  1. Sample a schema from the Spider/BIRD schema pool (CREATE TABLE DDL).
  2. For each topic bucket, ask Claude to generate N (question, SQL) pairs
     that use that schema and exercise the target syntax.
  3. Execute and validate against DuckDB v1.5.
  4. Deduplicate by SQL similarity (e.g. ROUGE-L > 0.9 → drop one).

### Phase 3 — Choose base model and training setup
Options (pick one):

- [ ] **Option A (recommended):** LoRA fine-tune of DuckDB-NSQL-7B
      Base: motherduckdb/DuckDB-NSQL-7B-v0.1 (already DuckDB-aware)
      Advantage: minimal data needed for v1.5 delta + extension idioms.
      Risk: Llama-2 base is older; context length limited to 4k.

- [ ] **Option B:** LoRA fine-tune of Qwen2.5-Coder-3B-Instruct
      Smaller, faster inference, fits in ~4 GB RAM quantized.
      Better base code understanding than Llama-2.
      Needs more data to learn DuckDB from scratch (~25k+ pairs).
      Good fit for embedded use in this extension.

- [ ] **Option C:** LoRA fine-tune of DeepSeek-Coder-7B-Instruct
      Same base as sqlcoder-7b-2 but without Defog's PostgreSQL bias baked in.
      Use the full assembled dataset (Phase 1 + 2).

Training setup:
- [ ] Use QLoRA (4-bit NF4) — fits on a single A100 40GB or 2× RTX 4090.
- [ ] Hyperparams starting point: lr=2e-4, rank=64, alpha=128, 3 epochs.
- [ ] Framework: HuggingFace TRL + PEFT, or Axolotl for convenience.
- [ ] Prompt format: keep SQLCoder's format for compatibility:
      ### Task:\n{question}\n\n### Database Schema:\n{schema}\n\n### Answer:\n{sql}
- [ ] Estimated training time: 4-8 hours on A100 for 7B model, 50k examples.
- [ ] Estimated cloud cost: $30-100 on Lambda Labs / RunPod / Modal.

### Phase 4 — Export and integrate
- [ ] Convert fine-tuned weights to GGUF:
      python llama.cpp/convert-hf-to-gguf.py --outtype q4_k_m ...
- [ ] Quantize: Q4_K_M is the sweet spot for quality vs size.
- [ ] Test GGUF in the existing llama backend (set worker model path, run .claude).
- [ ] Adjust worker prompt format in RunCoordinatorLoop if needed.
- [ ] Optionally publish to HuggingFace under CC-BY-SA 4.0.

### Phase 5 — Evaluation
- [ ] Build a DuckDB-specific eval set (~100-200 examples held out from Phase 2).
- [ ] Metric: execution accuracy (generated SQL produces same result as reference SQL).
- [ ] Compare: sqlcoder-7b-2 baseline vs DuckDB-NSQL-7B vs fine-tuned model.
- [ ] Track failure modes: wrong table name, wrong column name, unsupported syntax.

---

## Key resources

- DuckDB-NSQL-7B (GGUF): https://huggingface.co/motherduckdb/DuckDB-NSQL-7B-v0.1-GGUF
- duckdb-text2sql-25k dataset: https://huggingface.co/datasets/motherduckdb/duckdb-text2sql-25k
- sql-create-context dataset: https://huggingface.co/datasets/b-mc2/sql-create-context
- Spider dataset: https://huggingface.co/datasets/xlangai/spider
- BIRD benchmark: https://bird-bench.github.io/
- MotherDuck NSQL blog: https://motherduck.com/blog/duckdb-text2sql-llm/
- sqlglot (dialect transpiler): https://github.com/tobymao/sqlglot
- Axolotl (fine-tuning framework): https://github.com/OpenAccess-AI-Collective/axolotl

---

## Notes

- sqlglot can do mechanical dialect transpilation (SQLite/PG → DuckDB) for free
  but misses idioms. Claude validation + execution-based filtering is the
  quality gate — do not skip it.
- The unified model interface TODO in claude_extension.cpp (above ask_models)
  describes the architecture where this model slots in as the worker GGUF.
- DuckDB v1.5 specific changes vs v0.9.2 (what NSQL was trained on) should be
  captured as a dedicated synthetic batch in Phase 2. Check the DuckDB changelog
  for new functions and syntax added between 0.9 and 1.5.
