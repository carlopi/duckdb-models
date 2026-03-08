#pragma once

#include "models_internal.hpp"  // provides InteractionStats, WorkerFn
#include "duckdb/common/string.hpp"
#include "duckdb/main/client_context.hpp"

#ifdef MODELS_SHELL_EXT
#include "duckdb/main/shell_command_extension.hpp"
#else
#include "duckdb/common/exception.hpp"
namespace duckdb {
struct BaseShellState {
	virtual idx_t GetTerminalWidth() const {
		throw NotImplementedException("GetTerminalWidth not implemented");
	}
	virtual void ShellPrint(const string &str) {
		throw NotImplementedException("ShellPrint not implemented");
	}
	virtual void ShellPrintError(const string &str) {
		throw NotImplementedException("ShellPrintError not implemented");
	}
	virtual ~BaseShellState() = default;
};
} // namespace duckdb
#endif

#include <functional>
#include <llama.h>

namespace duckdb {

// ---------------------------------------------------------------------------
// LlamaState — model + context, lazily loaded, lives in ModelsExtensionState
// ---------------------------------------------------------------------------

struct LlamaState {
	llama_model   *model = nullptr;
	llama_context *ctx   = nullptr;

	LlamaState()  = default;
	~LlamaState();

	// Movable but not copyable
	LlamaState(const LlamaState &)            = delete;
	LlamaState &operator=(const LlamaState &) = delete;
	LlamaState(LlamaState &&o) noexcept : model(o.model), ctx(o.ctx)
	    { o.model = nullptr; o.ctx = nullptr; }
	LlamaState &operator=(LlamaState &&o) noexcept {
		if (this != &o) {
			if (ctx)   { llama_free(ctx);         ctx   = nullptr; }
			if (model) { llama_model_free(model);  model = nullptr; }
			model = o.model; ctx = o.ctx;
			o.model = nullptr; o.ctx = nullptr;
		}
		return *this;
	}

	// Load model from path. Returns false and prints error on failure.
	bool Load(const string &model_path, int32_t n_ctx, int32_t n_gpu_layers,
	          BaseShellState &shell_state);

	bool IsLoaded() const { return model != nullptr; }
};

// ---------------------------------------------------------------------------
// RunLlamaLoop — in-process inference, same SQL tool protocol as RunModelsSubprocessLoop
// ---------------------------------------------------------------------------

// system_prompt: overrides SUBPROCESS_SYSTEM_PROMPT when non-empty.
// single_shot_sql: for SQL-specialist models (sqlcoder etc.) — generate one SQL,
//                  execute it, return "SQL: <sql>\nResult:\n<result>" immediately.
//                  No looping, no force_final, no natural-language response attempt.
// stats: optional pointer to InteractionStats for instrumentation (nullptr = skip).
string RunLlamaLoop(ClientContext &context, BaseShellState &shell_state,
                    LlamaState &lm, const string &user_input, idx_t term_width,
                    const string &system_prompt = "",
                    bool single_shot_sql = false,
                    InteractionStats *stats = nullptr);

// worker_style: "chat" (use SUBPROCESS_SYSTEM_PROMPT + SQL_QUERY: protocol) or
//              "sqlcoder" (use SQLCODER_SYSTEM_PROMPT, bare SQL output, schema auto-injected).
// RunCoordinatorLoop auto-fetches schema and enriches the worker prompt before each DELEGATE_SQL.
// stats: optional pointer to InteractionStats for instrumentation (nullptr = skip).
string RunCoordinatorLoop(ClientContext &context, BaseShellState &shell_state,
                          LlamaState &coordinator, LlamaState &worker,
                          const string &user_input, idx_t term_width,
                          double time_budget_seconds, int32_t max_rounds,
                          const string &worker_style = "chat",
                          InteractionStats *stats = nullptr);

// Backend-agnostic coordinator loop: coordinator is always a llama model, but the worker
// is a callback — allowing any backend (Models CLI, another llama model, etc.) as the worker.
// worker_fn receives the delegate question (with schema pre-injected in "Question:...\n\nSchema:..."
// format) and should return a result string (natural language or "SQL: ...\nResult:\n..." format).
// WorkerFn is declared in models_internal.hpp, included by llama_backend.cpp.
// stats: optional pointer to InteractionStats for instrumentation (nullptr = skip).
string RunCoordinatorLoopWithWorker(ClientContext &context, BaseShellState &shell_state,
                                    LlamaState &coordinator,
                                    const WorkerFn &worker_fn,
                                    const string &user_input, idx_t term_width,
                                    double time_budget_seconds, int32_t max_rounds,
                                    InteractionStats *stats = nullptr);

} // namespace duckdb
