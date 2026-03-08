#include "models_extension.hpp"
#include "models_internal.hpp"

#include "llama_backend.hpp"

#include "duckdb/function/table_function.hpp"
#include "duckdb/logging/log_manager.hpp"
#include "duckdb/logging/log_type.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/shell_command_extension.hpp"
#include "duckdb/common/http_util.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parser/tableref/column_data_ref.hpp"
#include "yyjson.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <regex>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <functional>
#include <thread>
#include <unordered_set>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>

using namespace duckdb_yyjson; // NOLINT

namespace duckdb {

// System prompt sent to the claude subprocess.
// Instructs it to use the SQL_QUERY: text protocol for querying the database.
const char * const SUBPROCESS_SYSTEM_PROMPT =
    "You are a DuckDB SQL assistant. You are connected to a live, running DuckDB instance.\n"
    "This DuckDB instance is listening for queries in a loop: every SQL_QUERY: you emit is\n"
    "executed immediately and its full result is returned to you as the next message.\n"
    "You can issue as many queries as you need — there is no limit on rounds.\n"
    "\n"
    "HOW TO QUERY:\n"
    "Output EXACTLY this on its own line, with nothing else in your response:\n"
    "SQL_QUERY: <sql statement>\n"
    "The DuckDB instance will execute it and send the result back as the next message.\n"
    "You then output another SQL_QUERY: or your final answer. This loop runs until you answer.\n"
    "\n"
    "SUGGESTED WORKFLOW:\n"
    "1. Run SQL_QUERY: FROM duckdb_summary() to orient yourself (schema, settings, extensions).\n"
    "2. For questions about actual data, run targeted queries against the tables.\n"
    "3. Once you have enough information, write your final answer — do not keep querying.\n"
    "If a query returns the information needed to answer, stop issuing queries and answer immediately.\n"
    "Never repeat the same query twice — if you already have the result, use it.\n"
    "\n"
    "HELPER MACRO:\n"
    "duckdb_summary([topic]) returns (section, category, value, details).\n"
    "  FROM duckdb_summary('schema')   — tables and columns with types\n"
    "  FROM duckdb_summary('context')  — attached databases, extensions, settings\n"
    "  FROM duckdb_summary()           — everything above\n"
    "To discover available functions (e.g. from loaded extensions or user macros):\n"
    "  SELECT function_name, description FROM duckdb_functions() WHERE function_type = 'macro'\n"
    "  SELECT function_name, description FROM duckdb_functions() WHERE function_name ILIKE '%<keyword>%'\n"
    "\n"
    "TASK INSTRUCTIONS:\n"
    "The table 'llama_instructions' (task VARCHAR, hint VARCHAR) contains DuckDB-specific\n"
    "guidance for common subtasks. Available tasks:\n"
    "  describe-table, list-tables, profile-data, query-performance, indexes, constraints,\n"
    "  functions, extensions, file-formats, aggregate-stats, window-functions, date-time,\n"
    "  string-ops, json\n"
    "Fetch hints for relevant tasks before querying, e.g.:\n"
    "  SQL_QUERY: FROM llama_instructions WHERE task IN ('describe-table', 'profile-data')\n"
    "\n"
    "CONVERSATION HISTORY:\n"
    "Previous Q&A is in the virtual table 'models_interactions' (id BIGINT, question VARCHAR, answer VARCHAR, elapsed_seconds DOUBLE, sql_expert_model VARCHAR, coordinator_model VARCHAR, stats VARCHAR).\n"
    "\n"
    "DUCKDB GOTCHAS (common mistakes to avoid):\n"
    "- duckdb_tables() takes NO arguments. Filter with WHERE:\n"
    "    FROM duckdb_tables() WHERE database_name = '<db>'   -- CORRECT\n"
    "    FROM duckdb_tables('<db>')                           -- WRONG, will error\n"
    "- Same for duckdb_columns(), duckdb_schemas(), etc. — always use WHERE to filter.\n"
    "- Do NOT use information_schema.tables WHERE table_schema = '<db>' for attached\n"
    "  databases — that returns 0 rows. Use duckdb_tables() WHERE database_name = '<db>'.\n"
    "- duckdb_summary(topic) valid topics are ONLY: 'schema', 'context', 'all' (default).\n"
    "  'database' is NOT a valid topic. To list databases: FROM duckdb_summary('context').\n"
    "- If a query returns 0 rows, the filter is likely wrong — broaden it or use\n"
    "  FROM duckdb_summary('context') to see what databases actually exist.\n"
    "\n"
    "RULES:\n"
    "- Only SELECT, WITH, FROM, SHOW, DESCRIBE, EXPLAIN, PRAGMA are allowed.\n"
    "- Be concise in your final answer.";

// ---------------------------------------------------------------------------
// Per-invocation state stored on the ShellCommandExtension
// ---------------------------------------------------------------------------

struct ConversationEntry {
	idx_t  id;
	string question;
	string answer;          // empty when pending
	double elapsed_seconds; // negative when pending
	bool   pending;
	string sql_expert_model;
	string coordinator_model;
	string stats; // JSON: {"sql_total":N,"sql_ok":N,"sql_error":N,"rounds":N,"delegations":N}

	ConversationEntry(idx_t id, string question, string answer, double elapsed_seconds, bool pending,
	                  string sql_expert_model = "", string coordinator_model = "",
	                  string stats = "{}")
	    : id(id), question(std::move(question)), answer(std::move(answer)),
	      elapsed_seconds(elapsed_seconds), pending(pending),
	      sql_expert_model(std::move(sql_expert_model)),
	      coordinator_model(std::move(coordinator_model)),
	      stats(std::move(stats)) {
	}
};

#ifdef MODELS_SHELL_EXT
struct ModelsExtensionState : public ShellCommandExtensionInfo {
#else
struct ModelsExtensionState {
#endif
	vector<ConversationEntry> history;

	LlamaState llama;
	string     llama_model_path;
	int32_t    llama_n_ctx        = 8192;
	int32_t    llama_n_gpu_layers = -1; // -1 = all on GPU

	LlamaState coordinator_llama;
	string     coordinator_model_path;
};

// Passed through TableFunction::function_info so the bind function can access state
struct ModelsTableFunctionInfo : public TableFunctionInfo {
	shared_ptr<ModelsExtensionState> state;

	explicit ModelsTableFunctionInfo(shared_ptr<ModelsExtensionState> s) : state(std::move(s)) {
	}
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// SQL safety guard
// ---------------------------------------------------------------------------

bool IsSafeSQL(const string &sql) {
	idx_t i = 0;
	while (i < sql.size() && (sql[i] == ' ' || sql[i] == '\t' || sql[i] == '\n' || sql[i] == '\r')) {
		i++;
	}
	string prefix = StringUtil::Lower(sql.substr(i, 8));
	return StringUtil::StartsWith(prefix, "select")  ||
	       StringUtil::StartsWith(prefix, "with")    ||
	       StringUtil::StartsWith(prefix, "from")    ||
	       StringUtil::StartsWith(prefix, "show")    ||
	       StringUtil::StartsWith(prefix, "describ") ||
	       StringUtil::StartsWith(prefix, "pragma")  ||
	       StringUtil::StartsWith(prefix, "explain");
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

QueryExecutionResult ExecuteQuery(DatabaseInstance &db, const string &sql) {
	if (!IsSafeSQL(sql)) {
		return {true, "only SELECT / WITH / FROM / SHOW / DESCRIBE / EXPLAIN / PRAGMA are allowed"};
	}

	// Each query gets its own connection so that interrupting it does not
	// affect subsequent queries (interrupt state is per-ClientContext).
	Connection conn(db);

	// Enforce a hard timeout: a background thread calls conn.Interrupt() if the
	// query does not finish within QUERY_TIMEOUT_SECONDS.
	std::atomic<bool> query_done {false};
	std::thread timeout_thread([&conn, &query_done]() {
		for (int i = 0; i < QUERY_TIMEOUT_SECONDS * 10; i++) {
			if (query_done.load(std::memory_order_relaxed)) {
				return;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		if (!query_done.load(std::memory_order_relaxed)) {
			conn.Interrupt();
		}
	});

	auto result = conn.Query(sql);
	query_done.store(true, std::memory_order_relaxed);
	timeout_thread.join();

	if (result->HasError()) {
		const string &err = result->GetError();
		if (err.find("Interrupted") != string::npos) {
			return {true, StringUtil::Format("query timed out after %d seconds", QUERY_TIMEOUT_SECONDS)};
		}
		return {true, err};
	}

	// Collect rows first so we can suppress headers on empty results.
	// Wrap in try/catch: complex types (MAP, UNION, nested arrays) can throw
	// during Value::ToString(), which would otherwise crash the process.
	vector<vector<string>> rows;
	bool truncated = false;
	try {
		while (!truncated) {
			auto chunk = result->Fetch();
			if (!chunk || chunk->size() == 0) break;
			for (idx_t r = 0; r < chunk->size(); r++) {
				if (rows.size() >= MAX_RESULT_ROWS) { truncated = true; break; }
				vector<string> row;
				for (idx_t c = 0; c < chunk->ColumnCount(); c++) {
					try {
						row.push_back(chunk->GetValue(c, r).ToString());
					} catch (...) {
						row.push_back("<unrepresentable>");
					}
				}
				rows.push_back(std::move(row));
			}
		}
	} catch (const std::exception &e) {
		return {true, StringUtil::Format("Error reading result: %s", e.what())};
	} catch (...) {
		return {true, "Error reading result: unknown exception"};
	}

	if (rows.empty()) {
		return {false, "(0 rows)\n"};
	}

	// Emit header + rows.
	string out;
	for (idx_t c = 0; c < result->names.size(); c++) {
		if (c) out += '\t';
		out += result->names[c];
	}
	out += '\n';
	for (auto &row : rows) {
		for (idx_t c = 0; c < row.size(); c++) {
			if (c) out += '\t';
			out += row[c];
		}
		out += '\n';
	}
	if (truncated) out += "(truncated after " + to_string(MAX_RESULT_ROWS) + " rows)\n";
	return {false, out};
}

// ---------------------------------------------------------------------------
// Extract SQL_QUERY: <sql> from a response line
// Returns empty string if not found.
// Matches "SQL_QUERY:" at the start of any line in `text`.
// ---------------------------------------------------------------------------

string ExtractSQLQuery(const string &text) {
	static const char MARKER[]   = "SQL_QUERY:";
	static const idx_t MARKER_LEN = sizeof(MARKER) - 1;

	idx_t pos = 0;
	while (pos + MARKER_LEN <= text.size()) {
		idx_t found = text.find(MARKER, pos);
		if (found == string::npos) return "";

		bool at_line_start = (found == 0 || text[found - 1] == '\n');
		if (at_line_start) {
			idx_t sql_start = found + MARKER_LEN;
			while (sql_start < text.size() && (text[sql_start] == ' ' || text[sql_start] == '\t'))
				sql_start++;
			idx_t sql_end = text.find('\n', sql_start);
			if (sql_end == string::npos) sql_end = text.size();
			while (sql_end > sql_start && (text[sql_end - 1] == ' ' || text[sql_end - 1] == '\r'))
				sql_end--;
			return StripSpecialTokens(text.substr(sql_start, sql_end - sql_start));
		}
		pos = found + 1;
	}
	return "";
}

// ---------------------------------------------------------------------------
// Core agentic loop — spawns `claude -p` as a subprocess and runs the
// SQL query tool protocol over stdin/stdout pipes.
// ---------------------------------------------------------------------------

// Launch a `claude -p` subprocess with the given system_prompt.
// Returns the pid (child has called exec or _exit); parent gets pipe FILE handles.
// Returns -1 on fork failure (pipes already closed).
static pid_t SpawnModelsProcess(const char *system_prompt,
                                FILE **to_claude_out, FILE **from_claude_out) {
	int stdin_fds[2], stdout_fds[2];
	if (pipe(stdin_fds) < 0 || pipe(stdout_fds) < 0) {
		return -1;
	}

	pid_t pid = fork();
	if (pid == 0) {
		// Child: wire up pipes and exec claude
		dup2(stdin_fds[0],  STDIN_FILENO);
		dup2(stdout_fds[1], STDOUT_FILENO);
		close(stdin_fds[0]);  close(stdin_fds[1]);
		close(stdout_fds[0]); close(stdout_fds[1]);

		// Allow nested invocation (we may be inside a Claude Code session)
		unsetenv("CLAUDECODE");

		execlp("claude", "claude",
		       "-p",
		       "--verbose",
		       "--input-format",            "stream-json",
		       "--output-format",           "stream-json",
		       "--no-session-persistence",
		       "--tools",                   "",
		       "--system-prompt",           system_prompt,
		       nullptr);
		// exec failed
		const char msg[] = "Error: claude executable not found\n";
		write(STDERR_FILENO, msg, sizeof(msg) - 1);
		_exit(127);
	}

	// Parent: close child-side ends
	close(stdin_fds[0]);
	close(stdout_fds[1]);

	*to_claude_out   = fdopen(stdin_fds[1],  "w");
	*from_claude_out = fdopen(stdout_fds[0], "r");
	return pid;
}

static string RunModelsSubprocessLoop(ClientContext &context, BaseShellState &shell_state,
                                      const string &user_input, idx_t term_width,
                                      InteractionStats &stats) {
	FILE *to_claude = nullptr, *from_claude = nullptr;
	pid_t pid = SpawnModelsProcess(SUBPROCESS_SYSTEM_PROMPT, &to_claude, &from_claude);
	if (pid < 0 || !to_claude || !from_claude) {
		shell_state.ShellPrintError(StringUtil::Format("Error: failed to spawn claude subprocess: %s\n", strerror(errno)));
		if (to_claude)   fclose(to_claude);
		if (from_claude) fclose(from_claude);
		if (pid > 0) waitpid(pid, nullptr, 0);
		return "";
	}

	// Send the initial user question
	string init_msg = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":" + ModelsJsonEscapeString(user_input) + "}}\n";
	DUCKDB_LOG(context, ModelsLogType, "out", init_msg);
	fputs(init_msg.c_str(), to_claude);
	fflush(to_claude);

	DUCKDB_LOG(context, ModelsLogType, "start", StringUtil::Format("pid=%d", (int)pid));

	// Dynamic line reader: handles lines of any length (no truncation).
	auto read_line = [](FILE *fp, string &line) -> bool {
		line.clear();
		int c;
		bool got = false;
		while ((c = fgetc(fp)) != EOF) {
			got = true;
			if (c == '\n') break;
			line += (char)c;
		}
		if (!line.empty() && line.back() == '\r') line.pop_back();
		return got;
	};

	int    sql_rounds = 0;
	bool   done       = false;
	string final_response;
	string current_round_text; // text accumulated from assistant events in the current round

	string line_buf_str;
	while (!done && read_line(from_claude, line_buf_str)) {
		if (line_buf_str.empty()) continue;
		const char *line_ptr = line_buf_str.c_str();
		idx_t       len      = line_buf_str.size();

		DUCKDB_LOG(context, ModelsLogType, "in", line_buf_str);

		yyjson_doc *doc = yyjson_read(line_ptr, len, 0);
		if (!doc) continue;
		yyjson_val *root = yyjson_doc_get_root(doc);

		yyjson_val *type_v     = yyjson_obj_get(root, "type");
		yyjson_val *subtype_v  = yyjson_obj_get(root, "subtype");
		string event_type  = (type_v    && yyjson_is_str(type_v))    ? yyjson_get_str(type_v)    : "";
		string event_sub   = (subtype_v && yyjson_is_str(subtype_v)) ? yyjson_get_str(subtype_v) : "";

		if (event_type == "assistant") {
			// Capture text blocks and thinking from assistant events.
			// The result event's "result" field sometimes only carries the first
			// line; the full response accumulates here across all assistant events
			// in the current round.
			yyjson_val *msg_v     = yyjson_obj_get(root, "message");
			yyjson_val *content_v = msg_v ? yyjson_obj_get(msg_v, "content") : nullptr;
			if (content_v && yyjson_is_str(content_v)) {
				// content is a plain string
				current_round_text += yyjson_get_str(content_v);
			} else if (content_v && yyjson_is_arr(content_v)) {
				// content is an array of blocks (text / thinking / tool_use …)
				idx_t arr_len = yyjson_arr_size(content_v);
				for (idx_t i = 0; i < arr_len; i++) {
					yyjson_val *block = yyjson_arr_get(content_v, i);
					yyjson_val *btype = block ? yyjson_obj_get(block, "type") : nullptr;
					if (!btype || !yyjson_is_str(btype)) continue;
					string bt = yyjson_get_str(btype);
					if (bt == "thinking") {
						yyjson_val *thinking_v = yyjson_obj_get(block, "thinking");
						if (thinking_v && yyjson_is_str(thinking_v)) {
							// Take first line, truncate to 120 chars
							string t = yyjson_get_str(thinking_v);
							auto nl = t.find('\n');
							if (nl != string::npos) t = t.substr(0, nl);
							static constexpr idx_t PREFIX_LEN = 11; // "[thinking] "
							idx_t max_text = term_width > PREFIX_LEN + 3 ? term_width - PREFIX_LEN - 3 : 0;
							if (max_text > 0 && t.size() > max_text) t = t.substr(0, max_text) + "...";
							shell_state.ShellPrintError("\r\033[K[thinking] " + t);
						}
					} else if (bt == "text") {
						yyjson_val *text_v = yyjson_obj_get(block, "text");
						if (text_v && yyjson_is_str(text_v)) {
							current_round_text += yyjson_get_str(text_v);
						}
					}
				}
			}
		} else if (event_type == "result") {
			if (event_sub == "error") {
				shell_state.ShellPrintError("\r\033[K");
				yyjson_val *err_v = yyjson_obj_get(root, "error");
				if (err_v && yyjson_is_str(err_v)) {
					DUCKDB_LOG(context, ModelsLogType, "misc",
					           StringUtil::Format("error: %s", yyjson_get_str(err_v)));
					shell_state.ShellPrintError(StringUtil::Format("Error: %s\n", yyjson_get_str(err_v)));
				}
				done = true;
			} else if (event_sub == "success") {
				stats.rounds++;
				yyjson_val *result_v = yyjson_obj_get(root, "result");
				string response = (result_v && yyjson_is_str(result_v)) ? yyjson_get_str(result_v) : "";
				// Prefer the full text accumulated from assistant events when it is
				// longer than the result field (which sometimes only has the first line).
				if (current_round_text.size() > response.size()) {
					response = current_round_text;
				}
				current_round_text.clear(); // reset for next round

				string sql = ExtractSQLQuery(response);
				if (!sql.empty() && IsSafeSQL(sql) && sql_rounds < MAX_SQL_ROUNDS) {
					sql_rounds++;
					// TODO: log sql_exec + sql_result entries here for testability via duckdb_logs_parsed('Models')
					auto qresult = ExecuteQuery(*context.db, sql);
					stats.sql_total++;
					if (qresult.is_error) stats.sql_error++; else stats.sql_ok++;

					// Send result back for the next round
					string feedback;
					if (qresult.is_error) {
						feedback = "SQL_ERROR: The query below failed. Do NOT retry the same query.\n"
						           "Diagnose the error, fix the SQL, and issue a corrected SQL_QUERY.\n"
						           "Failed SQL:\n" + sql + "\nError message:\n" + qresult.text;
					} else {
						feedback = "SQL_RESULT:\n" + qresult.text;
					}
					string result_msg = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":" +
					    ModelsJsonEscapeString(feedback) + "}}\n";
					DUCKDB_LOG(context, ModelsLogType, "out", result_msg);
					fputs(result_msg.c_str(), to_claude);
					fflush(to_claude);
				} else {
					if (sql_rounds >= MAX_SQL_ROUNDS) {
						DUCKDB_LOG(context, ModelsLogType, "misc",
						           StringUtil::Format("reached maximum query rounds (%d)", MAX_SQL_ROUNDS));
					}
					shell_state.ShellPrintError("\r\033[K");
					final_response = response;
					done = true;
				}
			}
		}

		yyjson_doc_free(doc);
	}

	fclose(to_claude);
	fclose(from_claude);
	int status = 0;
	waitpid(pid, &status, 0);
	if (WIFEXITED(status)) {
		int code = WEXITSTATUS(status);
		DUCKDB_LOG(context, ModelsLogType, "exit", StringUtil::Format("pid=%d code=%d", (int)pid, code));
		if (code == 127)
			shell_state.ShellPrintError("Error: 'claude' CLI not found. Install Claude Code to use this feature.\n");
	} else if (WIFSIGNALED(status)) {
		DUCKDB_LOG(context, ModelsLogType, "exit", StringUtil::Format("pid=%d signal=%d", (int)pid, WTERMSIG(status)));
	}
	return final_response;
}

// ---------------------------------------------------------------------------
// Coordinator loop with Claude subprocess as the coordinator.
// Uses COORDINATOR_SYSTEM_PROMPT; expects DELEGATE_SQL:/ASK_SQL:/FINAL_ANSWER: commands.
// worker_fn: called for each DELEGATE_SQL: command; receives (question + schema) and
//            returns the worker's answer.
// ---------------------------------------------------------------------------

// Helper: extract a coordinator command from a line, e.g. "DELEGATE_SQL: how many rows?"
// Returns the text after the marker, or "" if not found.
// Extract the value after `marker` at the start of a line.
// multiline=false: stops at the first newline (for SQL commands like ASK_SQL:).
// multiline=true:  takes everything to end-of-text (for FINAL_ANSWER:).
static string ExtractCoordCommand(const string &text, const char *marker, bool multiline = false) {
	static constexpr idx_t MAX_LINE = 2048;
	size_t mlen = strlen(marker);
	size_t pos  = 0;
	while (pos < text.size()) {
		size_t found = text.find(marker, pos);
		if (found == string::npos) return "";
		bool at_start = (found == 0 || text[found - 1] == '\n');
		if (at_start) {
			size_t val_start = found + mlen;
			while (val_start < text.size() && (text[val_start] == ' ' || text[val_start] == '\t'))
				val_start++;
			size_t val_end;
			if (multiline) {
				val_end = text.size();
			} else {
				val_end = text.find('\n', val_start);
				if (val_end == string::npos) val_end = text.size();
				if (val_end - val_start > MAX_LINE) val_end = val_start + MAX_LINE;
			}
			while (val_end > val_start && (text[val_end - 1] == ' ' || text[val_end - 1] == '\r' ||
			                               text[val_end - 1] == '\n'))
				val_end--;
			return text.substr(val_start, val_end - val_start);
		}
		pos = found + 1;
	}
	return "";
}


// Run Claude subprocess as the COORDINATOR; dispatch DELEGATE_SQL to worker_fn.
// worker_fn receives the delegate question (with schema pre-injected).
// Returns the final answer (FINAL_ANSWER: content, or last natural-language response).
static string RunModelsCoordinatorLoop(ClientContext &context, BaseShellState &shell_state,
                                       const WorkerFn &worker_fn,
                                       const string &user_input, idx_t term_width,
                                       int32_t max_rounds,
                                       InteractionStats &stats) {
	FILE *to_claude = nullptr, *from_claude = nullptr;
	pid_t pid = SpawnModelsProcess(COORDINATOR_SYSTEM_PROMPT, &to_claude, &from_claude);
	if (pid < 0 || !to_claude || !from_claude) {
		shell_state.ShellPrintError(StringUtil::Format("Error: failed to spawn coordinator subprocess: %s\n", strerror(errno)));
		if (to_claude)   fclose(to_claude);
		if (from_claude) fclose(from_claude);
		if (pid > 0) waitpid(pid, nullptr, 0);
		return "";
	}

	// Phase 0: fetch schema so we can inject it into worker questions.
	string schema_ddl;
	{
		auto schema_qr = ExecuteQuery(*context.db, "FROM duckdb_summary('schema')");
		if (!schema_qr.is_error && schema_qr.text != "(0 rows)\n") {
			// Use a compact representation for the coordinator context injection.
			// We just pass the raw summary text; coordinator can parse it.
			schema_ddl = schema_qr.text;
		}
	}

	// Build the initial user message: question + database context.
	string db_context;
	{
		auto ctx_qr = ExecuteQuery(*context.db, "FROM duckdb_summary('context')");
		if (!ctx_qr.is_error) db_context = "[Database context]\n" + ctx_qr.text + "\n";
		if (!schema_ddl.empty()) db_context += "[Database schema]\n" + schema_ddl + "\n";
	}
	string init_question = user_input;
	if (!db_context.empty()) {
		init_question = db_context + "\nUser question: " + user_input;
	}

	string init_msg = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":" +
	                  ModelsJsonEscapeString(init_question) + "}}\n";
	DUCKDB_LOG(context, ModelsLogType, "out", init_msg);
	fputs(init_msg.c_str(), to_claude);
	fflush(to_claude);
	DUCKDB_LOG(context, ModelsLogType, "start", StringUtil::Format("coordinator pid=%d", (int)pid));

	auto read_line_coord = [](FILE *fp, string &line) -> bool {
		line.clear();
		int c;
		bool got = false;
		while ((c = fgetc(fp)) != EOF) {
			got = true;
			if (c == '\n') break;
			line += (char)c;
		}
		if (!line.empty() && line.back() == '\r') line.pop_back();
		return got;
	};

	int    rounds     = 0;
	bool   done       = false;
	string final_response;

	std::unordered_set<string> seen_failed_delegations;

	string coord_line;
	while (!done && read_line_coord(from_claude, coord_line)) {
		if (coord_line.empty()) continue;
		const char *line_ptr = coord_line.c_str();
		idx_t       len      = coord_line.size();

		DUCKDB_LOG(context, ModelsLogType, "in", coord_line);

		yyjson_doc *doc = yyjson_read(line_ptr, len, 0);
		if (!doc) continue;
		yyjson_val *root = yyjson_doc_get_root(doc);

		yyjson_val *type_v    = yyjson_obj_get(root, "type");
		yyjson_val *subtype_v = yyjson_obj_get(root, "subtype");
		string event_type = (type_v    && yyjson_is_str(type_v))    ? yyjson_get_str(type_v)    : "";
		string event_sub  = (subtype_v && yyjson_is_str(subtype_v)) ? yyjson_get_str(subtype_v) : "";

		if (event_type == "assistant") {
			// Show thinking if present
			yyjson_val *msg_v     = yyjson_obj_get(root, "message");
			yyjson_val *content_v = msg_v ? yyjson_obj_get(msg_v, "content") : nullptr;
			if (content_v && yyjson_is_arr(content_v)) {
				for (idx_t i = 0; i < yyjson_arr_size(content_v); i++) {
					yyjson_val *block = yyjson_arr_get(content_v, i);
					yyjson_val *btype = block ? yyjson_obj_get(block, "type") : nullptr;
					if (btype && yyjson_is_str(btype) && string(yyjson_get_str(btype)) == "thinking") {
						yyjson_val *thinking_v = yyjson_obj_get(block, "thinking");
						if (thinking_v && yyjson_is_str(thinking_v)) {
							string t = yyjson_get_str(thinking_v);
							auto nl = t.find('\n');
							if (nl != string::npos) t = t.substr(0, nl);
							static constexpr idx_t PREFIX_LEN = 11;
							idx_t max_text = term_width > PREFIX_LEN + 3 ? term_width - PREFIX_LEN - 3 : 0;
							if (max_text > 0 && t.size() > max_text) t = t.substr(0, max_text) + "...";
							shell_state.ShellPrintError("\r\033[K[thinking] " + t);
						}
					}
				}
			}
		} else if (event_type == "result") {
			if (event_sub == "error") {
				shell_state.ShellPrintError("\r\033[K");
				yyjson_val *err_v = yyjson_obj_get(root, "error");
				if (err_v && yyjson_is_str(err_v)) {
					shell_state.ShellPrintError(StringUtil::Format("Error: %s\n", yyjson_get_str(err_v)));
				}
				done = true;
			} else if (event_sub == "success") {
				stats.rounds++;
				yyjson_val *result_v = yyjson_obj_get(root, "result");
				string response = (result_v && yyjson_is_str(result_v)) ? yyjson_get_str(result_v) : "";

				// Check for coordinator commands.
				string final_ans  = ExtractCoordCommand(response, "FINAL_ANSWER:", true);
				string delegate_q = ExtractCoordCommand(response, "DELEGATE_SQL:");
				string direct_sql = ExtractCoordCommand(response, "ASK_SQL:");

				if (!final_ans.empty()) {
					shell_state.ShellPrintError("\r\033[K");
					shell_state.ShellPrint(final_ans + "\n");
					final_response = final_ans;
					done = true;

				} else if (!delegate_q.empty() && rounds < max_rounds) {
					rounds++;
					stats.delegations++;
					shell_state.ShellPrint(StringUtil::Format("[coordinator] delegating: %s\n", delegate_q));

					if (seen_failed_delegations.count(delegate_q)) {
						string feedback = "[SYSTEM: This exact question already failed. "
						                  "Rephrase with more specific constraints, or use ASK_SQL directly.]";
						string msg = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":" +
						             ModelsJsonEscapeString(feedback) + "}}\n";
						fputs(msg.c_str(), to_claude);
						fflush(to_claude);
					} else {
						string worker_input = delegate_q;
						if (!schema_ddl.empty()) {
							worker_input = "Question: " + delegate_q + "\n\nSchema:\n" + schema_ddl;
						}
						string worker_ans = worker_fn(worker_input);
						size_t err_marker = worker_ans.find("\nError:\n");
						if (err_marker != string::npos) {
							seen_failed_delegations.insert(delegate_q);
						}
						string feedback = "[Worker answered \"" + delegate_q + "\"]\n" + worker_ans;
						string hints = MatchErrorHints(worker_ans);
						if (!hints.empty()) feedback += "\n[Error hints]\n" + hints;
						string msg = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":" +
						             ModelsJsonEscapeString(feedback) + "}}\n";
						DUCKDB_LOG(context, ModelsLogType, "out", msg);
						fputs(msg.c_str(), to_claude);
						fflush(to_claude);
					}

				} else if (!direct_sql.empty() && rounds < max_rounds) {
					rounds++;
					shell_state.ShellPrint(StringUtil::Format("[coordinator] SQL: %s\n", direct_sql));
					string feedback;
					if (IsSafeSQL(direct_sql)) {
						// TODO: log sql_exec + sql_result entries here for testability via duckdb_logs_parsed('Models')
						auto qr = ExecuteQuery(*context.db, direct_sql);
						stats.sql_total++;
						if (qr.is_error) stats.sql_error++; else stats.sql_ok++;
						feedback = qr.is_error
						    ? ("SQL_ERROR: The query below failed. Do NOT retry the same query.\n"
						       "Diagnose the error, fix the SQL, and issue a corrected SQL_QUERY.\n"
						       "Failed SQL:\n" + direct_sql + "\nError message:\n" + qr.text)
						    : ("SQL_RESULT:\n" + qr.text);
					} else {
						feedback = "[SQL rejected — only SELECT/FROM/... allowed]";
					}
					string msg = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":" +
					             ModelsJsonEscapeString(feedback) + "}}\n";
					DUCKDB_LOG(context, ModelsLogType, "out", msg);
					fputs(msg.c_str(), to_claude);
					fflush(to_claude);

				} else {
					// No recognized command, or budget exhausted — treat as final answer.
					shell_state.ShellPrintError("\r\033[K");
					if (!response.empty()) {
						shell_state.ShellPrint(response + "\n");
						final_response = response;
					}
					done = true;
				}
			}
		}

		yyjson_doc_free(doc);
	}

	fclose(to_claude);
	fclose(from_claude);
	int status = 0;
	waitpid(pid, &status, 0);
	if (WIFEXITED(status)) {
		int code = WEXITSTATUS(status);
		DUCKDB_LOG(context, ModelsLogType, "exit", StringUtil::Format("coordinator pid=%d code=%d", (int)pid, code));
		if (code == 127)
			shell_state.ShellPrintError("Error: 'claude' CLI not found. Install Claude Code to use this feature.\n");
	} else if (WIFSIGNALED(status)) {
		DUCKDB_LOG(context, ModelsLogType, "exit", StringUtil::Format("coordinator pid=%d signal=%d", (int)pid, WTERMSIG(status)));
	}
	return final_response;
}

// ---------------------------------------------------------------------------
// models_interactions() table function
// ---------------------------------------------------------------------------

struct ModelsInteractionsBindData : public FunctionData {
	vector<ConversationEntry> snapshot; // copy at bind time

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<ModelsInteractionsBindData>();
		copy->snapshot = snapshot;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other) const override {
		return false; // never cache
	}
};

struct ModelsInteractionsGlobalState : public GlobalTableFunctionState {
	idx_t row = 0;
};

static unique_ptr<FunctionData> ModelsInteractionsBind(ClientContext & /*context*/, TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types, vector<string> &names) {
	return_types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	names        = {"id", "question", "answer", "total_time", "sql_expert_model", "coordinator_model", "stats"};

	auto &fn_info = input.info->Cast<ModelsTableFunctionInfo>();
	auto bind_data = make_uniq<ModelsInteractionsBindData>();
	bind_data->snapshot = fn_info.state->history;
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> ModelsInteractionsInitGlobal(ClientContext & /*context*/,
                                                                         TableFunctionInitInput & /*input*/) {
	return make_uniq<ModelsInteractionsGlobalState>();
}

static void ModelsInteractionsScan(ClientContext & /*context*/, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<ModelsInteractionsBindData>();
	auto &gstate    = data_p.global_state->Cast<ModelsInteractionsGlobalState>();

	idx_t count = 0;
	while (gstate.row < bind_data.snapshot.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = bind_data.snapshot[gstate.row];
		output.data[0].SetValue(count, Value::BIGINT(entry.id));
		output.data[1].SetValue(count, Value(entry.question));
		output.data[2].SetValue(count, entry.pending ? Value(LogicalType::VARCHAR) : Value(entry.answer));
		output.data[3].SetValue(count, entry.pending ? Value(LogicalType::DOUBLE)  : Value(entry.elapsed_seconds));
		output.data[4].SetValue(count, Value(entry.sql_expert_model));
		output.data[5].SetValue(count, Value(entry.coordinator_model));
		output.data[6].SetValue(count, Value(entry.stats));
		gstate.row++;
		count++;
	}
	output.SetCardinality(count);
}

// ---------------------------------------------------------------------------
// Replacement scan: FROM models_interactions (without parentheses)
// ---------------------------------------------------------------------------

struct ModelsReplacementScanData : public ReplacementScanData {
	shared_ptr<ModelsExtensionState> state;

	explicit ModelsReplacementScanData(shared_ptr<ModelsExtensionState> s) : state(std::move(s)) {
	}
};

static unique_ptr<TableRef> ModelsInteractionsReplacementScan(ClientContext &context, ReplacementScanInput &input,
                                                              optional_ptr<ReplacementScanData> data) {
	if (input.table_name != "models_interactions") {
		return nullptr;
	}
	auto &scan_data = data->Cast<ModelsReplacementScanData>();
	auto &history   = scan_data.state->history;

	vector<LogicalType> types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	vector<string>      names = {"id", "question", "answer", "total_time", "sql_expert_model", "coordinator_model", "stats"};
	auto collection = make_uniq<ColumnDataCollection>(context, types);

	if (!history.empty()) {
		DataChunk chunk;
		chunk.Initialize(context, types);
		idx_t row = 0;
		for (auto &entry : history) {
			chunk.data[0].SetValue(row, Value::BIGINT(entry.id));
			chunk.data[1].SetValue(row, Value(entry.question));
			chunk.data[2].SetValue(row, entry.pending ? Value(LogicalType::VARCHAR) : Value(entry.answer));
			chunk.data[3].SetValue(row, entry.pending ? Value(LogicalType::DOUBLE)  : Value(entry.elapsed_seconds));
			chunk.data[4].SetValue(row, Value(entry.sql_expert_model));
			chunk.data[5].SetValue(row, Value(entry.coordinator_model));
			chunk.data[6].SetValue(row, Value(entry.stats));
			row++;
			if (row == STANDARD_VECTOR_SIZE) {
				chunk.SetCardinality(row);
				collection->Append(chunk);
				chunk.Reset();
				row = 0;
			}
		}
		if (row > 0) {
			chunk.SetCardinality(row);
			collection->Append(chunk);
		}
	}

	return make_uniq<ColumnDataRef>(std::move(collection), names);
}

// ---------------------------------------------------------------------------
// Model registry and path helpers (shared by download function and shell command)
// ---------------------------------------------------------------------------

struct ModelCatalogEntry {
	const char *name;
	const char *description;
	double      size_gb;
	const char *url;
};

// Catalog of known models. Used by both KNOWN_MODELS lookup and FROM llama_models.
static const ModelCatalogEntry MODEL_CATALOG[] = {
    // General-purpose instruct models (small/fast)
    {"qwen2.5-0.5b",
     "Qwen 2.5 0.5B Instruct — tiny, fast, general purpose",
     0.4,
     "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"},
    {"qwen2.5-1.5b",
     "Qwen 2.5 1.5B Instruct — small, general purpose",
     1.0,
     "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"},
    {"qwen2.5-3b",
     "Qwen 2.5 3B Instruct — good balance of size and quality",
     2.0,
     "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"},
    {"smollm2-1.7b",
     "SmolLM2 1.7B Instruct — compact, efficient general model",
     1.1,
     "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"},
    // SQL-specialised models (recommended for .claude)
    {"sqlcoder-7b-2",
     "SQLCoder 7B-2 — Defog SQL specialist, Q4_K_M (~4GB), fast on Apple Silicon (RECOMMENDED)",
     4.1,
     "https://huggingface.co/MaziyarPanahi/sqlcoder-7b-2-GGUF/resolve/main/sqlcoder-7b-2.Q4_K_M.gguf"},
    {"sqlcoder-7b",
     "SQLCoder 7B — Defog text-to-SQL model, Q4_K_M (~4.4GB)",
     4.4,
     "https://huggingface.co/TheBloke/sqlcoder-7B-GGUF/resolve/main/sqlcoder-7b.Q4_K_M.gguf"},
    {"duckdb-nsql-7b",
     "DuckDB-NSQL 7B — fine-tuned on 200k DuckDB SQL pairs by MotherDuck (q8_0, ~7.5GB — needs 8GB+ free RAM)",
     7.5,
     "https://huggingface.co/motherduckdb/DuckDB-NSQL-7B-v0.1-GGUF/resolve/main/DuckDB-NSQL-7B-v0.1-q8_0.gguf"},
};

static constexpr idx_t MODEL_CATALOG_COUNT = sizeof(MODEL_CATALOG) / sizeof(MODEL_CATALOG[0]);

// Fast lookup map built from MODEL_CATALOG — used by download + SET llama_model_path.
static const case_insensitive_map_t<string> KNOWN_MODELS = []() {
	case_insensitive_map_t<string> m;
	for (idx_t i = 0; i < MODEL_CATALOG_COUNT; i++) {
		m[MODEL_CATALOG[i].name] = MODEL_CATALOG[i].url;
	}
	return m;
}();

static string GetModelsDataDir(FileSystem &fs) {
	return fs.GetHomeDirectory() + "/.duckdb/extension_data/models";
}

static string GetModelsModelsDir(FileSystem &fs) {
	return GetModelsDataDir(fs) + "/models";
}

static string GetModelsSetupPath(FileSystem &fs) {
	return GetModelsDataDir(fs) + "/setup.sql";
}

// Read entire file into a string; returns "" if file doesn't exist.
static string ReadFileContent(const string &path) {
	FILE *f = fopen(path.c_str(), "r");
	if (!f) return "";
	fseek(f, 0, SEEK_END);
	long sz = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (sz <= 0) { fclose(f); return ""; }
	string content(static_cast<size_t>(sz), '\0');
	fread(&content[0], 1, static_cast<size_t>(sz), f);
	fclose(f);
	return content;
}

// Strip lines beginning with "--" (SQL comments) for change-detection comparisons.
static string StripSQLComments(const string &sql) {
	string result;
	size_t i = 0;
	while (i < sql.size()) {
		size_t nl = sql.find('\n', i);
		if (nl == string::npos) nl = sql.size();
		string line = sql.substr(i, nl - i);
		if (!(line.size() >= 2 && line[0] == '-' && line[1] == '-')) {
			result += line + "\n";
		}
		i = nl + 1;
	}
	return result;
}

// Resolve a model path from SET llama_model_path:
//   - Known short name (e.g. "qwen2.5-1.5b") → ~/.duckdb/.../models/<filename>.gguf
//   - "~/" prefix                             → expanded via fs.GetHomeDirectory()
//   - Anything else                           → used as-is
static string ResolveModelPath(const string &setting, FileSystem &fs) {
	if (setting.empty()) return setting;

	// Short name from registry?
	auto it = KNOWN_MODELS.find(setting);
	if (it != KNOWN_MODELS.end()) {
		auto slash = it->second.rfind('/');
		string filename = (slash != string::npos) ? it->second.substr(slash + 1) : it->second;
		return GetModelsModelsDir(fs) + "/" + filename;
	}

	// Expand leading "~/".
	if (setting.size() >= 2 && setting[0] == '~' && setting[1] == '/') {
		return fs.GetHomeDirectory() + setting.substr(1);
	}

	return setting;
}

// Resolve 'auto' worker style from model path (checks for known SQL-only model names).
static string ResolveWorkerStyle(const string &style, const string &model_path) {
	if (style != "auto") return style;
	string lower = StringUtil::Lower(model_path);
	if (lower.find("sqlcoder") != string::npos || lower.find("nsql") != string::npos) {
		return "sqlcoder";
	}
	return "chat";
}

// Recursively create directories (like mkdir -p).
static bool MakeDirectories(const string &path) {
	struct stat st;
	if (stat(path.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
	auto parent = path.substr(0, path.rfind('/'));
	if (!parent.empty() && parent != path && !MakeDirectories(parent)) return false;
	return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

// ---------------------------------------------------------------------------
// Shell dot-command callback: .claude <question...>
// Dispatches to the backend selected by SET claude_backend = 'claude' | 'llama'
// ---------------------------------------------------------------------------

#ifdef MODELS_SHELL_EXT
static ShellCommandResult ModelsShellCommand(DatabaseInstance &db, BaseShellState &shell_state,
                                             const vector<string> &args,
                                             optional_ptr<ShellCommandExtensionInfo> info) {
	if (args.size() < 2) {
		shell_state.ShellPrint("Usage: .claude <question>\n");
		shell_state.ShellPrint("       Ask a question about your DuckDB database.\n");
		shell_state.ShellPrint("       Run CALL setup_models() to configure a model.\n");
		return ShellCommandResult::PRINT_USAGE;
	}

	string question;
	for (idx_t i = 1; i < args.size(); i++) {
		if (i > 1) question += ' ';
		question += args[i];
	}

	auto &state = static_cast<ModelsExtensionState &>(*info);

	// Read settings each invocation so SET takes effect immediately.
	// Backend is inferred from sql_expert_model:
	//   empty          → not configured (prompt to run setup_models)
	//   "claude"       → Claude CLI subprocess
	//   anything else  → local llama.cpp GGUF
	Connection settings_conn(db);
	auto &ctx = *settings_conn.context;
	auto read_str = [&](const string &key, const string &def = "") -> string {
		Value v; return ctx.TryGetCurrentSetting(key, v) ? v.ToString() : def;
	};
	auto read_int = [&](const string &key, int32_t def) -> int32_t {
		Value v; return ctx.TryGetCurrentSetting(key, v) ? v.GetValue<int32_t>() : def;
	};
	auto read_dbl = [&](const string &key, double def) -> double {
		Value v; return ctx.TryGetCurrentSetting(key, v) ? v.GetValue<double>() : def;
	};

	string sql_expert = read_str("sql_expert_model");

	if (sql_expert.empty()) {
		shell_state.ShellPrintError(
		    "Error: no model configured.\n"
		    "Run one of:\n"
		    "  CALL setup_models();              -- local llama models (privacy-first)\n"
		    "  CALL setup_models(claude:=true);  -- Claude CLI (requires claude installed)\n");
		return ShellCommandResult::SUCCESS;
	}

	if (sql_expert != "claude") {
		auto   &fs         = FileSystem::GetFileSystem(ctx);
		string  model_path = ResolveModelPath(sql_expert, fs);
		int32_t n_ctx      = read_int("llama_context_size", 8192);
		int32_t n_gpu      = read_int("llama_gpu_layers",   -1);
		string  coord_raw  = read_str("coordinator_model");
		string  coord_path = coord_raw.empty() || coord_raw == "claude" ? "" : ResolveModelPath(coord_raw, fs);
		double  time_budget  = read_dbl("coordinator_time_budget", 60.0);
		int32_t coord_rounds = read_int("coordinator_max_rounds",   12);
		string  worker_style = ResolveWorkerStyle(read_str("sql_expert_model_style", "auto"), model_path);

		if (!state.llama.IsLoaded() || model_path != state.llama_model_path) {
			shell_state.ShellPrint(StringUtil::Format("Loading worker model '%s'...\n", model_path));
			state.llama = LlamaState();
			if (!state.llama.Load(model_path, n_ctx, n_gpu, shell_state)) {
				return ShellCommandResult::SUCCESS;
			}
			state.llama_model_path   = model_path;
			state.llama_n_ctx        = n_ctx;
			state.llama_n_gpu_layers = n_gpu;
		}

		// Load llama coordinator model if configured.
		if (!coord_path.empty() &&
		    (!state.coordinator_llama.IsLoaded() || coord_path != state.coordinator_model_path)) {
			shell_state.ShellPrint(StringUtil::Format("Loading coordinator model '%s'...\n", coord_path));
			state.coordinator_llama = LlamaState();
			if (!state.coordinator_llama.Load(coord_path, n_ctx, n_gpu, shell_state)) {
				shell_state.ShellPrintError("Warning: coordinator model failed to load; running without coordinator.\n");
				coord_path.clear();
			} else {
				state.coordinator_model_path = coord_path;
			}
		}

		idx_t next_id = state.history.empty() ? 1 : state.history.back().id + 1;
		state.history.push_back({next_id, question, "", -1.0, true, sql_expert, coord_raw});

		auto t0 = std::chrono::steady_clock::now();
		Connection conn(db);
		string answer;
		idx_t width = shell_state.GetTerminalWidth();
		InteractionStats istats;

		if (coord_raw == "claude") {
			// Claude coordinator + llama worker
			WorkerFn worker_fn = [&](const string &q) -> string {
				return RunLlamaLoop(*conn.context, shell_state, state.llama, q, width,
				                    worker_style == "sqlcoder" ? SQLCODER_SYSTEM_PROMPT : "",
				                    worker_style == "sqlcoder");
			};
			answer = RunModelsCoordinatorLoop(*conn.context, shell_state, worker_fn,
			                                 question, width, coord_rounds, istats);
		} else if (!coord_path.empty() && state.coordinator_llama.IsLoaded()) {
			// Llama coordinator + llama worker
			answer = RunCoordinatorLoop(*conn.context, shell_state,
			                           state.coordinator_llama, state.llama,
			                           question, width, time_budget, coord_rounds, worker_style);
		} else {
			// No coordinator — direct llama loop
			answer = RunLlamaLoop(*conn.context, shell_state, state.llama, question, width);
		}
		double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

		auto &entry = state.history.back();
		entry.answer          = answer;
		entry.elapsed_seconds = elapsed;
		entry.pending         = false;
		entry.stats           = istats.ToJSON();

		// RunLlamaLoop already streamed tokens via ShellPrint — no re-print needed.
		shell_state.ShellPrint("\n");
		return ShellCommandResult::SUCCESS;
	}

	// sql_expert_model == "claude": Claude CLI subprocess backend.
	// Dispatch based on coordinator_model:
	//   "claude" or ""  → Claude coordinator subprocess + Claude worker subprocess (or direct)
	//   llama GGUF      → llama coordinator + Claude worker (RunCoordinatorLoopWithWorker)
	idx_t next_id = state.history.empty() ? 1 : state.history.back().id + 1;
	string coord_for_history = read_str("coordinator_model");
	state.history.push_back({next_id, question, "", -1.0, true, sql_expert, coord_for_history});

	auto t0 = std::chrono::steady_clock::now();
	Connection conn(db);
	string answer;
	InteractionStats istats;
	{
		string coord_raw2 = read_str("coordinator_model");
		if (coord_raw2 == "claude") {
			// Claude coordinator + Claude worker (two subprocesses)
			idx_t width = shell_state.GetTerminalWidth();
			WorkerFn worker_fn = [&](const string &q) -> string {
				return RunModelsSubprocessLoop(*conn.context, shell_state, q, width, istats);
			};
			answer = RunModelsCoordinatorLoop(*conn.context, shell_state, worker_fn,
			                                 question, width,
			                                 read_int("coordinator_max_rounds", 12), istats);
			goto record_entry;
		}
	}

	{
		string coord_raw = read_str("coordinator_model");
		if (!coord_raw.empty() && coord_raw != "claude") {
			// Llama coordinator + Claude worker
			auto   &fs         = FileSystem::GetFileSystem(ctx);
			string  coord_path = ResolveModelPath(coord_raw, fs);
			int32_t n_ctx      = read_int("llama_context_size", 8192);
			int32_t n_gpu      = read_int("llama_gpu_layers",   -1);
			double  time_budget  = read_dbl("coordinator_time_budget", 60.0);
			int32_t coord_rounds = read_int("coordinator_max_rounds",   12);

			if (!state.coordinator_llama.IsLoaded() || coord_path != state.coordinator_model_path) {
				shell_state.ShellPrint(StringUtil::Format("Loading coordinator model '%s'...\n", coord_path));
				state.coordinator_llama = LlamaState();
				if (!state.coordinator_llama.Load(coord_path, n_ctx, n_gpu, shell_state)) {
					shell_state.ShellPrintError("Warning: coordinator model failed to load; falling back to Claude direct.\n");
					coord_path.clear();
				} else {
					state.coordinator_model_path = coord_path;
				}
			}

			if (!coord_path.empty() && state.coordinator_llama.IsLoaded()) {
				idx_t width = shell_state.GetTerminalWidth();
				WorkerFn worker_fn = [&](const string &q) -> string {
					return RunModelsSubprocessLoop(*conn.context, shell_state, q, width, istats);
				};
				answer = RunCoordinatorLoopWithWorker(*conn.context, shell_state,
				                                     state.coordinator_llama, worker_fn,
				                                     question, width,
				                                     time_budget, coord_rounds);
				goto record_entry;
			}
		}
	}

	answer = RunModelsSubprocessLoop(*conn.context, shell_state, question,
	                                 shell_state.GetTerminalWidth(), istats);

record_entry:
	{
		double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		auto &entry = state.history.back();
		entry.answer          = answer;
		entry.elapsed_seconds = elapsed;
		entry.pending         = false;
		entry.stats           = istats.ToJSON();
	}

	if (!answer.empty()) {
		shell_state.ShellPrint("\n" + answer + "\n");
	}
	return ShellCommandResult::SUCCESS;
}
#endif // MODELS_SHELL_EXT

// ---------------------------------------------------------------------------
// Extension entry points
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// llama_download_model(url_or_name) table function
//
// Downloads a GGUF model to ~/.duckdb/extension_data/models/models/.
// Accepts a full HTTPS URL or a short name from the built-in registry.
// Requires httpfs (auto-loaded). Idempotent — skips download if file exists.
//
// Usage:
//   CALL llama_download_model('qwen2.5-1.5b');
//   CALL llama_download_model('https://huggingface.co/.../model.gguf');
// ---------------------------------------------------------------------------


struct DownloadModelBindData : public FunctionData {
	string url;
	string model_name; // filename (e.g. "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
	string local_path;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<DownloadModelBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		return url == other_p.Cast<DownloadModelBindData>().url;
	}
};

struct DownloadModelGlobalState : public GlobalTableFunctionState {
	int64_t bytes_downloaded = 0;
	bool    returned         = false;
};

// ---------------------------------------------------------------------------
// Parallel model download helpers
// ---------------------------------------------------------------------------

// One BaseExecutorTask per 100 MB chunk. Uses HTTP Range header and pwrite()
// to write the chunk at the correct offset in the pre-allocated output file.
struct ChunkDownloadTask : public BaseExecutorTask {
	DatabaseInstance      &db;
	HTTPUtil              &http_util;
	const string          &url;
	int64_t                range_start;
	int64_t                range_end;
	int                    fd;
	std::atomic<int64_t>  &total_downloaded;

	ChunkDownloadTask(TaskExecutor &executor, DatabaseInstance &db_p, HTTPUtil &http_util_p,
	                  const string &url_p, int64_t rs, int64_t re,
	                  int fd_p, std::atomic<int64_t> &downloaded_p)
	    : BaseExecutorTask(executor), db(db_p), http_util(http_util_p), url(url_p),
	      range_start(rs), range_end(re), fd(fd_p), total_downloaded(downloaded_p) {
	}

	void ExecuteTask() override {
		auto params = http_util.InitializeParameters(db, url);
		params->timeout = 3600;
		HTTPHeaders headers(db);
		headers.Insert("Range", "bytes=" + to_string(range_start) + "-" + to_string(range_end));

		int64_t write_offset = range_start;
		auto content_h = [&](const_data_ptr_t data, idx_t len) -> bool {
			::pwrite(fd, data, static_cast<size_t>(len), static_cast<off_t>(write_offset));
			write_offset += static_cast<int64_t>(len);
			total_downloaded.fetch_add(static_cast<int64_t>(len), std::memory_order_relaxed);
			return true;
		};
		auto response_h = [](const HTTPResponse &) -> bool { return true; };

		GetRequestInfo req(url, headers, *params, response_h, content_h);
		auto resp = http_util.Request(req);
		if (!resp || !resp->Success()) {
			throw IOException("Chunk [%lld-%lld] failed: %s", range_start, range_end,
			                  resp ? resp->GetError().c_str() : "no response");
		}
	}

	string TaskType() const override { return "ModelChunkDownload"; }
};

static unique_ptr<FunctionData> DownloadModelBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT};
	names        = {"model_name", "local_path", "bytes_downloaded"};

	auto &fs    = FileSystem::GetFileSystem(context);
	auto result = make_uniq<DownloadModelBindData>();
	string arg  = input.inputs[0].GetValue<string>();

	if (StringUtil::StartsWith(arg, "http://") || StringUtil::StartsWith(arg, "https://")) {
		result->url = arg;
	} else {
		auto it = KNOWN_MODELS.find(arg);
		if (it == KNOWN_MODELS.end()) {
			string known;
			for (auto &kv : KNOWN_MODELS) known += " " + kv.first;
			throw InvalidInputException(
			    "Unknown model name '%s'. Use a full HTTPS URL or one of:%s", arg, known);
		}
		result->url = it->second;
	}

	auto slash = result->url.rfind('/');
	result->model_name = (slash != string::npos) ? result->url.substr(slash + 1) : "model.gguf";
	result->local_path = GetModelsModelsDir(fs) + "/" + result->model_name;
	return result;
}

static unique_ptr<GlobalTableFunctionState> DownloadModelInitGlobal(ClientContext &context,
                                                                      TableFunctionInitInput &input) {
	auto &bind  = input.bind_data->Cast<DownloadModelBindData>();
	auto  state = make_uniq<DownloadModelGlobalState>();

	auto &fs          = FileSystem::GetFileSystem(context);
	string models_dir = GetModelsModelsDir(fs);
	if (!MakeDirectories(models_dir)) {
		throw IOException("Failed to create models directory: %s", models_dir);
	}

	// Already downloaded — skip.
	struct stat st;
	if (stat(bind.local_path.c_str(), &st) == 0 && st.st_size > 0) {
		fprintf(stderr, "Model already exists: %s\n", bind.local_path.c_str());
		state->bytes_downloaded = static_cast<int64_t>(st.st_size);
		return state;
	}

	if (!ExtensionHelper::TryAutoLoadExtension(context, "httpfs")) {
		throw IOException("httpfs extension is required for model download but could not be loaded");
	}

	auto &http_util    = HTTPUtil::Get(*context.db);
	auto &db           = *context.db;
	HTTPHeaders base_headers(db);

	// ---------------------------------------------------------------------------
	// HEAD: get file size and check if server supports range requests.
	// ---------------------------------------------------------------------------
	int64_t total_size = -1;
	bool    range_ok   = false;
	{
		auto hparams = http_util.InitializeParameters(db, bind.url);
		hparams->timeout = 60;
		HeadRequestInfo head_req(bind.url, base_headers, *hparams);
		head_req.try_request = true;
		auto head_resp = http_util.Request(head_req);
		if (head_resp && head_resp->Success()) {
			if (head_resp->HasHeader("Content-Length")) {
				try { total_size = std::stoll(head_resp->GetHeaderValue("Content-Length")); }
				catch (...) {}
			}
			if (head_resp->HasHeader("Accept-Ranges")) {
				range_ok = (head_resp->GetHeaderValue("Accept-Ranges") == "bytes");
			}
		}
	}

	// ---------------------------------------------------------------------------
	// Parallel chunked download via DuckDB TaskExecutor (100 MB chunks).
	// Falls back to sequential if range requests are unsupported or size unknown.
	// ---------------------------------------------------------------------------
	static constexpr int64_t CHUNK_SIZE = 100LL * 1024 * 1024; // 100 MB

	if (range_ok && total_size > 0) {
		int n_chunks = static_cast<int>((total_size + CHUNK_SIZE - 1) / CHUNK_SIZE);
		fprintf(stderr, "Downloading %s (%.1f MB, %d chunks of 100 MB)...\n",
		        bind.model_name.c_str(), total_size / 1e6, n_chunks);

		// Pre-allocate the output file so pwrite() can write at arbitrary offsets.
		int fd = ::open(bind.local_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (fd < 0) {
			throw IOException("Cannot open '%s': %s", bind.local_path, strerror(errno));
		}
		if (::ftruncate(fd, static_cast<off_t>(total_size)) != 0) {
			::close(fd);
			throw IOException("Cannot pre-allocate '%s': %s", bind.local_path, strerror(errno));
		}

		std::atomic<int64_t> total_downloaded {0};

		// Schedule one ChunkDownloadTask per chunk into DuckDB's thread pool.
		TaskExecutor executor(context);
		for (int i = 0; i < n_chunks; i++) {
			int64_t rs = static_cast<int64_t>(i) * CHUNK_SIZE;
			int64_t re = std::min(rs + CHUNK_SIZE - 1, total_size - 1);
			executor.ScheduleTask(make_uniq<ChunkDownloadTask>(
			    executor, db, http_util, bind.url, rs, re, fd, total_downloaded));
		}

		// Progress display in a lightweight background thread while WorkOnTasks() runs.
		std::atomic<bool> progress_done {false};
		std::thread progress_thread([&]() {
			while (!progress_done.load(std::memory_order_relaxed)) {
				int64_t done = total_downloaded.load(std::memory_order_relaxed);
				fprintf(stderr, "\r  %.1f / %.1f MB (%.0f%%)   ",
				        done / 1e6, total_size / 1e6, 100.0 * done / total_size);
				fflush(stderr);
				std::this_thread::sleep_for(std::chrono::milliseconds(250));
			}
		});

		// Block until all chunks finish; propagates any chunk errors.
		try {
			executor.WorkOnTasks();
		} catch (...) {
			progress_done.store(true);
			progress_thread.join();
			::close(fd);
			::remove(bind.local_path.c_str());
			throw;
		}
		progress_done.store(true);
		progress_thread.join();
		::close(fd);
		fprintf(stderr, "\r  %.1f MB — done.%*s\n", total_size / 1e6, 20, "");
		state->bytes_downloaded = total_size;

	} else {
		// ---------------------------------------------------------------------------
		// Sequential fallback.
		// ---------------------------------------------------------------------------
		if (!range_ok) {
			fprintf(stderr, "Server does not support range requests — downloading sequentially.\n");
		}
		auto params = http_util.InitializeParameters(context, bind.url);
		params->timeout = 3600;

		FILE *f = fopen(bind.local_path.c_str(), "wb");
		if (!f) {
			throw IOException("Cannot open '%s' for writing: %s", bind.local_path, strerror(errno));
		}
		int64_t downloaded = 0;
		auto response_handler = [&total_size](const HTTPResponse &resp) -> bool {
			if (resp.HasHeader("Content-Length")) {
				try { total_size = std::stoll(resp.GetHeaderValue("Content-Length")); } catch (...) {}
			}
			return true;
		};
		auto content_handler = [&](const_data_ptr_t data, idx_t len) -> bool {
			fwrite(data, 1, static_cast<size_t>(len), f);
			downloaded += static_cast<int64_t>(len);
			if (total_size > 0) {
				fprintf(stderr, "\rDownloading %s: %.1f / %.1f MB (%.0f%%)   ",
				        bind.model_name.c_str(), downloaded / 1e6, total_size / 1e6,
				        100.0 * downloaded / total_size);
			} else {
				fprintf(stderr, "\rDownloading %s: %.1f MB   ", bind.model_name.c_str(), downloaded / 1e6);
			}
			fflush(stderr);
			return true;
		};
		GetRequestInfo req(bind.url, base_headers, *params, response_handler, content_handler);
		auto response = http_util.Request(req);
		fclose(f);
		fprintf(stderr, "\n");
		if (!response || !response->Success()) {
			remove(bind.local_path.c_str());
			throw IOException("Model download failed: %s",
			                  response ? response->GetError() : "no response");
		}
		state->bytes_downloaded = downloaded;
	}

	return state;
}

static void DownloadModelScan(ClientContext & /*context*/, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind  = data_p.bind_data->Cast<DownloadModelBindData>();
	auto &state = data_p.global_state->Cast<DownloadModelGlobalState>();
	if (state.returned) return;
	state.returned = true;
	output.SetCardinality(1);
	output.data[0].SetValue(0, Value(bind.model_name));
	output.data[1].SetValue(0, Value(bind.local_path));
	output.data[2].SetValue(0, Value::BIGINT(state.bytes_downloaded));
}

// ---------------------------------------------------------------------------
// llama_instructions — static task-hint table queryable by the model
// ---------------------------------------------------------------------------

struct LlamaInstruction {
	const char *task;
	const char *hint;
};

static const LlamaInstruction LLAMA_INSTRUCTIONS[] = {
    {"describe-table",
     "DESCRIBE <table> — column names and types.\n"
     "SUMMARIZE <table> — per-column stats: min/max/avg/count/null%/distinct.\n"
     "FROM duckdb_columns() WHERE table_name = '<table>' — programmatic access."},

    {"list-tables",
     "FROM duckdb_tables() — all tables with database_name, schema_name, table_name, estimated_size.\n"
     "FROM duckdb_tables() WHERE database_name = '<db>' — filter by attached database name.\n"
     "FROM duckdb_summary('schema') — tables with their column list.\n"
     "FROM duckdb_views() — all views.\n"
     "NOTE: do NOT use information_schema.tables to filter by database name; use duckdb_tables() instead."},

    {"profile-data",
     "SUMMARIZE <table> — per-column min/max/avg/count/null%/distinct in one shot.\n"
     "SELECT COUNT(*) FILTER (WHERE col IS NULL) AS nulls, COUNT(DISTINCT col) AS distinct_count FROM <table>."},

    {"query-performance",
     "EXPLAIN ANALYZE <query> — execution plan with actual timing and row counts.\n"
     "EXPLAIN <query> — logical plan without executing.\n"
     "SET threads = N; — control parallelism."},

    {"indexes",
     "FROM duckdb_indexes() — all indexes: table_name, index_name, column_names, is_unique.\n"
     "CREATE INDEX idx ON <table>(col); — add an index (write, not available here)."},

    {"constraints",
     "FROM duckdb_constraints() — all constraints: table_name, constraint_type (PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK), column_names."},

    {"functions",
     "FROM duckdb_functions() WHERE function_name ILIKE '%<keyword>%' — search by name.\n"
     "FROM duckdb_functions() WHERE function_type = 'macro' — user-defined macros.\n"
     "FROM duckdb_functions() WHERE function_type = 'aggregate' — aggregate functions."},

    {"extensions",
     "FROM duckdb_extensions() WHERE loaded = true — currently loaded extensions.\n"
     "FROM duckdb_extensions() — all known extensions with version and description."},

    {"file-formats",
     "SELECT * FROM read_csv('file.csv', auto_detect=true)\n"
     "SELECT * FROM read_parquet('file.parquet') — or glob: 'dir/*.parquet'\n"
     "SELECT * FROM read_json('file.json', auto_detect=true)\n"
     "SELECT * FROM read_json_auto('file.json')\n"
     "COPY <table> TO 'out.parquet' (FORMAT PARQUET);"},

    {"aggregate-stats",
     "COUNT(*), COUNT(DISTINCT col), AVG(col), SUM(col), MIN(col), MAX(col), STDDEV(col)\n"
     "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) AS median\n"
     "MODE() WITHIN GROUP (ORDER BY col)\n"
     "APPROX_COUNT_DISTINCT(col) — fast distinct estimate."},

    {"window-functions",
     "ROW_NUMBER() OVER (PARTITION BY grp ORDER BY col)\n"
     "RANK() / DENSE_RANK() OVER (...)\n"
     "LAG(col, 1) OVER (ORDER BY col) — previous row value\n"
     "SUM(col) OVER (PARTITION BY grp ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) — running total"},

    {"date-time",
     "DATE_TRUNC('month', col) — truncate to period (year/month/week/day/hour)\n"
     "DATE_DIFF('day', start_col, end_col) — difference in units\n"
     "STRFTIME(col, '%Y-%m-%d') — format a date\n"
     "NOW(), TODAY(), CURRENT_DATE\n"
     "col AT TIME ZONE 'UTC' — timezone conversion"},

    {"string-ops",
     "LOWER(col), UPPER(col), TRIM(col), LENGTH(col)\n"
     "col ILIKE '%pattern%' — case-insensitive LIKE\n"
     "REGEXP_MATCHES(col, 'pattern') — regex filter\n"
     "REGEXP_EXTRACT(col, '(group)') — capture group\n"
     "STRING_SPLIT(col, ',') — split to list\n"
     "LIST_JOIN(list_col, ', ') — join list to string"},

    {"json",
     "col->>'$.key' — extract JSON field as text\n"
     "col->'$.key' — extract JSON field as JSON value\n"
     "json_extract_string(col, '$.key')\n"
     "UNNEST(json_extract(col, '$.array')) — expand JSON array to rows\n"
     "FROM read_json_auto('file.json')"},
};

static constexpr idx_t LLAMA_INSTRUCTIONS_COUNT =
    sizeof(LLAMA_INSTRUCTIONS) / sizeof(LLAMA_INSTRUCTIONS[0]);

// ---------------------------------------------------------------------------
// llama_error_hints — common mistakes with correct usage examples
// ---------------------------------------------------------------------------

struct LlamaErrorHint {
	const char *error_pattern;   // keyword to ILIKE-match against error messages
	const char *wrong_usage;     // what the model did wrong
	const char *correct_usage;   // the correct SQL to use instead
};

static const LlamaErrorHint LLAMA_ERROR_HINTS[] = {
    {"No function matches.*duckdb_tables",
     "FROM duckdb_tables('mydb')",
     "FROM duckdb_tables() WHERE database_name = 'mydb'"},

    {"No function matches.*duckdb_columns",
     "FROM duckdb_columns('mydb')",
     "FROM duckdb_columns() WHERE database_name = 'mydb' AND table_name = 'mytable'"},

    {"No function matches.*duckdb_schemas",
     "FROM duckdb_schemas('mydb')",
     "FROM duckdb_schemas() WHERE database_name = 'mydb'"},

    {"No function matches.*duckdb_views",
     "FROM duckdb_views('mydb')",
     "FROM duckdb_views() WHERE database_name = 'mydb'"},

    {"No function matches.*duckdb_indexes",
     "FROM duckdb_indexes('mydb')",
     "FROM duckdb_indexes() WHERE database_name = 'mydb'"},

    {"No function matches.*duckdb_functions",
     "FROM duckdb_functions('keyword')",
     "FROM duckdb_functions() WHERE function_name ILIKE '%keyword%'"},

    {"information_schema.*0 rows|0 rows.*information_schema",
     "SELECT * FROM information_schema.tables WHERE table_schema = 'mydb'",
     "FROM duckdb_tables() WHERE database_name = 'mydb'"},

    // In DuckDB information_schema, table_schema is the schema (e.g. 'main'),
    // NOT the database name.  Use table_catalog for the database/catalog name.
    // Hint is phrased as a ready-to-run ASK_SQL so the coordinator can use it directly.
    {"information_schema",
     "WHERE table_schema = 'mydb'  -- wrong: table_schema is the schema name, not the database",
     "ASK_SQL: FROM duckdb_tables() WHERE database_name = 'mydb'\n"
     "-- OR: SELECT table_name FROM information_schema.tables WHERE table_catalog = 'mydb'\n"
     "-- table_catalog = database name (e.g. 'dd'), table_schema = schema (e.g. 'main')"},

    {"Catalog.*not found|database.*not found",
     "FROM mytable  -- when the table is in an attached database",
     "FROM mydb.main.mytable  -- qualify with database.schema.table"},

    {"Column.*not found|does not have a column",
     "SELECT bad_col FROM t",
     "Run: DESCRIBE t  -- to see actual column names\n"
     "Common mistakes:\n"
     "  duckdb_tables()  → table_name, database_name, schema_name, column_count, estimated_size\n"
     "                     (NO create_time, last_analyzed, or 'name' column)\n"
     "  duckdb_columns() → column_name, table_name, database_name, data_type, column_index\n"
     "  duckdb_schemas() → schema_name, database_name  (NOT 'name')"},

    {"Binder.*No table.*function.*match|table function.*no.*overload",
     "FROM duckdb_summary('wrong_topic')",
     "FROM duckdb_summary()  -- valid topics: 'schema', 'context', 'all'"},

    {"duckdb_summary|Invalid.*topic",
     "FROM duckdb_summary('database') WHERE database_name = 'mydb'",
     "FROM duckdb_summary('context')  -- shows attached databases; valid topics: 'schema', 'context', 'all'\n"
     "-- duckdb_summary does NOT accept a database_name filter; use duckdb_tables() for that"},
};

static constexpr idx_t LLAMA_ERROR_HINTS_COUNT =
    sizeof(LLAMA_ERROR_HINTS) / sizeof(LLAMA_ERROR_HINTS[0]);

// Scan LLAMA_ERROR_HINTS for entries whose pattern matches the error text.
// Returns formatted hint lines ready to append to context, or empty string.
string MatchErrorHints(const string &error_text) {
	string result;
	for (idx_t i = 0; i < LLAMA_ERROR_HINTS_COUNT; i++) {
		try {
			std::regex re(LLAMA_ERROR_HINTS[i].error_pattern,
			              std::regex::icase | std::regex::ECMAScript);
			if (std::regex_search(error_text, re)) {
				result += string("Hint — instead of: ") + LLAMA_ERROR_HINTS[i].wrong_usage + "\n";
				result += string("       use:        ") + LLAMA_ERROR_HINTS[i].correct_usage + "\n";
			}
		} catch (...) {
			// Malformed regex — skip.
		}
	}
	return result;
}

struct LlamaInstructionsGlobalState : public GlobalTableFunctionState {
	idx_t row = 0;
};

static unique_ptr<FunctionData> LlamaInstructionsBind(ClientContext & /*context*/,
                                                      TableFunctionBindInput & /*input*/,
                                                      vector<LogicalType> &return_types,
                                                      vector<string> &names) {
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
	names        = {"task", "hint"};
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> LlamaInstructionsInitGlobal(ClientContext & /*context*/,
                                                                        TableFunctionInitInput & /*input*/) {
	return make_uniq<LlamaInstructionsGlobalState>();
}

static void LlamaInstructionsScan(ClientContext & /*context*/, TableFunctionInput &data_p, DataChunk &output) {
	auto &gstate = data_p.global_state->Cast<LlamaInstructionsGlobalState>();
	idx_t count  = 0;
	while (gstate.row < LLAMA_INSTRUCTIONS_COUNT && count < STANDARD_VECTOR_SIZE) {
		auto &instr = LLAMA_INSTRUCTIONS[gstate.row];
		output.data[0].SetValue(count, Value(instr.task));
		output.data[1].SetValue(count, Value(instr.hint));
		gstate.row++;
		count++;
	}
	output.SetCardinality(count);
}

// Table function + replacement scan for llama_error_hints

struct LlamaErrorHintsGlobalState : public GlobalTableFunctionState {
	idx_t row = 0;
};

static unique_ptr<FunctionData> LlamaErrorHintsBind(ClientContext & /*context*/,
                                                    TableFunctionBindInput & /*input*/,
                                                    vector<LogicalType> &return_types,
                                                    vector<string> &names) {
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	names        = {"error_pattern", "wrong_usage", "correct_usage"};
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> LlamaErrorHintsInitGlobal(ClientContext & /*context*/,
                                                                      TableFunctionInitInput & /*input*/) {
	return make_uniq<LlamaErrorHintsGlobalState>();
}

static void LlamaErrorHintsScan(ClientContext & /*context*/, TableFunctionInput &data_p, DataChunk &output) {
	auto &gstate = data_p.global_state->Cast<LlamaErrorHintsGlobalState>();
	idx_t count  = 0;
	while (gstate.row < LLAMA_ERROR_HINTS_COUNT && count < STANDARD_VECTOR_SIZE) {
		auto &h = LLAMA_ERROR_HINTS[gstate.row];
		output.data[0].SetValue(count, Value(h.error_pattern));
		output.data[1].SetValue(count, Value(h.wrong_usage));
		output.data[2].SetValue(count, Value(h.correct_usage));
		gstate.row++;
		count++;
	}
	output.SetCardinality(count);
}

static unique_ptr<TableRef> LlamaErrorHintsReplacementScan(ClientContext &context,
                                                            ReplacementScanInput &input,
                                                            optional_ptr<ReplacementScanData> /*data*/) {
	if (input.table_name != "llama_error_hints") {
		return nullptr;
	}
	vector<LogicalType> types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	vector<string>      names = {"error_pattern", "wrong_usage", "correct_usage"};
	auto collection = make_uniq<ColumnDataCollection>(context, types);
	DataChunk chunk;
	chunk.Initialize(context, types);
	for (idx_t i = 0; i < LLAMA_ERROR_HINTS_COUNT; i++) {
		chunk.data[0].SetValue(i, Value(LLAMA_ERROR_HINTS[i].error_pattern));
		chunk.data[1].SetValue(i, Value(LLAMA_ERROR_HINTS[i].wrong_usage));
		chunk.data[2].SetValue(i, Value(LLAMA_ERROR_HINTS[i].correct_usage));
	}
	chunk.SetCardinality(LLAMA_ERROR_HINTS_COUNT);
	collection->Append(chunk);
	return make_uniq<ColumnDataRef>(std::move(collection), names);
}

// ---------------------------------------------------------------------------
// llama_models — catalog of known models with download status
// ---------------------------------------------------------------------------

struct LlamaModelsGlobalState : public GlobalTableFunctionState {
	idx_t row = 0;
	string models_dir;
};

static unique_ptr<FunctionData> LlamaModelsBind(ClientContext & /*context*/,
                                                TableFunctionBindInput & /*input*/,
                                                vector<LogicalType> &return_types,
                                                vector<string> &names) {
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::DOUBLE,  LogicalType::BOOLEAN, LogicalType::VARCHAR};
	names        = {"name", "description", "size_gb", "downloaded", "url"};
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> LlamaModelsInitGlobal(ClientContext &context,
                                                                   TableFunctionInitInput & /*input*/) {
	auto gstate = make_uniq<LlamaModelsGlobalState>();
	gstate->models_dir = GetModelsModelsDir(FileSystem::GetFileSystem(context));
	return gstate;
}

static void LlamaModelsScan(ClientContext & /*context*/, TableFunctionInput &data_p, DataChunk &output) {
	auto &gstate = data_p.global_state->Cast<LlamaModelsGlobalState>();
	idx_t count  = 0;
	while (gstate.row < MODEL_CATALOG_COUNT && count < STANDARD_VECTOR_SIZE) {
		auto &m = MODEL_CATALOG[gstate.row];
		// Determine local filename from URL (last path component).
		string url(m.url);
		auto slash = url.rfind('/');
		string filename = (slash != string::npos) ? url.substr(slash + 1) : url;
		string local_path = gstate.models_dir + "/" + filename;
		struct stat st;
		bool downloaded = (stat(local_path.c_str(), &st) == 0 && st.st_size > 0);

		output.data[0].SetValue(count, Value(m.name));
		output.data[1].SetValue(count, Value(m.description));
		output.data[2].SetValue(count, Value(m.size_gb));
		output.data[3].SetValue(count, Value(downloaded));
		output.data[4].SetValue(count, Value(m.url));
		gstate.row++;
		count++;
	}
	output.SetCardinality(count);
}

static unique_ptr<TableRef> LlamaModelsReplacementScan(ClientContext &context,
                                                        ReplacementScanInput &input,
                                                        optional_ptr<ReplacementScanData> /*data*/) {
	if (input.table_name != "llama_models") {
		return nullptr;
	}
	auto &fs = FileSystem::GetFileSystem(context);
	string models_dir = GetModelsModelsDir(fs);

	vector<LogicalType> types = {LogicalType::VARCHAR, LogicalType::VARCHAR,
	                              LogicalType::DOUBLE,  LogicalType::BOOLEAN, LogicalType::VARCHAR};
	vector<string>      names = {"name", "description", "size_gb", "downloaded", "url"};
	auto collection = make_uniq<ColumnDataCollection>(context, types);
	DataChunk chunk;
	chunk.Initialize(context, types);

	for (idx_t i = 0; i < MODEL_CATALOG_COUNT; i++) {
		auto &m = MODEL_CATALOG[i];
		string url(m.url);
		auto slash = url.rfind('/');
		string filename = (slash != string::npos) ? url.substr(slash + 1) : url;
		string local_path = models_dir + "/" + filename;
		struct stat st;
		bool downloaded = (stat(local_path.c_str(), &st) == 0 && st.st_size > 0);

		chunk.data[0].SetValue(i, Value(m.name));
		chunk.data[1].SetValue(i, Value(m.description));
		chunk.data[2].SetValue(i, Value(m.size_gb));
		chunk.data[3].SetValue(i, Value(downloaded));
		chunk.data[4].SetValue(i, Value(m.url));
	}
	chunk.SetCardinality(MODEL_CATALOG_COUNT);
	collection->Append(chunk);
	return make_uniq<ColumnDataRef>(std::move(collection), names);
}

// ---------------------------------------------------------------------------
// BufferedShellState — captures ShellPrint output to a string buffer.
// Used by ask_models() to collect the answer without a real terminal.
// ---------------------------------------------------------------------------

struct BufferedShellState : public BaseShellState {
	string output;

	void ShellPrint(const string &str) override {
		output += str;
	}
	void ShellPrintError(const string &str) override {
		fprintf(stderr, "%s", str.c_str());
	}
	idx_t GetTerminalWidth() const override {
		return 120;
	}
};

// ---------------------------------------------------------------------------
// TODO: Unified model interface redesign
//
// The current extension hard-codes two roles (coordinator = llama/qwen,
// worker = llama/sqlcoder) and a separate code path for the Claude CLI
// subprocess.  A cleaner architecture would expose a single, composable
// interface from SQL:
//
//   CALL setup_models(
//       coordinator := 'claude',          -- 'claude' | path-to-gguf
//       worker      := 'sqlcoder',        -- 'claude' | path-to-gguf | 'none'
//       worker_style := 'sqlcoder'        -- 'chat' | 'sqlcoder'
//   );
//   SELECT * FROM ask_models('how many orders shipped last month?');
//
// With this design:
//   - coordinator = 'claude'  → use the existing Claude CLI subprocess loop
//   - coordinator = '/path/to/qwen.gguf' → in-process llama inference (current)
//   - worker = 'claude'       → delegate SQL sub-questions to a second Claude
//                               subprocess instead of a local GGUF model
//   - worker = 'none'         → coordinator handles everything directly
//                               (pure Claude or pure llama, no two-model split)
//
// This unifies RunModelsSubprocessLoop and RunCoordinatorLoop behind a single
// dispatch layer: each "model slot" is a ModelHandle that is either a
// LlamaState (GGUF) or a ClaudeSubprocess (CLI).  The coordinator/worker
// protocol (DELEGATE_SQL:, ASK_SQL:, FINAL_ANSWER:) stays the same; only
// the transport differs.
//
// Additional setup helpers to consider:
//   SET claude_coordinator = 'claude';
//   SET claude_worker      = '/models/sqlcoder.gguf';
//   SET claude_api_key     = '...';    -- for future direct-API support
//
// Fine-tuning note: a DuckDB-specific fine-tune of sqlcoder (or a smaller
// Qwen/Phi model) trained on DuckDB v1.5 syntax would slot in here as the
// worker GGUF and would dramatically improve SQL accuracy without changing
// the protocol.  See memory/MEMORY.md "Ideas" section.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ask_models(question) — SQL table function equivalent of .claude <question>
// ---------------------------------------------------------------------------

struct AskModelsBindData : public FunctionData {
	string                           question;
	shared_ptr<ModelsExtensionState> state;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AskModelsBindData>(*this);
	}
	bool Equals(const FunctionData &other) const override {
		return question == other.Cast<AskModelsBindData>().question;
	}
};

struct AskModelsGlobalState : public GlobalTableFunctionState {
	string  answer;
	idx_t   entry_id       = 0;
	double  elapsed        = 0.0;
	string  stats_json;
	bool    returned       = false;
};

static unique_ptr<FunctionData> AskModelsBind(ClientContext & /*context*/,
                                               TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types,
                                               vector<string> &names) {
	return_types   = {LogicalType::VARCHAR, LogicalType::VARCHAR};
	names          = {"answer", "stats"};
	auto bind      = make_uniq<AskModelsBindData>();
	bind->question = input.inputs[0].GetValue<string>();
	bind->state    = input.info->Cast<ModelsTableFunctionInfo>().state;
	return bind;
}

static unique_ptr<GlobalTableFunctionState> AskModelsInitGlobal(ClientContext &context,
                                                                 TableFunctionInitInput &input) {
	auto &bind  = input.bind_data->Cast<AskModelsBindData>();
	auto &state = bind.state;
	auto  gstate = make_uniq<AskModelsGlobalState>();

	auto read_str = [&](const string &key, const string &def = "") -> string {
		Value v; return context.TryGetCurrentSetting(key, v) ? v.ToString() : def;
	};
	auto read_int = [&](const string &key, int32_t def) -> int32_t {
		Value v; return context.TryGetCurrentSetting(key, v) ? v.GetValue<int32_t>() : def;
	};
	auto read_dbl = [&](const string &key, double def) -> double {
		Value v; return context.TryGetCurrentSetting(key, v) ? v.GetValue<double>() : def;
	};

	string sql_expert = read_str("sql_expert_model");

	if (sql_expert.empty()) {
		gstate->answer = "Error: no model configured. Run: CALL setup_models();";
		return gstate;
	}

	string coord_raw = read_str("coordinator_model");

	auto t_ask = std::chrono::steady_clock::now();
	idx_t ask_id = state->history.empty() ? 1 : state->history.back().id + 1;
	state->history.push_back({ask_id, bind.question, "", -1.0, true, sql_expert, coord_raw});

	InteractionStats istats;
	auto finish_history = [&](const string &ans) {
	    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ask).count();
	    auto &entry = state->history.back();
	    entry.answer          = ans;
	    entry.elapsed_seconds = elapsed;
	    entry.pending         = false;
	    entry.stats           = istats.ToJSON();
	    // Print the full answer directly so it isn't truncated by DuckDB's table renderer.
	    fprintf(stdout, "\n%s\n\n", ans.c_str());
	    fflush(stdout);
	    // Merge id and elapsed_seconds into stats JSON.
	    string base = entry.stats; // e.g. {"sql_total":1,...}
	    // Insert id and elapsed_seconds before the closing '}'.
	    auto pos = base.rfind('}');
	    string merged = (pos == string::npos) ? base
	        : base.substr(0, pos) +
	          StringUtil::Format(",\"id\":%llu,\"elapsed_seconds\":%.3f}", (unsigned long long)entry.id, elapsed);
	    // Fill gstate for the returned table row.
	    gstate->answer     = ans;
	    gstate->entry_id   = entry.id;
	    gstate->elapsed    = elapsed;
	    gstate->stats_json = merged;
	};

	if (sql_expert != "claude") {
		auto   &fs         = FileSystem::GetFileSystem(context);
		string  model_path = ResolveModelPath(sql_expert, fs);
		int32_t n_ctx      = read_int("llama_context_size", 8192);
		int32_t n_gpu      = read_int("llama_gpu_layers",   -1);
		string  coord_path = coord_raw.empty() || coord_raw == "claude" ? "" : ResolveModelPath(coord_raw, fs);
		double  time_budget  = read_dbl("coordinator_time_budget", 60.0);
		int32_t coord_rounds = read_int("coordinator_max_rounds",   12);
		string  worker_style = ResolveWorkerStyle(read_str("sql_expert_model_style", "auto"), model_path);

		if (model_path.empty()) {
			// sql_expert_model didn't resolve — fall back to first GGUF in managed dir.
			auto gguf_files = fs.Glob(GetModelsModelsDir(fs) + "/*.gguf");
			if (!gguf_files.empty()) {
				model_path = gguf_files[0].path;
			} else {
				gstate->answer = "Error: model path did not resolve. Run: CALL setup_models();";
				finish_history(gstate->answer);
				return gstate;
			}
		}

		if (!state->llama.IsLoaded() || model_path != state->llama_model_path) {
			BufferedShellState load_state;
			load_state.ShellPrint(StringUtil::Format("Loading worker model '%s'...\n", model_path));
			state->llama = LlamaState();
			if (!state->llama.Load(model_path, n_ctx, n_gpu, load_state)) {
				gstate->answer = load_state.output;
				finish_history(gstate->answer);
				return gstate;
			}
			state->llama_model_path   = model_path;
			state->llama_n_ctx        = n_ctx;
			state->llama_n_gpu_layers = n_gpu;
		}

		if (!coord_path.empty() &&
		    (!state->coordinator_llama.IsLoaded() || coord_path != state->coordinator_model_path)) {
			BufferedShellState load_state;
			state->coordinator_llama = LlamaState();
			if (!state->coordinator_llama.Load(coord_path, n_ctx, n_gpu, load_state)) {
				coord_path.clear();
			} else {
				state->coordinator_model_path = coord_path;
			}
		}

		BufferedShellState shell;
		if (coord_raw == "claude") {
			// Claude coordinator + llama worker
			WorkerFn worker_fn = [&](const string &q) -> string {
				return RunModelsSubprocessLoop(context, shell, q, 120, istats);
			};
			gstate->answer = RunModelsCoordinatorLoop(context, shell, worker_fn,
			                                          bind.question, 120, coord_rounds, istats);
		} else if (!coord_path.empty() && state->coordinator_llama.IsLoaded()) {
			// Llama coordinator + llama worker
			gstate->answer = RunCoordinatorLoop(context, shell,
			                                    state->coordinator_llama, state->llama,
			                                    bind.question, 120,
			                                    time_budget, coord_rounds, worker_style);
		} else {
			// No coordinator — direct llama loop
			gstate->answer = RunLlamaLoop(context, shell, state->llama, bind.question, 120);
		}
		if (gstate->answer.empty()) gstate->answer = shell.output;
		gstate->answer = gstate->answer;
		finish_history(gstate->answer);
		return gstate;
	}

	// sql_expert_model == "claude": Claude CLI subprocess backend.
	// Dispatch based on coordinator_model:
	//   "claude"   → Claude coordinator + Claude worker (two subprocesses)
	//   llama GGUF → llama coordinator + Claude worker
	//   ""         → Claude subprocess direct
	if (coord_raw == "claude") {
		BufferedShellState shell;
		WorkerFn worker_fn = [&](const string &q) -> string {
			return RunModelsSubprocessLoop(context, shell, q, 120, istats);
		};
		gstate->answer = RunModelsCoordinatorLoop(context, shell, worker_fn,
		    bind.question, 120, read_int("coordinator_max_rounds", 12), istats);
		if (gstate->answer == "") gstate->answer = shell.output;
		finish_history(gstate->answer);
		return gstate;
	}

	if (!coord_raw.empty() && coord_raw != "claude") {
		auto   &fs         = FileSystem::GetFileSystem(context);
		string  coord_path = ResolveModelPath(coord_raw, fs);
		int32_t n_ctx      = read_int("llama_context_size", 8192);
		int32_t n_gpu      = read_int("llama_gpu_layers",   -1);
		double  time_budget  = read_dbl("coordinator_time_budget", 60.0);
		int32_t coord_rounds = read_int("coordinator_max_rounds",   12);

		if (!state->coordinator_llama.IsLoaded() || coord_path != state->coordinator_model_path) {
			BufferedShellState load_state;
			state->coordinator_llama = LlamaState();
			if (!state->coordinator_llama.Load(coord_path, n_ctx, n_gpu, load_state)) {
				coord_path.clear();
			} else {
				state->coordinator_model_path = coord_path;
			}
		}

		if (!coord_path.empty() && state->coordinator_llama.IsLoaded()) {
			BufferedShellState shell;
			WorkerFn worker_fn = [&](const string &q) -> string {
				return RunModelsSubprocessLoop(context, shell, q, 120, istats);
			};
			gstate->answer = RunCoordinatorLoopWithWorker(
			    context, shell, state->coordinator_llama, worker_fn,
			    bind.question, 120, time_budget, coord_rounds);
			if (gstate->answer == "") gstate->answer = shell.output;
			finish_history(gstate->answer);
			return gstate;
		}
	}

	{
		BufferedShellState shell;
		gstate->answer = RunModelsSubprocessLoop(context, shell, bind.question, 120, istats);
		if (gstate->answer == "") gstate->answer = shell.output;
	}
	finish_history(gstate->answer);
	return gstate;
}

static void AskModelsScan(ClientContext & /*context*/, TableFunctionInput &data_p, DataChunk &output) {
	auto &gstate = data_p.global_state->Cast<AskModelsGlobalState>();
	if (gstate.returned) return;
	gstate.returned = true;
	output.SetCardinality(1);
	output.data[0].SetValue(0, Value(gstate.answer));
	output.data[1].SetValue(0, Value(gstate.stats_json));
}

// ---------------------------------------------------------------------------
// llama_discover_models() — scan well-known GGUF locations and return a table
// ---------------------------------------------------------------------------

struct DiscoveredModel {
	string name;        // filename without .gguf extension
	string path;        // absolute path
	string size;        // human-readable, e.g. "3.8 GiB"
	idx_t  size_bytes;  // raw size for comparisons
	string source;      // "claude", "lmstudio", "jan", "huggingface", "gpt4all", "other"
};


// Returns a deduplicated list of GGUF files found in standard tool directories.
static vector<DiscoveredModel> ScanGgufModels(FileSystem &fs) {
	string home = fs.GetHomeDirectory();

	// (glob_pattern, source_label) — order matters; first match wins for dedup
	vector<std::pair<string, string>> scan_dirs = {
	    {GetModelsModelsDir(fs) + "/*.gguf",                              "models"},
	    {home + "/.lmstudio/models/**/*.gguf",                            "lmstudio"},
	    {home + "/jan/models/**/*.gguf",                                  "jan"},
	    {home + "/.cache/huggingface/hub/models--*/snapshots/**/*.gguf",  "huggingface"},
#ifdef __APPLE__
	    {home + "/Library/Application Support/nomic.ai/GPT4All/*.gguf",  "gpt4all"},
#else
	    {home + "/.local/share/nomic.ai/GPT4All/*.gguf",                 "gpt4all"},
#endif
	};

	vector<DiscoveredModel> results;
	set<string> seen_paths; // dedup by absolute path

	for (auto &[pattern, source] : scan_dirs) {
		vector<string> matches;
		try {
			auto glob_results = fs.Glob(pattern);
			for (auto &gr : glob_results) matches.push_back(gr.path);
		} catch (...) {
			continue; // directory doesn't exist — silently skip
		}

		for (auto &path : matches) {
			if (seen_paths.count(path)) continue;
			seen_paths.insert(path);

			struct stat st;
			if (stat(path.c_str(), &st) != 0 || st.st_size == 0) continue;

			// Derive name from filename without extension
			auto slash = path.rfind('/');
			string filename = (slash != string::npos) ? path.substr(slash + 1) : path;
			string name = filename;
			if (name.size() > 5 && name.substr(name.size() - 5) == ".gguf") {
				name = name.substr(0, name.size() - 5);
			}

			idx_t sz = static_cast<idx_t>(st.st_size);
			results.push_back({name, path, StringUtil::BytesToHumanReadableString(sz, 1024), sz, source});
		}
	}
	return results;
}

struct DiscoverModelsGlobalState : public GlobalTableFunctionState {
	vector<DiscoveredModel> models;
	idx_t row = 0;
};

static unique_ptr<FunctionData> DiscoverModelsBind(ClientContext & /*context*/,
                                                    TableFunctionBindInput & /*input*/,
                                                    vector<LogicalType> &return_types,
                                                    vector<string> &names) {
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::VARCHAR, LogicalType::VARCHAR};
	names        = {"name", "path", "size", "source"};
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> DiscoverModelsInitGlobal(ClientContext &context,
                                                                       TableFunctionInitInput & /*input*/) {
	auto gstate = make_uniq<DiscoverModelsGlobalState>();
	gstate->models = ScanGgufModels(FileSystem::GetFileSystem(context));
	return gstate;
}

static void DiscoverModelsScan(ClientContext & /*context*/, TableFunctionInput &data_p, DataChunk &output) {
	auto &gstate = data_p.global_state->Cast<DiscoverModelsGlobalState>();
	idx_t count  = 0;
	while (gstate.row < gstate.models.size() && count < STANDARD_VECTOR_SIZE) {
		auto &m = gstate.models[gstate.row];
		output.data[0].SetValue(count, Value(m.name));
		output.data[1].SetValue(count, Value(m.path));
		output.data[2].SetValue(count, Value(m.size));
		output.data[3].SetValue(count, Value(m.source));
		gstate.row++;
		count++;
	}
	output.SetCardinality(count);
}

// ---------------------------------------------------------------------------
// llama_import_model(source [, name]) — copy a GGUF into the managed dir
// ---------------------------------------------------------------------------

struct ImportModelBindData : public FunctionData {
	string source_path;
	string dest_name;   // filename without .gguf
	string dest_path;   // full destination path

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<ImportModelBindData>();
		copy->source_path = source_path;
		copy->dest_name   = dest_name;
		copy->dest_path   = dest_path;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<ImportModelBindData>();
		return source_path == o.source_path && dest_path == o.dest_path;
	}
};

struct ImportModelGlobalState : public GlobalTableFunctionState {
	string status;
	string destination;
	bool   returned = false;
};

static unique_ptr<FunctionData> ImportModelBind(ClientContext &context,
                                                 TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types,
                                                 vector<string> &names) {
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
	names        = {"status", "destination"};

	auto &fs   = FileSystem::GetFileSystem(context);
	auto  bind = make_uniq<ImportModelBindData>();

	bind->source_path = ResolveModelPath(StringValue::Get(input.inputs[0]), fs);

	// Optional name override
	if (input.inputs.size() > 1 && !input.inputs[1].IsNull()) {
		bind->dest_name = StringValue::Get(input.inputs[1]);
	} else {
		// Derive from source filename without extension
		auto slash = bind->source_path.rfind('/');
		string fn  = (slash != string::npos) ? bind->source_path.substr(slash + 1) : bind->source_path;
		bind->dest_name = fn;
		if (bind->dest_name.size() > 5 && bind->dest_name.substr(bind->dest_name.size() - 5) == ".gguf") {
			bind->dest_name = bind->dest_name.substr(0, bind->dest_name.size() - 5);
		}
	}

	bind->dest_path = GetModelsModelsDir(fs) + "/" + bind->dest_name + ".gguf";
	return std::move(bind);
}

static unique_ptr<GlobalTableFunctionState> ImportModelInitGlobal(ClientContext &context,
                                                                    TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<ImportModelBindData>();
	auto  state = make_uniq<ImportModelGlobalState>();
	auto &fs    = FileSystem::GetFileSystem(context);

	// Ensure managed dir exists
	if (!MakeDirectories(GetModelsModelsDir(fs))) {
		throw IOException("Failed to create models directory: %s", GetModelsModelsDir(fs));
	}

	// Check source exists
	struct stat src_st;
	if (stat(bind.source_path.c_str(), &src_st) != 0 || src_st.st_size == 0) {
		throw IOException("Source model not found or empty: %s", bind.source_path);
	}

	// Already there?
	struct stat dst_st;
	if (stat(bind.dest_path.c_str(), &dst_st) == 0 && dst_st.st_size == src_st.st_size) {
		state->status      = "already_present";
		state->destination = bind.dest_path;
		fprintf(stderr, "Model already in managed dir: %s\n", bind.dest_path.c_str());
		return state;
	}

	// Copy with progress
	int64_t total = static_cast<int64_t>(src_st.st_size);
	fprintf(stderr, "Importing %.1f MB → %s\n", total / 1e6, bind.dest_path.c_str());

	FILE *src = fopen(bind.source_path.c_str(), "rb");
	if (!src) throw IOException("Cannot open source: %s", bind.source_path);

	FILE *dst = fopen(bind.dest_path.c_str(), "wb");
	if (!dst) { fclose(src); throw IOException("Cannot open destination: %s", bind.dest_path); }

	static constexpr size_t BUF = 4 * 1024 * 1024; // 4 MB copy buffer
	vector<char> buf(BUF);
	int64_t copied = 0;
	size_t  n;
	while ((n = fread(buf.data(), 1, BUF, src)) > 0) {
		if (fwrite(buf.data(), 1, n, dst) != n) {
			fclose(src); fclose(dst);
			remove(bind.dest_path.c_str());
			throw IOException("Write error while importing %s", bind.dest_path);
		}
		copied += static_cast<int64_t>(n);
		fprintf(stderr, "\r  %.1f / %.1f MB (%.0f%%)   ",
		        copied / 1e6, total / 1e6, 100.0 * copied / total);
		fflush(stderr);
	}
	fclose(src);
	fclose(dst);
	fprintf(stderr, "\r  %.1f MB — done.%*s\n", total / 1e6, 20, "");

	state->status      = "imported";
	state->destination = bind.dest_path;
	fprintf(stderr, "Imported to: %s\n", bind.dest_path.c_str());
	fprintf(stderr, "To use: SET llama_model_path = '%s';\n", bind.dest_name.c_str());
	return state;
}

static void ImportModelScan(ClientContext & /*context*/, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = data_p.global_state->Cast<ImportModelGlobalState>();
	if (state.returned) return;
	state.returned = true;
	output.SetCardinality(1);
	output.data[0].SetValue(0, Value(state.status));
	output.data[1].SetValue(0, Value(state.destination));
}

// ---------------------------------------------------------------------------
// setup_models() — one-shot setup: download worker + coordinator, configure
// settings, and warm up the worker model so Metal compiles on first load.
// ---------------------------------------------------------------------------

struct SetupModelsGlobalState : public GlobalTableFunctionState {
	vector<string> messages;
	idx_t          row = 0;
};

struct SetupModelsBindData : public FunctionData {
	bool save_override = false; // override:=true — write setup.sql even if it differs
	bool claude        = false; // claude:=true  — configure Claude CLI instead of llama
	unique_ptr<FunctionData> Copy() const override {
		auto c = make_uniq<SetupModelsBindData>(); c->save_override = save_override; c->claude = claude; return std::move(c);
	}
	bool Equals(const FunctionData &o) const override {
		auto &oc = o.Cast<SetupModelsBindData>(); return save_override == oc.save_override && claude == oc.claude;
	}
};

static unique_ptr<FunctionData> SetupModelsBind(ClientContext & /*context*/,
                                                TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types,
                                                vector<string> &names) {
	return_types = {LogicalType::VARCHAR};
	names        = {"status"};
	auto bind = make_uniq<SetupModelsBindData>();
	auto get_bool = [&](const string &key) {
		auto it = input.named_parameters.find(key);
		return it != input.named_parameters.end() && !it->second.IsNull() && BooleanValue::Get(it->second);
	};
	bind->save_override = get_bool("override");
	bind->claude        = get_bool("claude");
	return std::move(bind);
}

// Generate the SET statements that represent the current claude/llama settings.
static string GenerateSetupSQL(ClientContext &context) {
	auto read = [&](const string &key) -> string {
		Value v;
		return context.TryGetCurrentSetting(key, v) ? v.ToString() : "";
	};

	string sql = "-- DuckDB models extension setup (auto-generated by setup_models)\n";
	sql += "SET sql_expert_model         = '" + read("sql_expert_model")         + "';\n";
	sql += "SET coordinator_model        = '" + read("coordinator_model")        + "';\n";
	sql += "SET coordinator_time_budget  = "  + read("coordinator_time_budget")  + ";\n";
	sql += "SET coordinator_max_rounds   = "  + read("coordinator_max_rounds")   + ";\n";
	sql += "SET sql_expert_model_style   = '" + read("sql_expert_model_style")   + "';\n";
	sql += "SET llama_context_size       = "  + read("llama_context_size")       + ";\n";
	sql += "SET llama_gpu_layers         = "  + read("llama_gpu_layers")         + ";\n";
	return sql;
}

static unique_ptr<GlobalTableFunctionState> SetupModelsInitGlobal(ClientContext &context,
                                                                   TableFunctionInitInput &input) {
	auto gstate = make_uniq<SetupModelsGlobalState>();
	auto &fs    = FileSystem::GetFileSystem(context);
	auto &db    = *context.db;

	bool save_override = input.bind_data ? input.bind_data->Cast<SetupModelsBindData>().save_override : false;
	bool use_claude    = input.bind_data ? input.bind_data->Cast<SetupModelsBindData>().claude        : false;

	auto &msgs = gstate->messages;
	auto emit  = [&](const string &s) { msgs.push_back(s); };

	// Helper: persist current settings to setup.sql.
	// - File absent        → create it (always)
	// - File == new content → no-op, emit "unchanged"
	// - File differs       → error unless override:=true
	auto save_setup = [&]() {
		string setup_path = GetModelsSetupPath(fs);
		if (!MakeDirectories(GetModelsDataDir(fs))) {
			emit("Warning: could not create data dir, setup not saved.");
			return;
		}
		string new_sql  = GenerateSetupSQL(context);
		string existing = ReadFileContent(setup_path);
		if (!existing.empty()) {
			if (StripSQLComments(existing) == StripSQLComments(new_sql)) {
				emit("Setup unchanged — " + setup_path + " already up to date.");
				return;
			}
			if (!save_override) {
				// Compute a simple line-level diff to show what changed.
				string diff_msg;
				string new_stripped = StripSQLComments(new_sql);
				string old_stripped = StripSQLComments(existing);
				auto lines = [](const string &s) {
					vector<string> v;
					size_t i = 0;
					while (i <= s.size()) {
						size_t nl = s.find('\n', i);
						if (nl == string::npos) nl = s.size();
						v.push_back(s.substr(i, nl - i));
						i = nl + 1;
					}
					return v;
				};
				auto old_lines = lines(old_stripped);
				auto new_lines = lines(new_stripped);
				for (auto &l : old_lines) {
					bool found = false;
					for (auto &nl : new_lines) { if (l == nl) { found = true; break; } }
					if (!found && !l.empty()) diff_msg += "  - " + l + "\n";
				}
				for (auto &nl : new_lines) {
					bool found = false;
					for (auto &l : old_lines) { if (nl == l) { found = true; break; } }
					if (!found && !nl.empty()) diff_msg += "  + " + nl + "\n";
				}
				emit("Warning: setup.sql has different content (not overwritten):");
				// Emit each diff line as a separate row so the table renders cleanly.
				size_t pos = 0;
				while (pos < diff_msg.size()) {
					size_t nl = diff_msg.find('\n', pos);
					if (nl == string::npos) nl = diff_msg.size();
					string line = diff_msg.substr(pos, nl - pos);
					if (!line.empty()) emit(line);
					pos = nl + 1;
				}
				emit("  Pass override:=true to overwrite: CALL setup_models(override:=true);");
				return;
			}
		}
		FILE *f = fopen(setup_path.c_str(), "w");
		if (!f) { emit("Warning: could not write " + setup_path); return; }
		fwrite(new_sql.c_str(), 1, new_sql.size(), f);
		fclose(f);
		emit("Setup saved to " + setup_path);
	};

	// ----- load from setup.sql as the base (explicit flags applied on top) --------------

	string setup_path = GetModelsSetupPath(fs);
	string existing   = ReadFileContent(setup_path);
	if (!existing.empty()) {
		emit("Loading setup from " + setup_path);
		// Strip comment lines first so a leading comment doesn't swallow the
		// first SET statement when we split by semicolons.
		string stripped = StripSQLComments(existing);
		size_t i = 0;
		while (i < stripped.size()) {
			size_t semi = stripped.find(';', i);
			if (semi == string::npos) break;
			string stmt = stripped.substr(i, semi - i + 1);
			size_t start = stmt.find_first_not_of(" \t\r\n");
			if (start != string::npos) {
				Connection conn(db);
				auto res = conn.Query(stmt);
				if (res->HasError()) {
					emit("  Warning: " + res->GetError());
				}
			}
			i = semi + 1;
		}
		// If no explicit flags were passed, we're done — base config is loaded.
		if (!use_claude) {
			emit("Ready. Use: .claude <your question>");
			save_setup();
			return gstate;
		}
		// Otherwise fall through so explicit flags are applied on top.
	}

	// ----- helpers --------------------------------------------------------

	auto is_downloaded = [&](const string &name) -> bool {
		auto it = KNOWN_MODELS.find(name);
		if (it == KNOWN_MODELS.end()) return false;
		auto slash = it->second.rfind('/');
		string filename = (slash != string::npos) ? it->second.substr(slash + 1) : it->second;
		string path = GetModelsModelsDir(fs) + "/" + filename;
		struct stat st;
		return stat(path.c_str(), &st) == 0 && st.st_size > 0;
	};

	// Try to import from a locally discovered model before downloading.
	// Returns true if imported (or already present), false if not found.
	bool used_local_models = false;
	auto try_import_local = [&](const string &name) -> bool {
		auto discovered = ScanGgufModels(fs);
		string name_lower = StringUtil::Lower(name);
		const DiscoveredModel *best = nullptr;
		for (auto &m : discovered) {
			if (StringUtil::Lower(m.name).find(name_lower) != string::npos) {
				if (!best || m.size_bytes > best->size_bytes) best = &m;
			}
		}
		if (!best) return false;

		auto it = KNOWN_MODELS.find(name);
		string dest_name;
		if (it != KNOWN_MODELS.end()) {
			auto slash = it->second.rfind('/');
			dest_name = (slash != string::npos) ? it->second.substr(slash + 1) : it->second;
			if (dest_name.size() > 5 && dest_name.substr(dest_name.size() - 5) == ".gguf") {
				dest_name = dest_name.substr(0, dest_name.size() - 5);
			}
		} else {
			dest_name = name;
		}
		string dest_path = GetModelsModelsDir(fs) + "/" + dest_name + ".gguf";
		struct stat dst_st;
		if (stat(dest_path.c_str(), &dst_st) == 0 && dst_st.st_size > 0) {
			return true;
		}

		emit(StringUtil::Format("[✓] Found %s in %s (%s) — importing instead of downloading",
		                        name, best->source, best->path));
		Connection conn(db);
		auto res = conn.Query("CALL llama_import_model('" + best->path + "', '" + dest_name + "')");
		if (res->HasError()) {
			emit("  Import failed: " + res->GetError() + " — will download instead.");
			return false;
		}
		used_local_models = true;
		emit("  Imported successfully.");
		return true;
	};

	auto download = [&](const string &name) {
		emit("Downloading " + name + " ...");
		Connection conn(db);
		auto res = conn.Query("CALL llama_download_model('" + name + "')");
		if (res->HasError()) {
			emit("  ERROR: " + res->GetError());
		} else {
			emit("  Done.");
		}
	};

	auto set_str = [&](const string &key, const string &val) {
		Connection conn(db);
		conn.Query("SET " + key + " = '" + val + "'");
	};
	auto set_num = [&](const string &key, const string &val) {
		Connection conn(db);
		conn.Query("SET " + key + " = " + val);
	};

	// ----- claude:=true path — no downloads, just configure Claude CLI -----

	if (use_claude) {
		set_str("sql_expert_model",       "claude");
		set_str("coordinator_model",      "claude");
		set_num("coordinator_time_budget","120.0");
		set_num("coordinator_max_rounds", "15");
		emit("Settings configured (Claude CLI backend):");
		emit("  sql_expert_model       = claude");
		emit("  coordinator_model      = claude");
		emit("  coordinator_time_budget = 120s");
		emit("  coordinator_max_rounds  = 15");
		// TODO: test and tune coordinator+Claude CLI interaction; consider whether
		// coordinator makes sense when both models are Claude CLI (same subprocess).
		emit("Ready. Use: .claude <your question>");
		save_setup();
		return gstate;
	}

	// ----- llama path: acquire models (local first, then download) -------

	static const char WORKER[]      = "sqlcoder-7b-2";
	static const char COORDINATOR[] = "qwen2.5-3b";

	if (is_downloaded(WORKER)) {
		emit(string("Worker model '") + WORKER + "' already present.");
	} else if (!try_import_local(WORKER)) {
		download(WORKER);
	}

	if (is_downloaded(COORDINATOR)) {
		emit(string("Coordinator model '") + COORDINATOR + "' already present.");
	} else if (!try_import_local(COORDINATOR)) {
		download(COORDINATOR);
	}

	// ----- configure settings --------------------------------------------

	set_str("sql_expert_model",        WORKER);
	set_str("coordinator_model",       COORDINATOR);
	set_num("coordinator_time_budget", "120.0");
	set_num("coordinator_max_rounds",  "15");

	emit("Settings configured (local llama backend):");
	emit(string("  sql_expert_model        = ") + WORKER);
	emit(string("  coordinator_model       = ") + COORDINATOR);
	emit("  coordinator_time_budget = 120s");
	emit("  coordinator_max_rounds  = 15");

	// Warmup: load the model and compile Metal pipelines now so the first
	// real question doesn't pay the compilation cost.
	emit("Warming up (loading model + compiling Metal pipelines)...");
	{
		Connection conn(db);
		conn.Query("CALL ask_models('just return OK, no info required')");
	}
	emit("Ready. Use: .claude <your question>");

	if (!used_local_models) {
		emit("Tip: install models via LM Studio or Jan first and setup_models() will import them");
		emit("     instead of downloading. Run: FROM llama_discover_models() to see what's on your machine.");
	}
	emit("Setup will be saved to setup.sql automatically.");

	save_setup();

	return gstate;
}

static void SetupModelsScan(ClientContext & /*context*/, TableFunctionInput &data_p, DataChunk &output) {
	auto &gstate = data_p.global_state->Cast<SetupModelsGlobalState>();
	idx_t count  = 0;
	while (gstate.row < gstate.messages.size() && count < STANDARD_VECTOR_SIZE) {
		output.data[0].SetValue(count, Value(gstate.messages[gstate.row]));
		gstate.row++;
		count++;
	}
	output.SetCardinality(count);
}

// Replacement scan: FROM llama_instructions (no parentheses)
static unique_ptr<TableRef> LlamaInstructionsReplacementScan(ClientContext &context,
                                                              ReplacementScanInput &input,
                                                              optional_ptr<ReplacementScanData> /*data*/) {
	if (input.table_name != "llama_instructions") {
		return nullptr;
	}
	vector<LogicalType> types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
	vector<string>      names = {"task", "hint"};
	auto collection = make_uniq<ColumnDataCollection>(context, types);
	DataChunk chunk;
	chunk.Initialize(context, types);
	for (idx_t i = 0; i < LLAMA_INSTRUCTIONS_COUNT; i++) {
		chunk.data[0].SetValue(i, Value(LLAMA_INSTRUCTIONS[i].task));
		chunk.data[1].SetValue(i, Value(LLAMA_INSTRUCTIONS[i].hint));
	}
	chunk.SetCardinality(LLAMA_INSTRUCTIONS_COUNT);
	collection->Append(chunk);
	return make_uniq<ColumnDataRef>(std::move(collection), names);
}

// ---------------------------------------------------------------------------
// Extension entry points
// ---------------------------------------------------------------------------

static void LoadInternal(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();

	// Register the "Models" log type (enable with: SET enable_logging=true; SET log_type='Models';)
	db.GetLogManager().RegisterLogType(make_uniq<ModelsLogType>());

	// Shared state — lives for the database lifetime
	auto state = make_shared_ptr<ModelsExtensionState>();

#ifdef MODELS_SHELL_EXT
	// Shell dot-command: .claude <question>
	ShellCommandExtension claude_shell_cmd;
	claude_shell_cmd.command     = "claude";
	claude_shell_cmd.usage       = "<question>";
	claude_shell_cmd.description = "Ask a question about your DuckDB database (run CALL setup_models() to configure)";
	claude_shell_cmd.callback    = ModelsShellCommand;
	claude_shell_cmd.info        = state;
	loader.RegisterShellCommand(std::move(claude_shell_cmd));
#endif // MODELS_SHELL_EXT

	auto &config = DBConfig::GetConfig(db);
	// Primary model settings — backend inferred from value:
	//   empty or "claude" → Claude CLI subprocess
	//   anything else     → local llama.cpp (GGUF path or short name)
	config.AddExtensionOption("sql_expert_model",
	    "SQL expert model: 'claude' = Claude CLI, short-name or path = local llama GGUF (default: '' = not configured)",
	    LogicalType::VARCHAR, Value(""), nullptr, SetScope::GLOBAL);
	config.AddExtensionOption("coordinator_model",
	    "Coordinator model: 'claude' = Claude CLI, short-name or path = local llama GGUF, '' = none",
	    LogicalType::VARCHAR, Value(""), nullptr, SetScope::GLOBAL);
	config.AddExtensionOption("coordinator_time_budget",
	    "Wall-clock time budget for coordinator loop (seconds)",
	    LogicalType::DOUBLE, Value(60.0), nullptr, SetScope::GLOBAL);
	config.AddExtensionOption("coordinator_max_rounds",
	    "Max coordinator iterations",
	    LogicalType::INTEGER, Value::INTEGER(12), nullptr, SetScope::GLOBAL);
	config.AddExtensionOption("sql_expert_model_style",
	    "Prompt style for sql_expert_model: 'auto' (detect from name), 'chat', 'sqlcoder'",
	    LogicalType::VARCHAR, Value("auto"), nullptr, SetScope::GLOBAL);

	// llama.cpp-specific tuning (only relevant when a GGUF model is active)
	config.AddExtensionOption("llama_context_size",
	    "Context window size in tokens for llama models",
	    LogicalType::INTEGER, Value::INTEGER(8192), nullptr, SetScope::GLOBAL);
	config.AddExtensionOption("llama_gpu_layers",
	    "Number of layers to offload to GPU for llama models (-1 = all)",
	    LogicalType::INTEGER, Value::INTEGER(-1), nullptr, SetScope::GLOBAL);

	// Table function: CALL llama_download_model(url_or_name)
	TableFunction download_tf("llama_download_model", {LogicalType::VARCHAR},
	                          DownloadModelScan, DownloadModelBind, DownloadModelInitGlobal);
	loader.RegisterFunction(std::move(download_tf));

	// Table function: FROM llama_discover_models() — scan well-known GGUF locations
	TableFunction discover_tf("llama_discover_models", {},
	                          DiscoverModelsScan, DiscoverModelsBind, DiscoverModelsInitGlobal);
	loader.RegisterFunction(std::move(discover_tf));

	// Table function: CALL llama_import_model(source [, name]) — copy GGUF to managed dir
	// Two overloads: 1-arg and 2-arg (source + optional dest_name).
	TableFunctionSet import_set("llama_import_model");
	import_set.AddFunction(TableFunction({LogicalType::VARCHAR},
	                        ImportModelScan, ImportModelBind, ImportModelInitGlobal));
	import_set.AddFunction(TableFunction({LogicalType::VARCHAR, LogicalType::VARCHAR},
	                        ImportModelScan, ImportModelBind, ImportModelInitGlobal));
	loader.RegisterFunction(std::move(import_set));

	// Table function: FROM models_interactions()
	TableFunction tf("models_interactions", {}, ModelsInteractionsScan, ModelsInteractionsBind,
	                 ModelsInteractionsInitGlobal);
	tf.function_info = make_shared_ptr<ModelsTableFunctionInfo>(state);
	loader.RegisterFunction(std::move(tf));

	// Replacement scan: FROM models_interactions (no parentheses)
	auto scan_data = make_uniq<ModelsReplacementScanData>(state);
	DBConfig::GetConfig(db).replacement_scans.emplace_back(ModelsInteractionsReplacementScan, std::move(scan_data));

	// Table function + replacement scan: FROM llama_instructions / FROM llama_instructions()
	TableFunction instr_tf("llama_instructions", {}, LlamaInstructionsScan,
	                       LlamaInstructionsBind, LlamaInstructionsInitGlobal);
	loader.RegisterFunction(std::move(instr_tf));
	DBConfig::GetConfig(db).replacement_scans.emplace_back(LlamaInstructionsReplacementScan, nullptr);

	// Table function + replacement scan: FROM llama_error_hints / FROM llama_error_hints()
	TableFunction errhints_tf("llama_error_hints", {}, LlamaErrorHintsScan,
	                          LlamaErrorHintsBind, LlamaErrorHintsInitGlobal);
	loader.RegisterFunction(std::move(errhints_tf));
	DBConfig::GetConfig(db).replacement_scans.emplace_back(LlamaErrorHintsReplacementScan, nullptr);

	// Table function + replacement scan: FROM llama_models / FROM llama_models()
	TableFunction models_tf("llama_models", {}, LlamaModelsScan,
	                        LlamaModelsBind, LlamaModelsInitGlobal);
	loader.RegisterFunction(std::move(models_tf));
	DBConfig::GetConfig(db).replacement_scans.emplace_back(LlamaModelsReplacementScan, nullptr);

	// Table function: CALL setup_models([claude:=true] [, override:=true])
	// setup_models()                  — load setup.sql if present; else full llama setup + save
	// setup_models(claude:=true)      — configure Claude CLI backend + save
	// setup_models(override:=true)    — allow overwriting an existing setup.sql that differs
	TableFunction setup_tf("setup_models", {}, SetupModelsScan,
	                       SetupModelsBind, SetupModelsInitGlobal);
	setup_tf.named_parameters["claude"]   = LogicalType::BOOLEAN;
	setup_tf.named_parameters["override"] = LogicalType::BOOLEAN;
	loader.RegisterFunction(std::move(setup_tf));

	// Table function: CALL ask_models('question') — SQL equivalent of .claude <question>
	TableFunction ask_tf("ask_models", {LogicalType::VARCHAR}, AskModelsScan,
	                     AskModelsBind, AskModelsInitGlobal);
	ask_tf.function_info = make_shared_ptr<ModelsTableFunctionInfo>(state);
	loader.RegisterFunction(std::move(ask_tf));

	// SQL macro: duckdb_summary([topic]) — unified self-documentation for Claude
	// duckdb_summary()          -> all topics
	// duckdb_summary('schema')  -> table/column info
	// duckdb_summary('context') -> databases, extensions, settings
	Connection conn(db);
	conn.Query(R"SQL(
CREATE OR REPLACE MACRO duckdb_summary(topic := 'all') AS TABLE
    -- 'context': summary stats
    SELECT 'context' AS section, 'stats'     AS category, 'schemas' AS value, (SELECT COUNT(*)::VARCHAR FROM duckdb_schemas()) AS details WHERE topic IN ('context', 'all')
    UNION ALL
    SELECT 'context',             'stats',                 'tables',           (SELECT COUNT(*)::VARCHAR FROM duckdb_tables())            WHERE topic IN ('context', 'all')
    UNION ALL
    SELECT 'context',             'stats',                 'views',            (SELECT COUNT(*)::VARCHAR FROM duckdb_views())             WHERE topic IN ('context', 'all')
    UNION ALL
    -- 'context': attached databases
    SELECT 'context', 'database',  database_name, COALESCE(path, '(in-memory)') FROM duckdb_databases() WHERE topic IN ('context', 'all')
    UNION ALL
    -- 'context': loaded extensions
    SELECT 'context', 'extension', extension_name, COALESCE(extension_version, '') FROM duckdb_extensions() WHERE loaded AND topic IN ('context', 'all')
    UNION ALL
    -- 'context': key settings
    SELECT 'context', 'setting', name, value FROM duckdb_settings()
    WHERE topic IN ('context', 'all') AND name IN ('threads', 'memory_limit', 'TimeZone', 'search_path', 'default_order')
    UNION ALL
    -- 'schema': one row per column
    SELECT
        'schema'       AS section,
        c.table_name   AS category,
        c.column_name  AS value,
        c.data_type
            || CASE WHEN NOT c.is_nullable THEN ' NOT NULL' ELSE '' END
            || CASE WHEN COALESCE(list_contains(pk.constraint_column_names, c.column_name), false)
                    THEN ' PK' ELSE '' END AS details
    FROM duckdb_columns() c
    JOIN duckdb_tables() t USING (table_name, schema_name, database_name)
    LEFT JOIN (
        SELECT table_name, schema_name, database_name, constraint_column_names
        FROM duckdb_constraints() WHERE constraint_type = 'PRIMARY KEY'
    ) pk USING (table_name, schema_name, database_name)
    WHERE topic IN ('schema', 'all')
    UNION ALL
    -- invalid topic: raise an error
    SELECT error('duckdb_summary: unknown topic ''' || topic || ''' — valid topics: schema, context'),
           NULL, NULL, NULL
    WHERE topic NOT IN ('schema', 'context', 'all')
)SQL");

	// Scalar macro: models_logs_parsed(message) — extract SQL query from a Models log message
	// Returns the SQL query text if the message is a result row with a SQL_QUERY, NULL otherwise.
	// Usage: FROM duckdb_logs_parsed('Models') SELECT models_logs_parsed(message)
	conn.Query(R"SQL(
CREATE OR REPLACE MACRO models_logs_parsed(message) AS
    CASE
      WHEN json_extract_string(message, '$.type') = 'result'
       AND json_extract_string(message, '$.result') LIKE 'SQL_QUERY:%'
      THEN ltrim(substr(
          json_extract_string(message, '$.result'),
          length('SQL_QUERY:') + 1
      ))
    END;
)SQL");
}

void ModelsExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string ModelsExtension::Name() {
	return "models";
}

std::string ModelsExtension::Version() const {
#ifdef EXT_VERSION_MODELS
	return EXT_VERSION_MODELS;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(models, loader) { // NOLINT
	duckdb::LoadInternal(loader);
}

} // extern "C"
