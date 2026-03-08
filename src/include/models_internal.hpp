#pragma once

// Internal shared declarations between models_extension.cpp and llama_backend.cpp.
// Not part of the public extension API.

#include "duckdb.hpp"
#include "duckdb/logging/log_type.hpp"
#include "duckdb/logging/logger.hpp"

#include <functional>

namespace duckdb {

// Stats collected during a single ask_models() / .models invocation.
// Serialised as JSON into ConversationEntry::stats.
struct InteractionStats {
	int sql_total   = 0; // SQL queries sent to ExecuteQuery
	int sql_ok      = 0; // queries that returned results without error
	int sql_error   = 0; // queries that returned an error
	int rounds      = 0; // coordinator loop iterations / subprocess result events
	int delegations = 0; // DELEGATE_SQL dispatches (coordinator mode)

	string ToJSON() const {
		return "{\"sql_total\":"    + to_string(sql_total) +
		       ",\"sql_ok\":"       + to_string(sql_ok) +
		       ",\"sql_error\":"    + to_string(sql_error) +
		       ",\"rounds\":"       + to_string(rounds) +
		       ",\"delegations\":"  + to_string(delegations) + "}";
	}
};

// Worker callback type — used by coordinator loops to dispatch sub-tasks to any backend.
// Receives the worker input (question, optionally with schema pre-injected) and returns
// the worker's result (natural language or "SQL: <sql>\nResult:\n<result>" format).
using WorkerFn = std::function<string(const string &worker_input)>;

// ---------------------------------------------------------------------------
// Logging — structured log type: {interaction_type VARCHAR, message VARCHAR}
// Enable: SET enable_logging=true; SET enabled_log_types='Models';
// Query:  FROM duckdb_logs_parsed('Models');
// ---------------------------------------------------------------------------

static inline string ModelsJsonEscapeString(const string &s) {
	string out;
	out.reserve(s.size() + 2);
	out += '"';
	for (unsigned char c : s) {
		switch (c) {
		case '"':  out += "\\\""; break;
		case '\\': out += "\\\\"; break;
		case '\n': out += "\\n";  break;
		case '\r': out += "\\r";  break;
		case '\t': out += "\\t";  break;
		default:
			if (c < 0x20) {
				char buf[8];
				snprintf(buf, sizeof(buf), "\\u%04x", c);
				out += buf;
			} else {
				out += (char)c;
			}
		}
	}
	out += '"';
	return out;
}

class ModelsLogType : public LogType {
public:
	static constexpr const char *NAME  = "Models";
	static constexpr LogLevel    LEVEL = LogLevel::LOG_DEBUG;

	ModelsLogType() : LogType(NAME, LogLevel::LOG_DEBUG, GetLogType()) {
	}

	static LogicalType GetLogType() {
		child_list_t<LogicalType> fields = {
		    {"interaction_type", LogicalType::VARCHAR},
		    {"message",          LogicalType::VARCHAR},
		};
		return LogicalType::STRUCT(fields);
	}

	// interaction_type: "out" | "in" | "start" | "exit" | "misc" | "llama"
	static string ConstructLogMessage(const string &interaction_type, const string &message) {
		return "{\"interaction_type\":" + ModelsJsonEscapeString(interaction_type) +
		       ",\"message\":"          + ModelsJsonEscapeString(message) + "}";
	}
};

// Strip chat-template special tokens (e.g. <|im_end|>) from SQL strings.
// These leak into generated text when the model outputs them as regular tokens.
static inline string StripSpecialTokens(const string &s) {
	static const char * const TOKENS[] = {
	    "<|im_end|>", "<|im_start|>", "</s>", "<s>", "<|endoftext|>",
	    "<|eot_id|>", "<|end|>", "[/INST]", "[INST]", nullptr
	};
	string result = s;
	bool changed  = true;
	while (changed) {
		changed = false;
		for (int i = 0; TOKENS[i]; i++) {
			auto pos = result.find(TOKENS[i]);
			if (pos != string::npos) {
				result  = result.substr(0, pos);
				changed = true;
			}
		}
		// Trim trailing whitespace left by stripping.
		while (!result.empty() && (result.back() == ' ' || result.back() == '\t' ||
		                            result.back() == '\n' || result.back() == '\r')) {
			result.pop_back();
		}
	}
	return result;
}

// Convenience macro for the llama backend: MODELS_LOG(context, "event", "fmt", args...)
#define MODELS_LOG(SOURCE, EVENT, ...)                                                                                  \
	DUCKDB_LOG(SOURCE, ModelsLogType, EVENT, StringUtil::Format(__VA_ARGS__))

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int    MAX_SQL_ROUNDS        = 50;
static constexpr idx_t  MAX_RESULT_ROWS       = 200;
static constexpr int    QUERY_TIMEOUT_SECONDS = 30;
static constexpr size_t MAX_RESULT_CHARS      = 4000; // cap per SQL result fed back into llama context

extern const char * const SUBPROCESS_SYSTEM_PROMPT;  // chat-style worker (qwen etc.)
extern const char * const SQLCODER_SYSTEM_PROMPT;    // SQL-only worker (sqlcoder, nsql)
extern const char * const COORDINATOR_SYSTEM_PROMPT;

// ---------------------------------------------------------------------------
// SQL tool helpers (defined in models_extension.cpp)
// ---------------------------------------------------------------------------

struct QueryExecutionResult {
	bool   is_error;
	string text;
};

bool                 IsSafeSQL(const string &sql);
QueryExecutionResult ExecuteQuery(DatabaseInstance &db, const string &sql);
string               ExtractSQLQuery(const string &text);
string               MatchErrorHints(const string &error_text);

} // namespace duckdb
