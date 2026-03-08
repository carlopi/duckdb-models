#include "models_internal.hpp" // must come before llama_backend.hpp (provides WorkerFn)
#include "llama_backend.hpp"

#include "duckdb/common/string_util.hpp"

#include <llama.h>

#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <thread>
#include <unordered_set>
#include <vector>

namespace duckdb {

// ---------------------------------------------------------------------------
// LlamaState
// ---------------------------------------------------------------------------

LlamaState::~LlamaState() {
	if (ctx) {
		llama_free(ctx);
		ctx = nullptr;
	}
	if (model) {
		llama_model_free(model);
		model = nullptr;
	}
}

bool LlamaState::Load(const string &model_path, int32_t n_ctx, int32_t n_gpu_layers, BaseShellState &shell_state) {
	llama_backend_init();

	llama_model_params mparams = llama_model_default_params();
	mparams.n_gpu_layers = n_gpu_layers;

	model = llama_model_load_from_file(model_path.c_str(), mparams);
	if (!model) {
		shell_state.ShellPrintError(StringUtil::Format("Error: failed to load llama model from '%s'\n", model_path));
		return false;
	}

	llama_context_params cparams = llama_context_default_params();
	cparams.n_ctx = static_cast<uint32_t>(n_ctx);
	cparams.n_batch = static_cast<uint32_t>(n_ctx);
	cparams.n_threads = static_cast<int32_t>(std::thread::hardware_concurrency());

	ctx = llama_init_from_model(model, cparams);
	if (!ctx) {
		shell_state.ShellPrintError("Error: failed to create llama context\n");
		llama_model_free(model);
		model = nullptr;
		return false;
	}
	return true;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Apply the model's built-in chat template and return the formatted string.
static string ApplyChatTemplate(llama_model * /*model*/, const std::vector<llama_chat_message> &msgs) {
	// First call with zero-length buffer to get required size
	int n = llama_chat_apply_template(nullptr, // use model's built-in template
	                                  msgs.data(), msgs.size(),
	                                  /*add_ass=*/true, nullptr, 0);
	if (n < 0) {
		// Template not found — minimal fallback
		string out;
		for (auto &m : msgs) {
			out += "<|";
			out += m.role;
			out += "|>\n";
			out += m.content;
			out += "\n";
		}
		out += "<|assistant|>\n";
		return out;
	}
	std::vector<char> buf(static_cast<size_t>(n) + 1, '\0');
	llama_chat_apply_template(nullptr, msgs.data(), msgs.size(),
	                          /*add_ass=*/true, buf.data(), static_cast<int32_t>(buf.size()));
	return string(buf.data(), static_cast<size_t>(n));
}

// Tokenize a string. Returns token vector or empty on failure.
static std::vector<llama_token> Tokenize(const llama_vocab *vocab, const string &text, bool add_special) {
	// First pass: get count
	int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_special,
	                       /*parse_special=*/true);
	if (n == 0)
		return {};
	// llama_tokenize returns negative value if buffer is too small; abs gives needed size
	if (n < 0)
		n = -n;
	std::vector<llama_token> tokens(static_cast<size_t>(n));
	int actual = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), n, add_special,
	                            /*parse_special=*/true);
	if (actual < 0)
		return {}; // still failed
	tokens.resize(static_cast<size_t>(actual));
	return tokens;
}

// Convert a single token to its string piece.
static string TokenToPiece(const llama_vocab *vocab, llama_token token) {
	char buf[256];
	int n = llama_token_to_piece(vocab, token, buf, static_cast<int32_t>(sizeof(buf) - 1),
	                             /*lstrip=*/0, /*special=*/true);
	if (n < 0) {
		// Buffer too small — rare for a single piece; retry with a larger buffer
		std::vector<char> big(static_cast<size_t>(-n) + 1, '\0');
		n = llama_token_to_piece(vocab, token, big.data(), static_cast<int32_t>(big.size()),
		                         /*lstrip=*/0, /*special=*/true);
		if (n > 0)
			return string(big.data(), static_cast<size_t>(n));
		return {};
	}
	return string(buf, static_cast<size_t>(n));
}

// Generate tokens until EOS/EOT or SQL_QUERY: detected. Returns generated text.
// Streams each piece to the shell as it is produced.
static string Generate(llama_context *ctx, const llama_vocab *vocab, llama_sampler *sampler,
                       BaseShellState &shell_state, ClientContext &client_ctx, uint32_t ctx_size,
                       uint32_t n_prompt_tokens, bool stream = true) {
	string response;
	llama_token eos = llama_vocab_eos(vocab);
	llama_token eot = llama_vocab_eot(vocab);

	// Reset sampler state for fresh generation
	llama_sampler_reset(sampler);

	auto t_gen_start = std::chrono::steady_clock::now();
	uint32_t n_generated = 0;
	while (true) {
		llama_token token = llama_sampler_sample(sampler, ctx, -1);
		if (token < 0)
			break; // invalid / LLAMA_TOKEN_NULL

		if (token == eos || token == eot)
			break;
		llama_sampler_accept(sampler, token);

		string piece = TokenToPiece(vocab, token);
		response += piece;
		if (stream)
			shell_state.ShellPrint(piece);

		// Stop generating once we have a *complete* SQL_QUERY: line (newline-terminated)
		{
			auto sql_pos = response.find("SQL_QUERY:");
			if (sql_pos != string::npos && response.find('\n', sql_pos) != string::npos)
				break;
		}

		// Periodic progress log every 50 tokens so slow generation is visible.
		n_generated++;
		if (n_generated % 50 == 0) {
			double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen_start).count();
			MODELS_LOG(client_ctx, "llama", "generating... %u tokens in %.1fs (%.1f tok/s)", n_generated, elapsed,
			           elapsed > 0 ? n_generated / elapsed : 0.0);
		}

		// Abort if context is nearly full (leave room for tool results)
		if (n_prompt_tokens + n_generated + 64 >= ctx_size) {
			MODELS_LOG(client_ctx, "llama", "context nearly full (%u + %u >= %u), stopping", n_prompt_tokens,
			           n_generated, ctx_size - 64);
			break;
		}

		// Feed token back into the context for the next step
		llama_batch next = llama_batch_get_one(&token, 1);
		if (llama_decode(ctx, next) != 0) {
			MODELS_LOG(client_ctx, "llama", "llama_decode failed at token %u", n_generated);
			break;
		}
	}
	MODELS_LOG(client_ctx, "llama", "generate done: %u tokens", n_generated);
	return response;
}

// ---------------------------------------------------------------------------
// RunLlamaLoop
// ---------------------------------------------------------------------------

string RunLlamaLoop(ClientContext &context, BaseShellState &shell_state, LlamaState &lm, const string &user_input,
                    idx_t /*term_width*/, const string &system_prompt_override, bool single_shot_sql,
                    InteractionStats *stats) {
	MODELS_LOG(context, "llama", "start, question=%s", user_input.c_str());
	const llama_vocab *vocab = llama_model_get_vocab(lm.model);
	uint32_t ctx_size = llama_n_ctx(lm.ctx);
	MODELS_LOG(context, "llama", "ctx_size=%u", ctx_size);

	// Cap each SQL result to at most 1/6 of the context window (in characters,
	// treating chars ≈ tokens as a conservative heuristic).  This leaves room
	// for the system prompt, the user question, and several rounds of history.
	// Never exceed the hard constant from the header.
	const size_t max_result_chars = std::min(MAX_RESULT_CHARS, static_cast<size_t>(ctx_size) / 6);
	MODELS_LOG(context, "llama", "max_result_chars=%llu", static_cast<unsigned long long>(max_result_chars));

	// Sampler: temperature 0.2 for fairly deterministic SQL generation
	auto sparams = llama_sampler_chain_default_params();
	llama_sampler *sampler = llama_sampler_chain_init(sparams);
	llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.2f));
	llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

	// Conversation history as parallel role/content string vectors
	// (llama_chat_message holds const char* so we need the strings to stay alive)
	std::vector<string> roles, contents;
	const string &sys = system_prompt_override.empty() ? SUBPROCESS_SYSTEM_PROMPT : system_prompt_override;
	roles.push_back("system");
	contents.push_back(sys);
	roles.push_back("user");
	contents.push_back(user_input);

	string final_response;
	int sql_rounds = 0;
	std::unordered_set<string> seen_queries; // detect any repeated query (cycles included)
	bool force_final = false;                // next response must be treated as final regardless of content

	// In single_shot_sql mode the worker retries on error but aborts cleanly on
	// loop detection or round limit — no force_final (sqlcoder can't produce prose).
	const int max_worker_rounds = single_shot_sql ? 6 : MAX_SQL_ROUNDS;
	string last_error_sql, last_error_text; // for single_shot_sql abort messages

	while (sql_rounds <= max_worker_rounds) {
		// Build prompt, pruning oldest SQL round-trips if it doesn't fit in context.
		// roles/contents layout: [system, user_question, asst_sql, user_result, asst_sql, user_result, ...]
		// The first two entries are always kept; pairs at index 2+3, 4+5, ... are prunable.
		std::vector<llama_chat_message> msgs;
		string prompt;
		std::vector<llama_token> tokens;

		while (true) {
			msgs.clear();
			msgs.reserve(roles.size());
			for (size_t i = 0; i < roles.size(); i++) {
				msgs.push_back({roles[i].c_str(), contents[i].c_str()});
			}
			prompt = ApplyChatTemplate(lm.model, msgs);
			tokens = Tokenize(vocab, prompt, /*add_special=*/true);

			if (tokens.empty()) {
				shell_state.ShellPrintError("Error: failed to tokenize prompt\n");
				goto done;
			}
			if (tokens.size() < ctx_size)
				break; // fits

			// Too long — drop the oldest SQL round-trip (indices 2 and 3) if possible.
			if (roles.size() > 2) {
				MODELS_LOG(context, "llama", "prompt too long (%llu tokens), pruning oldest SQL round",
				           static_cast<unsigned long long>(tokens.size()));
				roles.erase(roles.begin() + 2, roles.begin() + 4);
				contents.erase(contents.begin() + 2, contents.begin() + 4);
			} else {
				shell_state.ShellPrintError(
				    StringUtil::Format("Error: prompt too long (%llu tokens) even after pruning\n",
				                       static_cast<unsigned long long>(tokens.size())));
				goto done;
			}
		}

		MODELS_LOG(context, "llama", "round=%d prompt_tokens=%llu", sql_rounds,
		           static_cast<unsigned long long>(tokens.size()));

		// Clear KV cache and prefill
		{
			auto *mem = llama_get_memory(lm.ctx);
			if (!mem) {
				shell_state.ShellPrintError("Error: llama_get_memory() returned null\n");
				break;
			}
			llama_memory_clear(mem, /*data=*/true);
		}
		{
			llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
			auto t0 = std::chrono::steady_clock::now();
			MODELS_LOG(context, "llama", "prefill start: %llu tokens", static_cast<unsigned long long>(tokens.size()));
			if (llama_decode(lm.ctx, batch) != 0) {
				shell_state.ShellPrintError("Error: llama_decode() failed on prompt\n");
				break;
			}
			double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000;
			MODELS_LOG(context, "llama", "prefill done: %.0f ms", ms);
		}

		// Generate response tokens (never stream mid-loop — SQL queries would print to the user).
		string response = Generate(lm.ctx, vocab, sampler, shell_state, context, ctx_size,
		                           static_cast<uint32_t>(tokens.size()), /*stream=*/false);
		MODELS_LOG(context, "llama", "response(%llu chars): %.200s", static_cast<unsigned long long>(response.size()),
		           response.c_str());

		// If we forced a final answer, treat this response as-is regardless of content.
		string sql = force_final ? string() : ExtractSQLQuery(response);

		// Fallback: SQL-specialist models (e.g. sqlcoder) often output bare SQL
		// with no SQL_QUERY: prefix. Take the first non-empty line only — multi-line
		// responses indicate contaminated output (echoed context, special tokens, etc.).
		// Also strip chat-template tokens (e.g. <|im_end|>) before executing.
		if (sql.empty() && !force_final) {
			size_t s = response.find_first_not_of(" \t\n\r");
			if (s != string::npos) {
				size_t nl = response.find('\n', s);
				string first_line =
				    StripSpecialTokens(nl != string::npos ? response.substr(s, nl - s) : response.substr(s));
				if (!first_line.empty() && IsSafeSQL(first_line)) {
					sql = first_line;
					MODELS_LOG(context, "llama", "bare SQL fallback: '%s'", sql.c_str());
				}
			}
		}
		MODELS_LOG(context, "llama", "sql extracted: '%s'%s", sql.c_str(),
		           force_final ? " (force_final, sql ignored)" : "");

		if (!sql.empty() && IsSafeSQL(sql) && sql_rounds < MAX_SQL_ROUNDS) {
			auto request_final = [&](const string &reason, const string &msg) {
				MODELS_LOG(context, "llama", "%s", reason.c_str());
				roles.push_back("assistant");
				contents.push_back(response);
				roles.push_back("user");
				contents.push_back(msg);
				force_final = true;
			};

			// If the model wants more SQL but we've hit the round limit…
			if (sql_rounds + 1 >= max_worker_rounds) {
				if (single_shot_sql) {
					final_response = "SQL: " + sql + "\nError:\n" +
					                 (last_error_text.empty() ? "max retry rounds reached" : last_error_text);
					goto done;
				}
				request_final("max SQL rounds reached, requesting final answer",
				              "You have reached the maximum number of queries. "
				              "Do not issue any more SQL_QUERY: lines. "
				              "Write your final answer now based on everything you have gathered so far.");
				continue;
			}
			// If we've run this query before (catches consecutive repeats and cycles)…
			if (seen_queries.count(sql)) {
				if (single_shot_sql) {
					final_response =
					    "SQL: " + sql + "\nError:\n" + (last_error_text.empty() ? "stuck in loop" : last_error_text);
					goto done;
				}
				request_final("loop detected (query already seen), forcing final answer",
				              "You already have the result of that query. "
				              "Do not issue any more SQL_QUERY: lines. "
				              "Write your final answer now using the information you already have.");
				continue;
			}
			seen_queries.insert(sql);
			sql_rounds++;
			auto qresult = ExecuteQuery(*context.db, sql);
			if (stats) {
				stats->sql_total++;
				if (qresult.is_error)
					stats->sql_error++;
				else
					stats->sql_ok++;
			}

			string result_text = qresult.is_error
			                         ? "ERROR: SQL query failed.\n"
			                           "SQL:   " +
			                               sql +
			                               "\n"
			                               "Cause: " +
			                               qresult.text +
			                               "\n"
			                               "ACTION: Do NOT repeat the same query. Issue a corrected SQL_QUERY:.\n"
			                         : "Query result:\n" + qresult.text;

			// Cap result size to avoid blowing up the context on large outputs.
			static const char TRUNC_SUFFIX[] = "\n... (truncated)";
			static constexpr size_t TRUNC_SUFFIX_LEN = sizeof(TRUNC_SUFFIX) - 1;
			if (result_text.size() > max_result_chars) {
				result_text.resize(max_result_chars - TRUNC_SUFFIX_LEN);
				result_text += TRUNC_SUFFIX;
				MODELS_LOG(context, "llama", "result truncated to %llu chars",
				           static_cast<unsigned long long>(max_result_chars));
			}

			if (single_shot_sql) {
				if (!qresult.is_error) {
					// Success — return immediately.
					final_response = "SQL: " + sql + "\nResult:\n" + qresult.text;
					break;
				}
				// Error — track it and let the conversation extension below trigger a retry.
				last_error_sql = sql;
				last_error_text = qresult.text;
			}

			// Extend conversation: assistant's SQL turn + user's result turn
			roles.push_back("assistant");
			contents.push_back(response);
			roles.push_back("user");
			contents.push_back(result_text);
		} else if (single_shot_sql) {
			// sqlcoder produced no SQL — can't generate natural language.
			final_response = "Error: worker produced no SQL for this question.";
			goto done;
		} else {
			// Final answer — strip any stray SQL_QUERY: lines the model may have
			// emitted on a forced-final round, then stream.
			string clean;
			{
				size_t pos = 0;
				while (pos < response.size()) {
					size_t nl = response.find('\n', pos);
					string line = (nl == string::npos) ? response.substr(pos) : response.substr(pos, nl - pos + 1);
					if (line.find("SQL_QUERY:") == string::npos) {
						clean += line;
					}
					pos = (nl == string::npos) ? response.size() : nl + 1;
				}
				// Trim leading/trailing whitespace left by stripped lines.
				size_t s = clean.find_first_not_of(" \t\n\r");
				if (s == string::npos)
					clean.clear();
				else
					clean = clean.substr(s);
			}
			if (clean.empty()) {
				clean = "I was unable to gather enough information to answer.";
			}
			shell_state.ShellPrint(clean);
			shell_state.ShellPrint("\n");
			final_response = clean;
			break;
		}
	}
done:

	MODELS_LOG(context, "llama", "done, final_response(%llu chars): %.200s",
	           static_cast<unsigned long long>(final_response.size()), final_response.c_str());
	llama_sampler_free(sampler);
	return final_response;
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

// SQL-only worker prompt (sqlcoder, duckdb-nsql).
// These models output bare SQL — no explanation, no protocol wrapper needed.
const char *const SQLCODER_SYSTEM_PROMPT =
    "You are a DuckDB SQL expert. Your only job is to write a single DuckDB SQL query.\n"
    "Rules:\n"
    "- Output ONLY the SQL query. No explanation, no markdown, no comments.\n"
    "- The query must be valid DuckDB SQL.\n"
    "- Use only SELECT, WITH, FROM, SHOW, DESCRIBE, or EXPLAIN.\n"
    "- The question and schema will be provided. Use the schema to write the correct query.";

// Coordinator system prompt.
// Schema and context are pre-loaded into the conversation by phase-0 bootstrap —
// the coordinator does NOT need to discover them.
const char *const COORDINATOR_SYSTEM_PROMPT =
    "You are a DuckDB analysis coordinator. The database schema and context are already\n"
    "in your conversation — use them directly without re-querying.\n"
    "\n"
    "COMMANDS — output exactly one per response:\n"
    "  ASK_SQL: <sql>            Execute a SQL query (data retrieval, aggregations, lookups).\n"
    "  DELEGATE_SQL: <question>  Ask the SQL specialist to write a complex query.\n"
    "                            Schema is auto-injected. Specialist returns SQL+result or SQL+error.\n"
    "                            On error: rephrase with more specific constraints and re-delegate,\n"
    "                            or run the corrected SQL directly with ASK_SQL.\n"
    "  FINAL_ANSWER: <text>      Write the answer. Must contain real data, never placeholders.\n"
    "\n"
    "RULES:\n"
    "  - Schema is already in context. Do NOT run schema discovery queries.\n"
    "  - Output exactly one command per line. Never mix commands.\n"
    "  - FINAL_ANSWER must use actual query results — never write [list of...] or similar.\n"
    "  - If ASK_SQL fails, read [Error hints] and fix the SQL before retrying.\n"
    "  - When budget is low, issue FINAL_ANSWER immediately with what you have.\n"
    "\n"
    "COLUMN NAMES (common mistakes):\n"
    "  duckdb_tables()  → table_name, database_name, schema_name   (NOT 'name')\n"
    "  duckdb_columns() → column_name, table_name, database_name, data_type\n"
    "  information_schema.tables → table_catalog = database, table_schema = schema";

// ---------------------------------------------------------------------------
// RunCoordinatorLoop
// ---------------------------------------------------------------------------

// Parse duckdb_summary('schema') tab-separated output into CREATE TABLE DDL.
// Collapses large ENUMs and hard-caps the result so it fits in model context.
static string FormatSchemaAsCreateTable(const string &summary_text, size_t max_chars = 2000) {
	std::vector<string> table_order;
	std::map<string, std::vector<std::pair<string, string>>> tables;

	size_t pos = 0;
	bool header = true;
	while (pos < summary_text.size()) {
		size_t nl = summary_text.find('\n', pos);
		string line = nl == string::npos ? summary_text.substr(pos) : summary_text.substr(pos, nl - pos);
		pos = nl == string::npos ? summary_text.size() : nl + 1;

		if (header) {
			header = false;
			continue;
		}
		if (line.empty())
			continue;

		size_t t1 = line.find('\t');
		if (t1 == string::npos)
			continue;
		size_t t2 = line.find('\t', t1 + 1);
		if (t2 == string::npos)
			continue;
		size_t t3 = line.find('\t', t2 + 1);

		string section = line.substr(0, t1);
		string tbl = line.substr(t1 + 1, t2 - t1 - 1);
		string col = line.substr(t2 + 1, t3 == string::npos ? string::npos : t3 - t2 - 1);
		string type_str = t3 == string::npos ? string() : line.substr(t3 + 1);

		if (section != "schema" || tbl.empty() || col.empty())
			continue;

		// Collapse large ENUMs to save tokens.
		auto enum_pos = type_str.find("ENUM(");
		if (enum_pos != string::npos) {
			size_t depth = 0, i = enum_pos;
			while (i < type_str.size()) {
				if (type_str[i] == '(')
					depth++;
				else if (type_str[i] == ')') {
					if (--depth == 0) {
						i++;
						break;
					}
				}
				i++;
			}
			size_t n_vals = 1;
			for (size_t j = enum_pos; j < i; j++)
				if (type_str[j] == ',')
					n_vals++;
			if (n_vals > 5)
				type_str = type_str.substr(0, enum_pos) + "ENUM(" + to_string(n_vals) + " values)";
		}

		if (tables.find(tbl) == tables.end())
			table_order.push_back(tbl);
		tables[tbl].push_back({col, type_str});
	}

	string result;
	for (const auto &tname : table_order) {
		result += "CREATE TABLE " + tname + " (";
		const auto &cols = tables[tname];
		for (size_t i = 0; i < cols.size(); i++) {
			if (i > 0)
				result += ", ";
			result += cols[i].first + " " + cols[i].second;
		}
		result += ");\n";
	}
	if (result.empty())
		return summary_text; // fallback

	if (result.size() > max_chars) {
		result.resize(max_chars);
		result += "\n... (schema truncated)";
	}
	return result;
}

// Extract a single-line value after a marker at line start (e.g. "DELEGATE_SQL: <value>").
static string ExtractCommand(const string &text, const string &marker) {
	size_t pos = 0;
	while (pos < text.size()) {
		size_t found = text.find(marker, pos);
		if (found == string::npos)
			return "";
		bool at_line_start = (found == 0 || text[found - 1] == '\n');
		if (at_line_start) {
			size_t val = found + marker.size();
			while (val < text.size() && (text[val] == ' ' || text[val] == '\t'))
				val++;
			size_t end = text.find('\n', val);
			if (end == string::npos)
				end = text.size();
			// Trim trailing CR
			while (end > val && text[end - 1] == '\r')
				end--;
			return text.substr(val, end - val);
		}
		pos = found + 1;
	}
	return "";
}

// Extract everything after "FINAL_ANSWER:" (including subsequent lines).
static string ExtractFinalAnswer(const string &text) {
	static const string MARKER = "FINAL_ANSWER:";
	size_t pos = 0;
	while (pos < text.size()) {
		size_t found = text.find(MARKER, pos);
		if (found == string::npos)
			return "";
		bool at_line_start = (found == 0 || text[found - 1] == '\n');
		if (at_line_start) {
			size_t val = found + MARKER.size();
			while (val < text.size() && (text[val] == ' ' || text[val] == '\t'))
				val++;
			// Everything from here to end of response is the answer.
			string ans = text.substr(val);
			// Trim leading/trailing whitespace.
			size_t s = ans.find_first_not_of(" \t\n\r");
			size_t e = ans.find_last_not_of(" \t\n\r");
			if (s == string::npos)
				return "";
			return ans.substr(s, e - s + 1);
		}
		pos = found + 1;
	}
	return "";
}

string RunCoordinatorLoop(ClientContext &context, BaseShellState &shell_state, LlamaState &coordinator,
                          LlamaState &worker, const string &user_input, idx_t term_width, double time_budget_seconds,
                          int32_t max_rounds, const string &worker_style, InteractionStats *stats) {
	MODELS_LOG(context, "llama", "coordinator start, question=%s", user_input.c_str());
	auto t_start = std::chrono::steady_clock::now();

	const llama_vocab *cvocab = llama_model_get_vocab(coordinator.model);
	uint32_t cctx_sz = llama_n_ctx(coordinator.ctx);

	auto sparams = llama_sampler_chain_default_params();
	llama_sampler *sampler = llama_sampler_chain_init(sparams);
	llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.3f));
	llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

	// Coordinator conversation
	std::vector<string> roles, contents;
	roles.push_back("system");
	contents.push_back(COORDINATOR_SYSTEM_PROMPT);

	string accumulated; // results from DELEGATE_SQL / ASK_SQL rounds
	string final_response;
	std::vector<string> backing_sql;                    // successful queries, for verification footer
	std::unordered_set<string> seen_failed_delegations; // block re-delegation of exact questions that already failed
	std::unordered_set<string> seen_direct_sql;         // detect coordinator stuck on repeated ASK_SQL

	// ── Phase 0: auto-bootstrap schema + context ──────────────────────────────
	// Run unconditionally before the coordinator loop so the coordinator always
	// has accurate schema and doesn't need to figure out how to discover it.
	string schema_ddl; // CREATE TABLE DDL, re-injected into each sqlcoder call
	{
		auto ctx_qr = ExecuteQuery(*context.db, "FROM duckdb_summary('context')");
		if (!ctx_qr.is_error) {
			accumulated += "[Database context]\n" + ctx_qr.text + "\n";
		}
		auto schema_qr = ExecuteQuery(*context.db, "FROM duckdb_summary('schema')");
		if (!schema_qr.is_error && schema_qr.text != "(0 rows)\n") {
			// Full DDL for sqlcoder (cap at 2000 chars).
			schema_ddl = FormatSchemaAsCreateTable(schema_qr.text, 2000);
			// Shorter version for coordinator accumulated context (cap at 1500 chars).
			string schema_for_coord = FormatSchemaAsCreateTable(schema_qr.text, 1500);
			accumulated += "[Database schema]\n" + schema_for_coord + "\n";
			MODELS_LOG(context, "llama", "phase0: schema_ddl=%llu chars", (unsigned long long)schema_ddl.size());
		}
	}

	for (int round = 0; round < max_rounds; round++) {
		auto now = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(now - t_start).count();
		double remaining = time_budget_seconds - elapsed;
		int iters_rem = max_rounds - round;

		// Build user message for this coordinator turn.
		string user_msg;
		if (round == 0) {
			user_msg = "User question: " + user_input + "\n";
		}
		user_msg += StringUtil::Format("TIME_BUDGET_REMAINING: %.0fs  ITERATIONS_REMAINING: %d\n",
		                               std::max(0.0, remaining), iters_rem);
		if (!accumulated.empty()) {
			user_msg += "\nAccumulated context from previous steps:\n" + accumulated;
		}

		roles.push_back("user");
		contents.push_back(user_msg);

		// Build and tokenize prompt (prune oldest non-system pairs if too long).
		std::vector<llama_chat_message> msgs;
		std::vector<llama_token> tokens;
		string prompt;

		while (true) {
			msgs.clear();
			msgs.reserve(roles.size());
			for (size_t i = 0; i < roles.size(); i++)
				msgs.push_back({roles[i].c_str(), contents[i].c_str()});
			prompt = ApplyChatTemplate(coordinator.model, msgs);
			tokens = Tokenize(cvocab, prompt, true);
			if (tokens.empty()) {
				shell_state.ShellPrintError("Error: coordinator failed to tokenize prompt\n");
				goto done;
			}
			if (tokens.size() < cctx_sz)
				break;
			if (roles.size() > 2) {
				roles.erase(roles.begin() + 2, roles.begin() + 4);
				contents.erase(contents.begin() + 2, contents.begin() + 4);
			} else {
				shell_state.ShellPrintError("Error: coordinator prompt too long even after pruning\n");
				goto done;
			}
		}

		MODELS_LOG(context, "llama", "coordinator round=%d tokens=%llu remaining=%.0fs", round,
		           (unsigned long long)tokens.size(), remaining);

		// Prefill
		{
			auto *mem = llama_get_memory(coordinator.ctx);
			if (!mem) {
				shell_state.ShellPrintError("Error: llama_get_memory() null\n");
				break;
			}
			llama_memory_clear(mem, true);
		}
		{
			llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
			auto t0 = std::chrono::steady_clock::now();
			MODELS_LOG(context, "llama", "coordinator prefill start: %llu tokens",
			           static_cast<unsigned long long>(tokens.size()));
			if (llama_decode(coordinator.ctx, batch) != 0) {
				shell_state.ShellPrintError("Error: coordinator llama_decode failed\n");
				break;
			}
			double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000;
			MODELS_LOG(context, "llama", "coordinator prefill done: %.0f ms", ms);
		}

		// Generate coordinator response (don't stream — only stream the final answer).
		string response = Generate(coordinator.ctx, cvocab, sampler, shell_state, context, cctx_sz,
		                           static_cast<uint32_t>(tokens.size()), /*stream=*/false);
		MODELS_LOG(context, "llama", "coordinator response: %.300s", response.c_str());
		roles.push_back("assistant");
		contents.push_back(response);
		if (stats)
			stats->rounds++;

		// Parse command.
		string final_ans = ExtractFinalAnswer(response);
		string delegate_q = ExtractCommand(response, "DELEGATE_SQL:");
		string direct_sql = ExtractCommand(response, "ASK_SQL:");

		if (!final_ans.empty()) {
			shell_state.ShellPrint(final_ans);
			// Append backing SQL footer so the user can verify the answer.
			if (!backing_sql.empty()) {
				string footer = "\n\n---\nBacking queries:\n";
				for (size_t i = 0; i < backing_sql.size(); i++) {
					footer +=
					    StringUtil::Format("  [%llu] %s\n", static_cast<unsigned long long>(i + 1), backing_sql[i]);
				}
				shell_state.ShellPrint(footer);
				final_ans += footer;
			}
			shell_state.ShellPrint("\n");
			final_response = final_ans;
			break;

		} else if (!delegate_q.empty() && worker.IsLoaded()) {
			// Block re-delegation of the EXACT same question that already failed.
			// The coordinator may re-delegate with a rephrased/more-specific question.
			if (seen_failed_delegations.count(delegate_q)) {
				MODELS_LOG(context, "llama", "coordinator: exact failed delegation repeated for '%s', blocking",
				           delegate_q.c_str());
				accumulated += "\n[SYSTEM: This exact question already failed. "
				               "Rephrase with more specific SQL constraints, "
				               "or use ASK_SQL directly.]\n";
				continue;
			}
			if (stats)
				stats->delegations++;
			shell_state.ShellPrint(StringUtil::Format("[coordinator] delegating: %s\n", delegate_q));

			string worker_input = delegate_q;
			string worker_prompt = "";
			bool is_sqlcoder = (worker_style == "sqlcoder");

			if (is_sqlcoder) {
				worker_prompt = SQLCODER_SYSTEM_PROMPT;
				// Inject schema so sqlcoder knows the actual table/column names.
				// Using the DDL from phase-0 (sqlcoder was trained on CREATE TABLE format).
				if (!schema_ddl.empty()) {
					worker_input = "Question: " + delegate_q + "\n\nSchema:\n" + schema_ddl;
				}
			}

			string worker_ans =
			    RunLlamaLoop(context, shell_state, worker, worker_input, term_width, worker_prompt, is_sqlcoder, stats);
			// Parse worker response: "SQL: <sql>\nResult:\n..." or "SQL: <sql>\nError:\n..."
			size_t err_marker = worker_ans.find("\nError:\n");
			bool worker_failed = (err_marker != string::npos);

			// Track successful SQL for the verification footer.
			if (is_sqlcoder && worker_ans.substr(0, 4) == "SQL:") {
				size_t nl = worker_ans.find('\n');
				string wql = worker_ans.substr(4, nl == string::npos ? string::npos : nl - 4);
				size_t s = wql.find_first_not_of(" \t");
				if (s != string::npos && !worker_failed)
					backing_sql.push_back(wql.substr(s));
			}

			// On failure: record this delegation so exact repeats are blocked;
			// inject error hints for the coordinator.
			if (worker_failed) {
				seen_failed_delegations.insert(delegate_q);
				string error_text = worker_ans.substr(err_marker + 8);
				string hints = MatchErrorHints(error_text);
				accumulated += "\n[Worker answered \"" + delegate_q + "\"]\n" + worker_ans + "\n";
				if (!hints.empty()) {
					accumulated += "[Error hints]\n" + hints;
					MODELS_LOG(context, "llama", "injected worker error hints: %s", hints.c_str());
				}
			} else {
				accumulated += "\n[Worker answered \"" + delegate_q + "\"]\n" + worker_ans + "\n";
			}

		} else if (!direct_sql.empty()) {
			// Block the coordinator if it keeps issuing the exact same ASK_SQL after failure.
			if (seen_direct_sql.count(direct_sql)) {
				MODELS_LOG(context, "llama", "coordinator: repeated ASK_SQL blocked: %.120s", direct_sql.c_str());
				// Auto-run DESCRIBE on duckdb_tables/duckdb_columns if those are in the query,
				// so the coordinator can see the real column list and fix the query.
				string describe_hint;
				if (direct_sql.find("duckdb_tables") != string::npos) {
					auto dr = ExecuteQuery(*context.db, "DESCRIBE duckdb_tables()");
					if (!dr.is_error)
						describe_hint = "duckdb_tables() columns:\n" + dr.text;
				} else if (direct_sql.find("duckdb_columns") != string::npos) {
					auto dr = ExecuteQuery(*context.db, "DESCRIBE duckdb_columns()");
					if (!dr.is_error)
						describe_hint = "duckdb_columns() columns:\n" + dr.text;
				}
				accumulated += "\n[SYSTEM: Identical ASK_SQL blocked — this query already failed. "
				               "Fix the column names and retry.\n" +
				               describe_hint + "]\n";
				continue;
			}
			seen_direct_sql.insert(direct_sql);

			shell_state.ShellPrint(StringUtil::Format("[coordinator] SQL: %s\n", direct_sql));
			if (IsSafeSQL(direct_sql)) {
				auto qr = ExecuteQuery(*context.db, direct_sql);
				if (stats) {
					stats->sql_total++;
					if (qr.is_error)
						stats->sql_error++;
					else
						stats->sql_ok++;
				}
				if (!qr.is_error) {
					backing_sql.push_back(direct_sql);
					accumulated += "\n[Direct SQL: " + direct_sql + "]\n" + qr.text + "\n";
				} else {
					accumulated += "\n[Direct SQL: " + direct_sql + "]\nERROR: " + qr.text +
					               "\n"
					               "[SYSTEM: Query FAILED — you have no data from this step. "
					               "Fix the SQL before writing FINAL_ANSWER.]\n";
					string hints = MatchErrorHints(qr.text);
					if (!hints.empty()) {
						accumulated += "[Error hints]\n" + hints;
						MODELS_LOG(context, "llama", "injected ASK_SQL error hints: %s", hints.c_str());
					}
				}
			} else {
				accumulated += "\n[SQL rejected — only SELECT/FROM/... allowed]\n";
			}

		} else {
			// No recognized command — treat response as final answer.
			MODELS_LOG(context, "llama", "coordinator: no command found, treating as final answer");
			shell_state.ShellPrint(response);
			shell_state.ShellPrint("\n");
			final_response = response;
			break;
		}

		// Check time budget before next round.
		elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
		if (elapsed >= time_budget_seconds) {
			MODELS_LOG(context, "llama", "coordinator: time budget exhausted (%.1fs >= %.1fs)", elapsed,
			           time_budget_seconds);
			// Force a final-answer turn.
			roles.push_back("user");
			contents.push_back("Time budget exhausted. Write FINAL_ANSWER: now using what you have gathered.");
			// Re-run generation once more — don't recurse, just fall through to next iteration.
			// The iteration limit will also catch this.
		}
	}

done:
	// If no FINAL_ANSWER was produced (rounds exhausted, budget expired, or error),
	// synthesize a response from accumulated context rather than returning empty string.
	if (final_response.empty()) {
		if (!accumulated.empty()) {
			final_response = "I was unable to produce a complete answer within the budget. "
			                 "Here is what I gathered:\n" +
			                 accumulated;
		} else {
			final_response = "I was unable to answer the question within the given budget.";
		}
		shell_state.ShellPrint(final_response + "\n");
	}

	llama_sampler_free(sampler);
	MODELS_LOG(context, "llama", "coordinator done, final_response(%llu chars)",
	           (unsigned long long)final_response.size());
	return final_response;
}

// ---------------------------------------------------------------------------
// RunCoordinatorLoopWithWorker — coordinator is llama, worker is a callback.
// Identical logic to RunCoordinatorLoop except:
//   - No LlamaState &worker parameter; worker calls go through worker_fn.
//   - Schema is always injected into worker_input ("Question: ...\n\nSchema:\n...").
//   - No backing-SQL footer (worker output may be natural language, not SQL: format).
// ---------------------------------------------------------------------------

string RunCoordinatorLoopWithWorker(ClientContext &context, BaseShellState &shell_state, LlamaState &coordinator,
                                    const WorkerFn &worker_fn, const string &user_input, idx_t term_width,
                                    double time_budget_seconds, int32_t max_rounds, InteractionStats *stats) {
	MODELS_LOG(context, "llama", "coordinator_with_worker start, question=%s", user_input.c_str());
	auto t_start = std::chrono::steady_clock::now();

	const llama_vocab *cvocab = llama_model_get_vocab(coordinator.model);
	uint32_t cctx_sz = llama_n_ctx(coordinator.ctx);

	auto sparams = llama_sampler_chain_default_params();
	llama_sampler *sampler = llama_sampler_chain_init(sparams);
	llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.3f));
	llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

	std::vector<string> roles, contents;
	roles.push_back("system");
	contents.push_back(COORDINATOR_SYSTEM_PROMPT);

	string accumulated;
	string final_response;
	std::unordered_set<string> seen_failed_delegations;
	std::unordered_set<string> seen_direct_sql;

	// ── Phase 0: auto-bootstrap schema + context ─────────────────────────────
	string schema_ddl;
	{
		auto ctx_qr = ExecuteQuery(*context.db, "FROM duckdb_summary('context')");
		if (!ctx_qr.is_error) {
			accumulated += "[Database context]\n" + ctx_qr.text + "\n";
		}
		auto schema_qr = ExecuteQuery(*context.db, "FROM duckdb_summary('schema')");
		if (!schema_qr.is_error && schema_qr.text != "(0 rows)\n") {
			schema_ddl = FormatSchemaAsCreateTable(schema_qr.text, 2000);
			string schema_for_coord = FormatSchemaAsCreateTable(schema_qr.text, 1500);
			accumulated += "[Database schema]\n" + schema_for_coord + "\n";
			MODELS_LOG(context, "llama", "coordinator_with_worker phase0: schema_ddl=%llu chars",
			           (unsigned long long)schema_ddl.size());
		}
	}

	for (int round = 0; round < max_rounds; round++) {
		auto now = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(now - t_start).count();
		double remaining = time_budget_seconds - elapsed;
		int iters_rem = max_rounds - round;

		string user_msg;
		if (round == 0) {
			user_msg = "User question: " + user_input + "\n";
		}
		user_msg += StringUtil::Format("TIME_BUDGET_REMAINING: %.0fs  ITERATIONS_REMAINING: %d\n",
		                               std::max(0.0, remaining), iters_rem);
		if (!accumulated.empty()) {
			user_msg += "\nAccumulated context from previous steps:\n" + accumulated;
		}

		roles.push_back("user");
		contents.push_back(user_msg);

		// Build and tokenize prompt (prune oldest non-system pairs if too long).
		std::vector<llama_chat_message> msgs;
		std::vector<llama_token> tokens;
		string prompt;

		while (true) {
			msgs.clear();
			msgs.reserve(roles.size());
			for (size_t i = 0; i < roles.size(); i++)
				msgs.push_back({roles[i].c_str(), contents[i].c_str()});
			prompt = ApplyChatTemplate(coordinator.model, msgs);
			tokens = Tokenize(cvocab, prompt, true);
			if (tokens.empty()) {
				shell_state.ShellPrintError("Error: coordinator failed to tokenize prompt\n");
				goto done;
			}
			if (tokens.size() < cctx_sz)
				break;
			if (roles.size() > 2) {
				roles.erase(roles.begin() + 2, roles.begin() + 4);
				contents.erase(contents.begin() + 2, contents.begin() + 4);
			} else {
				shell_state.ShellPrintError("Error: coordinator prompt too long even after pruning\n");
				goto done;
			}
		}

		MODELS_LOG(context, "llama", "coordinator_with_worker round=%d tokens=%llu remaining=%.0fs", round,
		           (unsigned long long)tokens.size(), remaining);

		// Prefill
		{
			auto *mem = llama_get_memory(coordinator.ctx);
			if (!mem) {
				shell_state.ShellPrintError("Error: llama_get_memory() null\n");
				break;
			}
			llama_memory_clear(mem, true);
		}
		{
			llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
			auto t0 = std::chrono::steady_clock::now();
			MODELS_LOG(context, "llama", "coordinator_with_worker prefill start: %llu tokens",
			           static_cast<unsigned long long>(tokens.size()));
			if (llama_decode(coordinator.ctx, batch) != 0) {
				shell_state.ShellPrintError("Error: coordinator llama_decode failed\n");
				break;
			}
			double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000;
			MODELS_LOG(context, "llama", "coordinator_with_worker prefill done: %.0f ms", ms);
		}

		// Generate coordinator response (don't stream — only stream final answer).
		string response = Generate(coordinator.ctx, cvocab, sampler, shell_state, context, cctx_sz,
		                           static_cast<uint32_t>(tokens.size()), /*stream=*/false);
		MODELS_LOG(context, "llama", "coordinator_with_worker response: %.300s", response.c_str());
		roles.push_back("assistant");
		contents.push_back(response);
		if (stats)
			stats->rounds++;

		// Parse command.
		string final_ans = ExtractFinalAnswer(response);
		string delegate_q = ExtractCommand(response, "DELEGATE_SQL:");
		string direct_sql = ExtractCommand(response, "ASK_SQL:");

		if (!final_ans.empty()) {
			shell_state.ShellPrint(final_ans);
			shell_state.ShellPrint("\n");
			final_response = final_ans;
			break;

		} else if (!delegate_q.empty()) {
			if (seen_failed_delegations.count(delegate_q)) {
				MODELS_LOG(context, "llama",
				           "coordinator_with_worker: exact failed delegation repeated for '%s', blocking",
				           delegate_q.c_str());
				accumulated += "\n[SYSTEM: This exact question already failed. "
				               "Rephrase with more specific SQL constraints, "
				               "or use ASK_SQL directly.]\n";
				continue;
			}
			if (stats)
				stats->delegations++;
			shell_state.ShellPrint(StringUtil::Format("[coordinator] delegating: %s\n", delegate_q));

			// Inject schema so the worker knows the tables without needing to discover them.
			string worker_input = delegate_q;
			if (!schema_ddl.empty()) {
				worker_input = "Question: " + delegate_q + "\n\nSchema:\n" + schema_ddl;
			}

			string worker_ans = worker_fn(worker_input);
			size_t err_marker = worker_ans.find("\nError:\n");
			bool worker_failed = (err_marker != string::npos);

			if (worker_failed) {
				seen_failed_delegations.insert(delegate_q);
				string error_text = worker_ans.substr(err_marker + 8);
				string hints = MatchErrorHints(error_text);
				accumulated += "\n[Worker answered \"" + delegate_q + "\"]\n" + worker_ans + "\n";
				if (!hints.empty()) {
					accumulated += "[Error hints]\n" + hints;
					MODELS_LOG(context, "llama", "injected worker error hints: %s", hints.c_str());
				}
			} else {
				accumulated += "\n[Worker answered \"" + delegate_q + "\"]\n" + worker_ans + "\n";
			}

		} else if (!direct_sql.empty()) {
			if (seen_direct_sql.count(direct_sql)) {
				MODELS_LOG(context, "llama", "coordinator_with_worker: repeated ASK_SQL blocked: %.120s",
				           direct_sql.c_str());
				string describe_hint;
				if (direct_sql.find("duckdb_tables") != string::npos) {
					auto dr = ExecuteQuery(*context.db, "DESCRIBE duckdb_tables()");
					if (!dr.is_error)
						describe_hint = "duckdb_tables() columns:\n" + dr.text;
				} else if (direct_sql.find("duckdb_columns") != string::npos) {
					auto dr = ExecuteQuery(*context.db, "DESCRIBE duckdb_columns()");
					if (!dr.is_error)
						describe_hint = "duckdb_columns() columns:\n" + dr.text;
				}
				accumulated += "\n[SYSTEM: Identical ASK_SQL blocked — this query already failed. "
				               "Fix the column names and retry.\n" +
				               describe_hint + "]\n";
				continue;
			}
			seen_direct_sql.insert(direct_sql);

			shell_state.ShellPrint(StringUtil::Format("[coordinator] SQL: %s\n", direct_sql));
			if (IsSafeSQL(direct_sql)) {
				auto qr = ExecuteQuery(*context.db, direct_sql);
				if (stats) {
					stats->sql_total++;
					if (qr.is_error)
						stats->sql_error++;
					else
						stats->sql_ok++;
				}
				if (!qr.is_error) {
					accumulated += "\n[Direct SQL: " + direct_sql + "]\n" + qr.text + "\n";
				} else {
					accumulated += "\n[Direct SQL: " + direct_sql + "]\nERROR: " + qr.text +
					               "\n"
					               "[SYSTEM: Query FAILED — you have no data from this step. "
					               "Fix the SQL before writing FINAL_ANSWER.]\n";
					string hints = MatchErrorHints(qr.text);
					if (!hints.empty()) {
						accumulated += "[Error hints]\n" + hints;
						MODELS_LOG(context, "llama", "injected ASK_SQL error hints: %s", hints.c_str());
					}
				}
			} else {
				accumulated += "\n[SQL rejected — only SELECT/FROM/... allowed]\n";
			}

		} else {
			MODELS_LOG(context, "llama", "coordinator_with_worker: no command found, treating as final answer");
			shell_state.ShellPrint(response);
			shell_state.ShellPrint("\n");
			final_response = response;
			break;
		}

		// Check time budget before next round.
		elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
		if (elapsed >= time_budget_seconds) {
			MODELS_LOG(context, "llama", "coordinator_with_worker: time budget exhausted (%.1fs >= %.1fs)", elapsed,
			           time_budget_seconds);
			roles.push_back("user");
			contents.push_back("Time budget exhausted. Write FINAL_ANSWER: now using what you have gathered.");
		}
	}

done:
	if (final_response.empty()) {
		if (!accumulated.empty()) {
			final_response = "I was unable to produce a complete answer within the budget. "
			                 "Here is what I gathered:\n" +
			                 accumulated;
		} else {
			final_response = "I was unable to answer the question within the given budget.";
		}
		shell_state.ShellPrint(final_response + "\n");
	}

	llama_sampler_free(sampler);
	MODELS_LOG(context, "llama", "coordinator_with_worker done, final_response(%llu chars)",
	           (unsigned long long)final_response.size());
	return final_response;
}

} // namespace duckdb
