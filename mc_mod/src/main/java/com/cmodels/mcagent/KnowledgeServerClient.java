package com.cmodels.mcagent;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

/**
 * Thin HTTP client for the stage-1 knowledge server (fact_pipeline/
 * tool_oracle_cuda/knowledge_server.py), the SAME service the Mineflayer
 * bot's postJson()/askKnowledgeServer() already call -- POST /ask.
 * Deliberately not a reimplementation of the agentic loop: the model, the
 * oracle, and the verification all stay in the Python service. This class
 * exists so a Fabric mod player gets identical oracle-assisted answers without
 * needing Node.js/mineflayer at all. Free-form answer text is not automatically
 * verified merely because an oracle lookup occurred.
 *
 * NOT YET VERIFIED LIVE from inside a running Minecraft client -- written
 * and (if the build succeeded) compiles against the API surface, but
 * in-game execution requires a running Minecraft client + server, which
 * this development environment cannot provide. Flagged honestly rather
 * than claimed proven; see the stage-6 diff summary for what's actually
 * been checked.
 */
final class KnowledgeServerClient {
	static final String DEFAULT_BASE_URL = "http://127.0.0.1:8420";

	private static final HttpClient HTTP = HttpClient.newBuilder()
		.connectTimeout(Duration.ofSeconds(5))
		.build();

	private KnowledgeServerClient() {
	}

	/**
	 * POST {"question": "..."} to /ask, return the "answer" field. Matches
	 * bot.js's askKnowledgeServer() request/response shape exactly (see
	 * fact_pipeline/mc_bot/bot.js and fact_pipeline/tool_oracle_cuda/
	 * knowledge_server.py's AskRequest/AskResponse models) so this mod and
	 * the Mineflayer bot are interchangeable clients of the same backend.
	 */
	static CompletableFuture<String> askAsync(String question) {
		JsonObject body = new JsonObject();
		body.addProperty("question", question);

		HttpRequest request = HttpRequest.newBuilder()
			.uri(URI.create(DEFAULT_BASE_URL + "/ask"))
			.header("Content-Type", "application/json")
			.timeout(Duration.ofSeconds(30))
			.POST(HttpRequest.BodyPublishers.ofString(body.toString()))
			.build();

		return HTTP.sendAsync(request, HttpResponse.BodyHandlers.ofString())
			.thenApply(response -> {
				if (response.statusCode() != 200) {
					throw new RuntimeException("knowledge server HTTP " + response.statusCode());
				}
				JsonObject parsed = JsonParser.parseString(response.body()).getAsJsonObject();
				return parsed.get("answer").getAsString();
			});
	}
}
