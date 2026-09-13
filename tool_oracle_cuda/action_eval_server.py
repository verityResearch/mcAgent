#!/usr/bin/env python3
"""Stage 5, the actual ask: "same two
gates, held-out tasks the generator never sourced" -- genuine LIVE
re-execution evaluation, not the teacher-forced probe
(probe_action_first_hop.py) or held-out training loss
(prepare_action_corpus.py). Those were real, cheap intermediate signals;
this is the real thing.

This process is the "brain" half only -- a thin HTTP wrapper exposing ONE
model turn at a time, given real conversation history. It does NOT drive
the hop loop or execute anything itself. mc_bot/
live_eval_action_tasks.js is the "body" half: it owns the loop, calls this
server once per hop, EXECUTES whatever tool call comes back for real
against a live Minecraft server via action_tools.js, and feeds the REAL
result back as the next turn -- mirroring this project's own "mod is the
oracle, bot is the agent" split (the reviewer's stage-6 architecture reply) at a
smaller scale: this server is stateless per request, all real-world state
and gate-2 verification live on the Node side, matching how
knowledge_server.py's own /ask already keeps the model and the real state
source in separate processes.

Deliberately NOT reusing knowledge_server.py's /ask (which runs its own
internal multi-hop loop against the offline DB via
eval_tool_skill_cuda.py's answer_question_cuda) -- that function owns its
own loop and calls lookup() internally; this needs a single EXTERNAL turn
at a time so the Node side can execute REAL actions between hops instead
of DB lookups.

    python3 action_eval_server.py --adapter adapter-tool-v12-action-cuda --port 8423
"""
import argparse

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"

app = FastAPI(title="action-tool live-eval brain")
_state = {}


class Message(BaseModel):
    role: str
    content: str


class NextTurnRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 128


class NextTurnResponse(BaseModel):
    content: str


@app.post("/next_turn", response_model=NextTurnResponse)
def next_turn(req: NextTurnRequest) -> NextTurnResponse:
    tok = _state["tok"]
    model = _state["model"]
    convo = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = tok.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    # Greedy decode (do_sample=False), per the read: generation diversity is valuable in the trace GENERATOR
    # (gen_action_traces.js's own model calls, unaffected by this file), but
    # in an EVAL it's only variance to average out -- at temperature=0.2 this
    # server could not tell a real score delta from sampling noise (confirmed
    # live: two tasks that flipped success on a guard-fix re-run had never
    # triggered the guard at all, meaning the "+2" was noise, not the fix).
    # Greedy decode makes every future eval run against this adapter
    # deterministic and repeatable, so a score change can only mean a real
    # code/data change, not a different sample.
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=req.max_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id)
    content = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    return NextTurnResponse(content=content)


@app.get("/health")
def health():
    return {"status": "ok", "model": _state.get("model_id"), "adapter": _state.get("adapter")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8423)
    args = ap.parse_args()

    print(f"loading {args.model} + adapter {args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    _state["tok"] = tok
    _state["model"] = model
    _state["model_id"] = args.model
    _state["adapter"] = args.adapter

    print(f"ready -- serving on http://{args.host}:{args.port}/next_turn")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
