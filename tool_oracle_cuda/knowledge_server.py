#!/usr/bin/env python3
"""Stage 1 of the embodied-agent plan: wrap the tool-augmented model behind a
stable JSON-schema HTTP service. This is the seam the Mineflayer bot (stage 2)
and any future embodiment adapter plug into -- POST a question, get an
answer composed after zero or more offline-oracle lookups. The model can still
skip or misuse the tool, so callers must not treat free-form text as verified
solely because it came through this service.

Reuses the CUDA agentic loop from eval_tool_skill_cuda.py unchanged (same
model loading, same answer_question_cuda hop loop) -- this is a thin HTTP
wrapper, not a reimplementation, so it inherits the same parsing and evidence
boundaries.

    python3 knowledge_server.py --adapter adapter-tool-v11-cuda --db minecraft.db --port 8420

    curl -X POST http://127.0.0.1:8420/ask -H 'Content-Type: application/json' \
        -d '{"question": "What do I need to craft a torch?"}'
"""
import argparse
import sys

import torch
import uvicorn
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from eval_tool_skill_cuda import make_answer_question_cuda  # noqa: E402

sys.path.insert(0, "tool_oracle")
from lookup import OracleDB  # noqa: E402

MODEL_ID = "Qwen/Qwen3-1.7B"

app = FastAPI(title="tool-oracle knowledge server")
_state = {}


class AskRequest(BaseModel):
    question: str
    max_tokens: int = 256
    max_hops: int = 3


class AskResponse(BaseModel):
    answer: str
    tool_calls: list
    n_calls: int


class RecipeRequest(BaseModel):
    item: str


class RecipeResponse(BaseModel):
    item: str
    ingredients: list
    found: bool


class AllRecipesResponse(BaseModel):
    recipes: dict  # {result_item: [ingredient, ...]}


class TagMembersResponse(BaseModel):
    tag: str
    members: list


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    res = _state["answer_question"](
        _state["model"], _state["tok"], _state["db"], req.question,
        max_tokens=req.max_tokens, max_hops=req.max_hops)
    return AskResponse(answer=res["answer"], tool_calls=res["calls"], n_calls=res["n_calls"])


@app.post("/lookup_recipe", response_model=RecipeResponse)
def lookup_recipe(req: RecipeRequest) -> RecipeResponse:
    """Direct oracle DB query, no model in the loop. For callers (e.g. the
    stage-3 state-tool composition) that want structured ingredients rather
    than a natural-language sentence -- still verified, just skipping the LLM
    hop since it isn't needed to answer this shape of question."""
    ingredients = _state["db"].recipe_ingredients(req.item)
    return RecipeResponse(item=req.item, ingredients=ingredients, found=bool(ingredients))


@app.get("/all_recipes", response_model=AllRecipesResponse)
def all_recipes():
    """Bulk dump of the recipe table -- {result_item: [ingredients]} for every
    recipe. For stage-5 task generation: the trace generator needs the whole
    recipe graph client-side to compute DAG depth (an item's craft difficulty
    = how many crafting steps below it before every ingredient is a raw/base
    material), not one lookup at a time. Excludes recipe-filename variants
    ("_from_...") the same way gen_query_traces.py does -- those are the
    recipe FILE id, not the crafted item."""
    rows = _state["db"].con.execute(
        "SELECT result_item, ingredient FROM recipe ORDER BY result_item").fetchall()
    recipes: dict = {}
    for result_item, ingredient in rows:
        if "_from_" in result_item:
            continue
        recipes.setdefault(result_item, []).append(ingredient)
    return AllRecipesResponse(recipes=recipes)


@app.get("/tag_members", response_model=TagMembersResponse)
def tag_members(tag: str):
    """Direct oracle DB query, no model in the loop -- for the stage-5 recipe-
    DAG builder, which needs to resolve tag-based ingredients (e.g. a recipe
    listing '#minecraft:planks' rather than one concrete plank item) to
    concrete items to determine real tier-1 reachability."""
    members = _state["db"].tag_members(tag)
    return TagMembersResponse(tag=tag, members=members)


@app.get("/health")
def health():
    return {"status": "ok", "model": _state.get("model_id"), "adapter": _state.get("adapter")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--db", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    args = ap.parse_args()

    print(f"loading {args.model}" + (f" + adapter {args.adapter}" if args.adapter else " (base)"))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    db = OracleDB(args.db)

    _state["model"] = model
    _state["tok"] = tok
    _state["db"] = db
    _state["model_id"] = args.model
    _state["adapter"] = args.adapter
    _state["answer_question"] = make_answer_question_cuda(model, tok)

    print(f"ready -- serving on http://{args.host}:{args.port}/ask")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
