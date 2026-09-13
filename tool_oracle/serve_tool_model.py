"""Local single-user REPL for the tool-augmented Minecraft model (MLX).

Loads the base model + a LoRA adapter and drives the SAME agentic loop the eval
uses (answer_question): read a question, let the model emit lookup() calls, run
each against the offline oracle (minecraft.db), feed the real result back, and
print the composed answer along with the lookups it made. This is a dev/demo
tool and executes only read-only DB lookups. The model can still skip a lookup
or add unsupported prose, so the printed answer is not automatically verified.

    PYTHONPATH=. python3 tool_oracle/serve_tool_model.py \
        --model <base> --db minecraft.db --adapter adapter-tool-v6

Type a question at the prompt; Ctrl-D or "quit" to exit. --show-calls prints the
lookups; --once "<question>" answers a single question and exits (for scripting).
"""
import argparse
import sys

from tool_oracle.eval_tool_skill import answer_question
from tool_oracle.lookup import OracleDB


def _format(res, show_calls: bool) -> str:
    lines = []
    if show_calls:
        if res["calls"]:
            for i, (table, key) in enumerate(res["calls"], 1):
                lines.append(f"  → lookup {i}: {table}({key})")
        else:
            lines.append("  → (no lookup emitted)")
    lines.append(res["answer"] or "(no answer)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--show-calls", action="store_true", help="print the lookups the model made")
    ap.add_argument("--once", default=None, help="answer a single question and exit")
    args = ap.parse_args()

    from mlx_lm import load

    db = OracleDB(args.db)
    model, tok = load(args.model, adapter_path=args.adapter) if args.adapter else load(args.model)

    def respond(q: str) -> None:
        res = answer_question(model, tok, db, q, args.max_tokens, args.max_hops)
        print(_format(res, args.show_calls))

    if args.once is not None:
        respond(args.once)
        return

    print("Tool-augmented Minecraft model. Ask a question (Ctrl-D or 'quit' to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            print()
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        respond(q)


if __name__ == "__main__":
    main()
