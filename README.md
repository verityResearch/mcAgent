# mcAgent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Minecraft Version](https://img.shields.io/badge/Minecraft-26.2-brightgreen.svg)]()
[![Model Size](https://img.shields.io/badge/Model-1.7B-orange.svg)]()

**mcAgent** is a local, lightweight AI agent designed for Minecraft. Instead of relying on the general knowledge baked into pre-trained models, mcAgent uses a small 1.7B parameter model (Qwen3-1.7B + LoRA) trained specifically to query a verified Minecraft oracle. This allows it to answer Minecraft-related questions accurately based on actual game data.

## Features

- **Offline Oracle**: An SQLite-based oracle built from Minecraft's own generated data reports.
- **Live Server Integration**: A Fabric mod that can walk a running server's registries, enabling support for modpacks and live game states.
- **Small Model Architecture**: Uses a fine-tuned Qwen3-1.7B model that is efficient enough to run locally while remaining highly accurate through tool usage.
- **Verifiable Traces**: Training data is self-verified against the real database, teaching the model to rely on tool calls rather than hallucinating facts.
- **Mineflayer Bot Integration**: An experimental HTTP-driven Mineflayer bot capable of consuming the oracle to generate and execute multi-step action traces.

## Components

- **`tool_oracle/`** & **`tool_oracle_cuda/`**: The core oracle implementation (`build_db.py`, `lookup.py`) and training data generators. Includes scripts for self-verifying traces and training the LoRA adapter.
- **`mc_mod/`**: A Fabric mod for live server interaction. Adds commands like `/mcagent ask` (queries the oracle) and `/mcagent builddb` (extracts live server registry data).
- **`mc_bot/`**: A Mineflayer bot that connects to the `knowledge_server` and executes verified actions.
- **`docs/`**: Contains design documents, methodology reports, and evaluation data detailing the training and architecture decisions.

## Quickstart

### 1. Installation

Install dependencies. Use `requirements-cuda.txt` if you plan to train or run the model with GPU acceleration. For CPU-only generation and testing, use `requirements.txt`.

```bash
pip install -r requirements-cuda.txt
```

### 2. Build the Oracle

Build the offline SQLite oracle from a pinned Minecraft data report:

```bash
python3 tool_oracle/build_db.py <data_report_dir> minecraft.db
```

### 3. Generate Training Data & Train

Generate the training data (which is automatically self-verified against the database) and train the LoRA adapter:

```bash
python3 tool_oracle/build_training_set.py --db minecraft.db --out-dir data-tool
python3 tool_oracle_cuda/train_lora_cuda.py --train data-tool/train.jsonl --valid data-tool/valid.jsonl --out-dir adapter
```

### 4. Evaluate

Evaluate the model against held-out data to verify tool usage and accuracy.

```bash
# Download the base model
hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

# Run the evaluation
python3 tool_oracle_cuda/eval_tool_skill_cuda.py \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-files-only \
  --adapter adapter \
  --db minecraft.db \
  --held-out \
  --train-jsonl data-tool/train.jsonl \
  --tool-manifest-sha256 $(sha256sum tool_oracle/eval_tool_manifest.txt | awk '{print $1}') \
  --out tool_oracle_cuda/held_out_adapter.json
```

### 5. Serve

Start the knowledge server and test it with a query:

```bash
python3 tool_oracle_cuda/knowledge_server.py --adapter adapter --db minecraft.db --port 8420

# In another terminal:
curl -X POST localhost:8420/ask -d '{"question": "What do I need to craft a torch?"}'
```

*Note: The Fabric mod (`mc_mod/`) builds separately via `./gradlew build` (JDK 25) and talks to this same knowledge server. See `mc_mod/README.md` for details.*

## Project Status & Limitations

mcAgent is a research artifact built to explore tool-use in small models for embodied agents. It is not currently maintained as a production-ready Minecraft mod.

- **Minecraft Version**: Pinned to Minecraft 26.2.
- **Known Issues**:
  - The Mineflayer bot cannot currently connect to 26.2 servers due to an upstream protocol gap. Server-side features (`builddb`, `ask`, and the HTTP oracle endpoint) are unaffected.
  - Evaluation scores should be interpreted alongside manual verification, as early evaluation schema flaws missed some recipe structure bugs.

## Authorship & Acknowledgments

This project was built as a collaborative effort between a human operator and AI (Anthropic's Claude). The operator guided the conception, direction, and verification, while the implementation was substantially machine-authored. Architecture and evaluation design went through review by cairn, a persistent Claude research collaborator; code comments that say "per the review" refer to that process.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
