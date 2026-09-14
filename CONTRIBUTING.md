# Contributing to mcAgent

Thanks for your interest. mcAgent is a research project exploring tool use and embodied agents with small, locally runnable models, so the most valuable contributions sharpen the method: better verification, better evaluation, clearer failure analysis.

## Reporting bugs and results

- Check existing issues first.
- Include your environment: Minecraft version, OS, GPU if relevant, and the relevant log output.
- **Play-test findings are especially welcome.** See [mc_mod/PLAYTEST.md](mc_mod/PLAYTEST.md). A case where the agent invents an answer instead of declining is the single most useful report.
- If you rerun an evaluation and get different numbers, open an issue with the exact command, model revision, and adapter hash.

## Pull requests

1. Fork the repository and create a feature branch.
2. **Keep the architecture:** *the mod is the oracle, the bot is the agent.* Don't add agent logic to the Fabric mod.
3. **Keep training data verifiable:** every tool observation in a trace must be a real return value, and outcomes are checked against real state.
4. **Run the tests:** `python3 -m pytest` (and `./gradlew build` in `mc_mod/` if you touched the mod).
5. Describe what changed and *why*, and link any related issue.

## Development setup

### Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # CPU: oracle, trace generation, tests
pip install -r requirements-cuda.txt   # GPU: training, evaluation, serving
```

### Fabric mod

Requires JDK 25.

```bash
cd mc_mod
./gradlew build
```

### Mineflayer bot

```bash
cd mc_bot
npm install
```

## License

By contributing, you agree that your contributions are licensed under the repository's Apache-2.0 license.
