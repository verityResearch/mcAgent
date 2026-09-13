# Contributing to mcAgent

Thank you for your interest in contributing to mcAgent! This is a research project designed to explore tool-use and embodied agent architectures using small, locally runnable models.

## How to Contribute

As a research artifact, our primary focus is on advancing the methodology behind training small models to use verified oracles. 

### 1. Bug Reports and Feature Requests
- Please check existing issues to see if your bug or feature request has already been reported.
- If you find a new bug, open an issue with details about your environment (e.g., Minecraft version, OS, specific log outputs) and a clear description of the problem.

### 2. Pull Requests
If you'd like to submit a fix or a new feature:
1. **Fork the repository** and create a feature branch.
2. **Ensure consistency**: Follow the existing architectural principles. Specifically, remember that *the mod is the oracle and the bot is the agent*. Do not add complex agent logic to the Fabric mod.
3. **Verify locally**: Make sure that your changes don't break existing offline oracle builds (`tool_oracle/build_db.py`) or the live-server Fabric mod endpoints.
4. **Submit your PR** with a clear description of what changed and *why*. If your change is related to an open issue, link to it.

## Development Setup

### Python Tool Oracle
We recommend using a virtual environment.
```bash
# For development with GPU support:
pip install -r requirements-cuda.txt
```

### Fabric Mod
The Fabric mod in `mc_mod/` requires JDK 25.
```bash
cd mc_mod
./gradlew build
```

## License
By contributing to this repository, you agree that your contributions will be licensed under its Apache-2.0 license.
