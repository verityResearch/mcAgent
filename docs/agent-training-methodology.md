# Debugging a Tool-Using Agent: A Methodology for Diagnosing Failure Modes

## Overview

Developing and debugging embodied agents with tool-use capabilities presents unique challenges. When an agent fails a task, the root cause could stem from the model's reasoning, gaps in the training corpus, flaws in the evaluation harness, or bugs within the tools themselves. Misdiagnosing the cause can lead to wasted training cycles or misdirected engineering effort. 

This document outlines a domain-independent, repeatable methodology for isolating and identifying these failure modes. Developed during the embodied-agent track for mcAgent, this process led to a Minecraft agent capable of completing 76% of held-out multi-step crafting tasks via genuine live execution.

## The Diagnostic Loop

1. **Measure with a Deterministic Baseline**
   Relying on aggregate scores with sampling-temperature decoding makes real progress indistinguishable from statistical noise. Switch evaluation to greedy decoding to turn measurements into fixed points rather than samples. Furthermore, report metrics per-stratum (or per-bucket) rather than as a single aggregate. This ensures that real signals—such as a specific category of tasks failing consistently—are not averaged away.

2. **Diagnose from Transcripts, Not Scores**
   A flat or negative score does not explain *why* a failure occurred. It is critical to read the actual generation transcripts before accepting a metric. 
   - A flat score might hide the fact that the model is performing correctly but is blocked by a harness bug.
   - A drop in score might be caused by unrelated, independently-explainable issues rather than a regression from a recent fix.
   - Always ask: "Is something already in place stopping the new behavior?" Previously shipped fixes (like safety guards) can become silent blockers for new capabilities.

3. **Identify Gaps in the Corpus**
   Models reliably reproduce what the corpus demonstrates and fail when encountering scenarios the corpus never shows. 
   - If the corpus lacks examples of querying before crafting, the model will rely on memorized knowledge instead of the oracle.
   - If the corpus lacks examples of recovering from a failure, the model will simply halt after diagnosing an error.
   Before training on a verified corpus, proactively identify what behaviors are *not* demonstrated. This gap is highly predictive of future failure modes.

4. **Enhance Tool Feedback over Model Training**
   Every dead end should return a useful observation. This is a core design principle for any tool-using agent.
   - A failed tool call should explain *what* is missing, not just that it failed (e.g., "missing ingredient: stick" instead of "no known recipe").
   - An unrecognized call should explicitly state that the function does not exist.
   - A repeated call should inform the agent that the action was already attempted.
   Improving the quality of tool feedback is often cheaper and more effective than running another training cycle.

5. **Record Predictions Before Measuring**
   Formulate and record hypotheses before running an evaluation. A prediction made beforehand is falsifiable evidence; an explanation crafted afterward is often just a story fit to the data. Recording wrong predictions can produce sharper, more specific findings than confirmed guesses.

## Key Findings & Architecture Principles

### The Two-Gate Model and Tool Verification
This project relies on a verified-corpus discipline with two gates:
1. **Authenticity**: Every observation in a trace must be a real return value from a tool.
2. **Outcome**: Task success is verified against real-world state (e.g., actual inventory or position), not inferred from the tool's claim.

However, neither gate verifies the underlying correctness of the tool itself. If a tool silently fails but reports success (e.g., an `attack()` tool that doesn't check distance), the corpus will include false positives.
**Principle**: Before a tool contributes to a training trace, its success signal must be confirmed against independent world state at least once.

### The Tool Ceiling
A model cannot learn past the limitations of its tools. If an agent consistently fails at a specific task across multiple training interventions, the issue likely resides in the tool layer, not the model's reasoning or planning.
**Principle**: The verified oracle removes the knowledge ceiling, and recovery traces remove the planning ceiling, but neither can bypass a tool ceiling. Independent tests of the tool layer—isolated from the model—are necessary to diagnose these limits.

## Standing Checks for Future Development

- **Independent Tool Verification**: Always check a tool's success signal against real-world state before trusting it in a trace.
- **Cache Invalidation for Guards**: If a guard caches a "failed" state to prevent infinite loops, ensure it tracks what could invalidate that state. The world changes, and a previously failed action might become valid.
- **Informative Dead Ends**: Ensure all tool failures return actionable, specific observations.
- **Predictive Gap Analysis**: Continually ask what the verified corpus *does not* show.
- **Falsifiable Predictions**: Write down hypotheses before measuring, and report both confirmed and falsified predictions to build a trustworthy diagnostic record.
