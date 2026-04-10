# Negotiation Challenge — Optimization Arena

> Part of the [Optimization Arena](https://www.optimizationarena.com/) hackathon by [Paradigm](https://www.paradigm.xyz/).  
> Official challenge repo: [paradigmxyz/negotiation-challenge](https://github.com/paradigmxyz/negotiation-challenge)

**Best Score:** 83.8% (submission_22)

## About the Challenge

The [Negotiation Challenge](https://www.optimizationarena.com/) is a multi-round resource negotiation game between AI agents. Write a strategy prompt for an AI agent that negotiates resource splits against a baseline. Your strategy prompt is injected as system instructions into a **Gemini** model, which then autonomously negotiates on your behalf.

Each game, two AI agents split a pool of **books, hats, and balls** with hidden valuations. Both agents see the full negotiation history, including all proposals, actions, and messages from every prior turn. Rounds include proposing splits, accepting, or rejecting. If no agreement is reached, both agents receive a penalty score.

- **Games:** 10 per evaluation, alternating who moves first
- **Scoring:** Mean percentage of maximum possible value received across all games
- **Model:** Google Gemini (strategy prompt as system instructions)

## Prompt Archive

This folder contains strategy prompts iterated across 24 submissions:

| Submission Range | Notes |
|-----------------|-------|
| `submission_01` – `submission_03` | Exact prompt text recovered from initial chat |
| `submission_04` – `submission_19` | Reconstructed from conversation notes and scoreboard history |
| `submission_20` – `submission_24` | Exact prompt text where available |

### Known Exact-Text Submissions

| File | Score |
|------|-------|
| `submission_20_78.7pct.txt` | 78.7% |
| `submission_21_68.8pct.txt` | 68.8% |
| **`submission_22_83.8pct.txt`** | **83.8%** ⭐ Best |
| `submission_23_77.1pct.txt` | 77.1% |
| `submission_24_82.2pct.txt` | 82.2% |

> Some later submissions reused the same prompt text with different random-seed outcomes.

## Best Strategy Summary (submission_22 — 83.8%)

The winning approach uses a simple scripted negotiation pattern:

1. Propose keeping ALL items + ask opponent's priorities
2. Propose keeping ALL items + state non-negotiable items
3. Propose keeping ALL items + request their valuations
4. ACCEPT opponent's latest proposal
5. If not closed, ACCEPT any offer immediately (always close the deal)

**Key insight:** Maximizing deal value matters, but an absolute deal is required—no deal results in a negative score. Willingness to accept closes deals reliably.

## Links

- 🌐 [Optimization Arena](https://www.optimizationarena.com/)
- 📦 [paradigmxyz/negotiation-challenge](https://github.com/paradigmxyz/negotiation-challenge)
