# GEPA: Reflective Prompt Optimization

This project explores **GEPA (Genetic-Pareto)**, a reflective prompt optimization framework that improves LLM prompts without modifying the underlying model weights.

The project was built to understand GEPA from both a conceptual and implementation perspective, with particular focus on:

- Reflective prompt mutation
- Evaluator-driven feedback
- Pareto-based candidate selection
- Candidate ancestry
- Validation of optimized prompts
- The importance of diagnostic feedback in prompt optimization

A simple customer-support classification task is used to demonstrate how GEPA can transform execution failures into improved natural-language instructions.

---

## Project Overview

Large Language Models are highly sensitive to the instructions provided in their prompts.

A weak or underspecified prompt can cause an otherwise capable model to:

- return the wrong output format,
- misunderstand business rules,
- classify examples incorrectly,
- or behave inconsistently across inputs.

Traditional prompt engineering usually involves a manual cycle:

```text
Observe failure
      ↓
Diagnose the problem
      ↓
Rewrite the prompt
      ↓
Test again



