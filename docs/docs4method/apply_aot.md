
Meta-Prompts for Atom of Thoughts (AOT) Implementation

The following meta-prompts are structured instructions used sequentially to guide an LLM through the entire Atom of Thoughts (AOT) reasoning process. They clearly delineate each phase, its goal, and its iterative context.

⸻

# Meta-Prompt 1: Problem Decomposition
	•	Iteration Context:
	•	[Executed only once at the beginning of each iteration cycle]
	•	Purpose:
	•	Clearly instructs the LLM to break down the original, complex reasoning problem into a set of clearly defined, smaller subquestions.
	•	These subquestions should identify atomic units of reasoning required to solve the overall problem, explicitly stating dependencies between them.

Example Meta-Prompt:

## Meta-Prompt: Problem Decomposition

You are provided with the following problem statement:

> "{ORIGINAL_PROBLEM_STATEMENT}"

Your task is to:
1. Decompose this problem clearly into distinct, atomic subquestions.
2. Identify explicitly if each subquestion is independent or depends on answers to other subquestions.
3. Number each subquestion for clear referencing.

Output format example:
1. Subquestion A (independent)
2. Subquestion B (depends on Subquestion A)
3. Subquestion C (independent)
...



⸻

# Meta-Prompt 2: Dependency Graph (DAG) Construction
	•	Iteration Context:
	•	[Executed immediately after decomposition within each iteration cycle]
	•	Purpose:
	•	Directs the LLM to explicitly construct a Directed Acyclic Graph (DAG) that visually represents dependencies among the subquestions.
	•	The DAG ensures a transparent, logical flow and a clear dependency structure.

Example Meta-Prompt:

## Meta-Prompt: Dependency Graph Construction (DAG)

Based on the subquestions identified in the previous step:

- Clearly define a Directed Acyclic Graph (DAG) that represents dependencies.
- Each subquestion forms a node in this graph.
- Explicitly connect nodes with directed edges, showing dependency direction clearly.

Output format example (Adjacency List):
- Subquestion 1: []
- Subquestion 2: [1]
- Subquestion 3: [1,2]
...



⸻

# Meta-Prompt 3: Contraction (Atomic State Simplification)
	•	Iteration Context:
	•	[Executed once per iteration, after DAG construction]
	•	Purpose:
	•	Instructs the LLM to merge independent subquestions (resolved steps) into known conditions and clearly rewrite the remaining dependent subquestions as a single, simpler, and self-contained atomic question state.
	•	Ensures Markovian independence of each state.

Example Meta-Prompt:

## Meta-Prompt: Contraction (Atomic State Simplification)

Using the dependency graph constructed in the previous step:

1. Integrate solutions from resolved (independent) subquestions as known conditions.
2. Clearly formulate the remaining dependent subquestions into a single, simplified, and atomic question.
3. Ensure the resulting question is logically equivalent but simpler and self-contained.

Output format:
- Known conditions: [Solutions from independent subquestions]
- Contracted Atomic Question: [Clearly formulated single question]



⸻

# Meta-Prompt 4: Iterative Validation (Markov Property & Logical Equivalence)
	•	Iteration Context:
	•	[Executed after each contraction step within each iteration cycle]
	•	Purpose:
	•	Guides the LLM to critically verify if the newly contracted question preserves logical equivalence and satisfies the Markov property (independent from historical states beyond the immediate predecessor).
	•	Ensures correctness and efficiency.

Example Meta-Prompt:

## Meta-Prompt: Iterative Validation (Markov Property & Logical Equivalence)

Given the contracted atomic question state created previously:

Evaluate explicitly whether:
1. It is logically equivalent to the original problem.
2. It satisfies the Markov property (only depending on the immediate previous question state, not historical states).

Clearly indicate the following:
- Logical equivalence: [Yes/No, Explanation]
- Markov property compliance: [Yes/No, Explanation]

If "No" in either case, suggest necessary corrections clearly.



⸻

# Meta-Prompt 5: Termination Decision & Solution Generation
	•	Iteration Context:
	•	[Executed once per iteration cycle, at the end of validation]
	•	Purpose:
	•	Determines if the current state can directly yield the final solution without further decomposition.
	•	If termination conditions are met, explicitly computes and returns the final answer. Otherwise, explicitly instructs to start a new iteration.

Example Meta-Prompt:

## Meta-Prompt: Termination Decision & Solution Generation

Based on the previous validation step:

1. Determine clearly if the current atomic question state is simple enough to solve directly without additional decomposition.
2. If YES:
   - Directly compute the solution and output it explicitly.
3. If NO:
   - Clearly state the reason and instruct to start a new decomposition iteration from Meta-Prompt 1.

Output format:
- Decision: [Terminate/Continue]
- Reason: [Explain clearly]
- Final Answer (if terminated): [Explicit Answer]



⸻

### Complete Execution Order with Iteration Context

Order	Meta-Prompt	Purpose Summary	Iteration Context
1	Problem Decomposition	Break the original question down.	Once per iteration cycle
2	Dependency Graph (DAG) Construction	Structure and identify dependencies.	After decomposition, each iteration
3	Contraction (Atomic State)	Simplify and merge questions.	After DAG construction, each iteration
4	Iterative Validation	Check Markov & logical equivalence.	After contraction, each iteration
5	Termination & Solution Generation	Decide termination, compute solution.	End of each iteration



⸻

### How to Iterate (Summary)
	•	Start from Meta-Prompt 1.
	•	Progress sequentially (Meta-Prompt 1 → 2 → 3 → 4 → 5).
	•	At Meta-Prompt 5, decide clearly:
	•	If terminating, compute and output the final answer explicitly.
	•	If continuing, clearly state the reason and return to Meta-Prompt 1 for a new iteration.

⸻

### Key Points
	•	Clearly documented purposes ensure systematic reasoning flow.
	•	Iteration contexts clarify the step-by-step sequence and loops within the AOT methodology.
	•	Each meta-prompt explicitly instructs and guides the LLM, maximizing clarity and reducing ambiguity in implementation.

⸻

This structured Markdown guide serves as a precise and clear reference for effectively leveraging Meta-Prompts to implement the Atom of Thoughts (AOT) methodology in practice.