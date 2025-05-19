from src.planner import TaskPlanner

# Initialize planner (now uses evaluator internally)
planner = TaskPlanner()

# Plan with specific evaluation model
flow_config, shortlisted_prompts = planner.plan_task_sequence(
    goal="Create a comprehensive security audit for our cloud infrastructure",
    max_prompts=5,
    similarity_threshold=0.3,
    embedding_model="Jina-embeddings-v3"  # Same model used in evaluator
)

# Analyze individual prompt relevance
goal = "Perform code review"
prompt_data = shortlisted_prompts["prompt_001"]
relevance = planner.evaluate_prompt_relevance(goal, prompt_data)