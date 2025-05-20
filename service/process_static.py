# Static Processes (such as database building)

# TODO 1: Knowledge Base Establishment
# - all files (pdf or md) are processed, certain files are built with vector db and graph db, and 
# - other files are built with vector db.
# - if knowledge base already exist, just load them rather than re-building
# >>> Referenced Code: dbbuilder.py
 
# TODO 2: Task Planning
# - Review the ultimate goal from runner
# - Review the task prompts json object (task_prompt_library.json)
# - Select the task prompts in relevance to the ultimate goal, the selection would be based on a config (pm.json)
# - Predict the sequence of the task prompts to be executed
# - Form a workflow config object (executor_config.json), such that it can be passed to Task Executor (see TODO 4)
# >>> Referenced Code: planner.py

# TODO 3: Initial Information Retrieval
# - Based on the task prompt selected in workflow config, create a schema for information that are commonly used by more than one prompt and that need to be retrieved.
# - Use the schema to retrieve information from the knowledge base.
# - Store the initial information retrieval results as Memory object (a separate vector db parquet file)
# >>> Referenced Code: retriever.py

# TODO 4: Task Execution
# - Based on the workflow config, execute the task prompts in sequence.
# - For each task prompt execution:
#   - Assess sufficiency of the information retrieved and stored in Memory against the task needs.
#   - If insufficient, select retrieval method and db type (based on mle.json), retrieve additional information from the knowledge base, and store in separate schema (i.e. a schema driven by task) 
#   - Based on the type of task, select model, hyperparameters, tools, topology, start with a fixed set of parameters in json object (mle.json)
# >>> Referenced Code: executor.py, topologist.py, retriever.py

# TODO 5: Task Evaluation
# - Based on the workflow config, execute the task prompts in sequence.
# - For each task prompt execution:
#   - Evaluate retrieval performance
#   - Evaluate response performance
# >>> Referenced Code: evaluator.py

# TODO 6: Output Collation
# - Based on the workflow config, collate the outputs of the task prompts.
# - Publish the outputs to a specified location.
# >>> Referenced Code: publisher.py

# TODO 7: Planing Optimisation
# - reinforcement learning based algorithm to optimise the workflow plan (pm.json)
# - evolution algorithm based algorithm to optimise the workflow plan (pm.json)
# >>> Referenced Code: HH_pm.py

# TODO 8: Selection Optimisation
# - reinforment learning based hyperheuristic algorithm to optimise the retrieval method and db type selection (mle.json)
# - evolution algorithm based hyperheuristic algorithm to optimise the retrieval method and db type selection (mle.json)
# - reinforcement learning based hyperheuristic algorithm to optimise the model, hyperparameters, tools, topology selection (mle.json)
# - evolution algorithm based hyperheuristic algorithm to optimise the model, hyperparameters, tools, topology selection (mle.json)
# >>> Referenced Code: HH_mle.py

# TODO 9: Code Base Optimisation
# - evolution algorithm based or self-reflection based self-evolution of code base
# >>> Referenced Code: HH_swe.py
