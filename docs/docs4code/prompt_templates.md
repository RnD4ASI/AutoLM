# Prompt Templates Documentation

This document details the prompt templates used in the APRA Analysis System, their categorization, and features.

## Overview

The prompt templates are organized into two main categories:
1. **Prompt Templates**: For manipulating and optimizing prompts
2. **Response Templates**: For handling and evaluating responses

## Prompt Templates

### 1. Manipulation Templates
Templates for basic prompt operations and modifications.

#### evaluate
- **Purpose**: Evaluate prompt quality and effectiveness
- **Features**:
  - Ambiguity detection
  - Clarity assessment
  - Effectiveness scoring (1-10)
  - Improvement suggestions
```json
{
    "prompt": "Your prompt here",
    "analysis": {
        "ambiguity_points": ["..."],
        "clarity_issues": ["..."],
        "effectiveness_score": 8,
        "suggestions": ["..."]
    }
}
```

#### rewrite
- **Purpose**: Rewrite prompts to address specific issues
- **Features**:
  - Intent preservation
  - Clarity improvement
  - Ambiguity removal
  - Specificity enhancement

#### edit
- **Purpose**: Edit prompts according to specific requirements
- **Features**:
  - Structured modifications
  - Change tracking
  - Rationale documentation

### 2. Evolution Templates
Templates for evolving and combining prompts.

#### crossover
- **Purpose**: Combine elements from multiple prompts
- **Features**:
  - Key concept preservation
  - Clarity maintenance
  - Effectiveness optimization
- **Example Use Case**: Merging domain-specific prompts with general templates

#### mutate
- **Purpose**: Evolve prompts to improve effectiveness
- **Features**:
  - Core meaning preservation
  - Clarity enhancement
  - Specificity addition

### 3. Reasoning Templates
Templates for different reasoning approaches.

#### cot (Chain-of-Thought)
- **Purpose**: Transform prompts into step-by-step reasoning format
- **Features**:
  - Step-by-step breakdown
  - Explicit thought process
  - Clear conclusion structure

#### program_synthesis
- **Purpose**: Convert tasks into detailed program specifications
- **Features**:
  - Problem specification
  - Input/output requirements
  - Constraints and assumptions
  - Solution approach
  - Pseudocode generation
  - Edge case consideration
  - Optimization suggestions

## Response Templates

### 1. Evaluation Templates
Templates for assessing response quality.

#### evaluate
- **Purpose**: Basic response quality assessment
- **Features**:
  - Accuracy checking
  - Relevance assessment
  - Completeness evaluation
  - Clarity analysis

#### reflect
- **Purpose**: Analyze response effectiveness
- **Features**:
  - Prompt alignment check
  - Improvement identification
  - Alternative approach suggestion

#### judge
- **Purpose**: Comprehensive response evaluation
- **Features**:
  - Factual accuracy scoring (1-10)
  - Completeness assessment
  - Relevance evaluation
  - Clarity & coherence analysis
- **Output Format**:
```json
{
    "factual_accuracy": {
        "score": 8,
        "strengths": ["..."],
        "issues": ["..."]
    },
    "completeness": {
        "score": 9,
        "strengths": ["..."],
        "issues": ["..."]
    },
    "overall_score": 8.5,
    "summary": "..."
}
```

### 2. Generation Templates
Templates for response creation and refinement.

#### synthesize
- **Purpose**: Generate comprehensive answers from multiple responses
- **Features**:
  - Key insight combination
  - Accuracy maintenance
  - Clarity assurance
  - Comprehensive coverage

#### refine
- **Purpose**: Improve responses based on feedback
- **Features**:
  - Feedback incorporation
  - Strength preservation
  - Accuracy maintenance
  - Clarity improvement
- **Output Format**:
```json
{
    "refined_response": "...",
    "improvements_made": ["..."],
    "retained_strengths": ["..."],
    "confidence_score": 0.85
}
```

### 3. Criteria Templates
Templates for evaluation criteria generation.

#### generate
- **Purpose**: Create custom evaluation criteria for specific tasks
- **Features**:
  - Task-specific requirements
  - Domain-specific attributes
  - Output characteristics
  - Pitfall identification
- **Output Format**:
```json
{
    "criteria": [
        {
            "name": "...",
            "description": "...",
            "scoring_guide": {
                "1-3": "poor performance...",
                "4-6": "adequate performance...",
                "7-8": "good performance...",
                "9-10": "excellent performance..."
            }
        }
    ],
    "weights": {
        "criterion_name": 0.4
    }
}
```

## Usage Examples

### Program Synthesis
```python
prompt_op = PromptOperator(generator, prompt_library)
result = prompt_op.transform_to_program(
    task="Create a function to find prime numbers",
    additional_context={"language": "python"}
)
```

### Response Evaluation
```python
response_op = ResponseOperator(generator)
evaluation = response_op.evaluate_response(
    response="The solution involves...",
    context={"task": "Explain quantum computing"}
)
```

### Template Combination
```python
prompt_op = PromptOperator(generator, prompt_library)
combined = prompt_op.crossover_prompts(
    prompt1="Explain in technical terms...",
    prompt2="Explain for beginners...",
    requirements={"audience": "intermediate"}
)
```

## Best Practices

1. **Template Selection**:
   - Use manipulation templates for basic prompt improvements
   - Use evolution templates for complex prompt development
   - Use reasoning templates for structured problem-solving

2. **Response Handling**:
   - Start with basic evaluation before detailed judging
   - Use synthesis for multiple source integration
   - Apply refinement based on concrete feedback

3. **Criteria Management**:
   - Generate custom criteria for domain-specific tasks
   - Use weighted scoring for balanced evaluation
   - Include both qualitative and quantitative metrics

4. **Integration Tips**:
   - Chain templates for complex workflows
   - Preserve context between template applications
   - Monitor and log template effectiveness
