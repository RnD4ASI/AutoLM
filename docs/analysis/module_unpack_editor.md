This document breaks down the sequence of interactions in the `editor.py` module.

```mermaid
sequenceDiagram
    participant Client
    participant TemplateAdopter
    participant AIUtility
    
    Client->>TemplateAdopter: get_prompt_transformation(prompt_dict, fix_template, components)
    
    %% Validate input parameters
    TemplateAdopter->>TemplateAdopter: Validate fix_template
    
    %% Create deep copy of input prompt
    TemplateAdopter->>TemplateAdopter: Create deep copy of prompt_dict
    
    %% Get available fix types from template
    TemplateAdopter->>AIUtility: get_meta_fix(fix_template)
    AIUtility-->>TemplateAdopter: available_fix_types
    
    %% Process each fix type in order
    loop For each fix_type in ["replace", "prefix", "postfix"]
        TemplateAdopter->>AIUtility: get_meta_fix(fix_template, fix_type)
        AIUtility-->>TemplateAdopter: fix_mods
        
        %% Apply modifications to each component
        loop For each component in components
            alt fix_type is "replace"
                TemplateAdopter->>TemplateAdopter: Replace component content
                TemplateAdopter->>TemplateAdopter: Add to replaced_components
            else fix_type is "prefix"
                TemplateAdopter->>TemplateAdopter: Prefix component content
            else fix_type is "postfix"
                TemplateAdopter->>TemplateAdopter: Postfix component content
            end
        end
    end
    
    TemplateAdopter-->>Client: Transformed prompt dictionary
    
    %% Error Handling
    alt Error occurs
        TemplateAdopter->>TemplateAdopter: Log error
        TemplateAdopter-->>Client: Raise RuntimeError
    end
```

## Key Components and Interactions

1. **TemplateAdopter**: Main class that handles prompt transformations
   - Applies reasoning patterns to prompt components
   - Manages the transformation process

2. **AIUtility**: Provides access to template configurations
   - Supplies fix templates and modifications
   - Handles metadata about available transformations

## Main Workflow

1. **Initialization**:
   - Validates the requested transformation template
   - Creates a deep copy of the input prompt dictionary

2. **Template Application**:
   - Retrieves available fix types for the template
   - Processes each fix type in order (replace → prefix → postfix)
   - Applies modifications to specified components

3. **Component Modification**:
   - **Replace**: Completely replaces component content
   - **Prefix**: Adds content before component
   - **Postfix**: Adds content after component

4. **Result Handling**:
   - Returns transformed prompt dictionary
   - Includes error handling and logging

## Error Handling
- Validates template types before processing
- Skips components that don't exist in the prompt
- Logs detailed error information
- Preserves original prompt on failure
