from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal, List, Set
from src.logging import get_logger
import copy
from src.utility import AIUtility

logger = get_logger(__name__)

class TemplateAdopter:
    """
    A class for adopting prompt templates to enhance prompts with different reasoning patterns.
    Uses the ppfix structure from config/prompt_templates.json to apply modifications to prompt components.
    
    Components are parts of the prompt like:
    - task: The main task or question
    - instruction: How to approach the task
    - response_format: Expected format for the response
    
    Fix types are ways to modify these components:
    - prefix: Add text before the component
    - postfix: Add text after the component
    - replace: Replace the component entirely
    """
    
    def __init__(self):
        """Initialize the TemplateAdopter with AIUtility for accessing templates."""
        self.aiutility = AIUtility()
        # Ensure templates are loaded
        #self.aiutility._load_prompts()
        
        # Valid reasoning templates
        self.valid_templates = ["chain_of_thought", "tree_of_thought", "program_synthesis", "deep_thought"]
        
        # The order in which fix types should be applied (replace should always come first)
        self.fix_type_order = ["replace", "prefix", "postfix"]
    
    def get_prompt_transformation(self, 
        prompt_dict: Dict[str, Any], 
        fix_template: str,
        components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Transformed a prompt dictionary with a specified reasoning behavior
        while preserving the original format and structure.
        
        Args:
            prompt_dict (dict): Original prompt dictionary with components like
                            'task', 'instruction', 'response_format', etc.
            fix_template (str): The type of reasoning to add. Options:
                                "chain_of_thought" - Chain of Thought
                                "tree_of_thought" - Tree of Thoughts
                                "program_synthesis" - Program Synthesis
                                "deep_thought" - Deep Thinking
            components (list, optional): Specific components to modify. If None, all components will be considered.
        
        Returns:
            dict: Transformed prompt dictionary with the specified reasoning behavior
        """
        # Validate the fix_template
        if fix_template not in self.valid_templates:
            raise ValueError(f"fix_template must be one of: {self.valid_templates}")
        
        # Create a deep copy to avoid modifying the original
        transformed_prompt = copy.deepcopy(prompt_dict)
        
        # If no components specified, use all components in the prompt
        if components is None:
            components = list(prompt_dict.keys())
        
        try:
            # Get available fix types from the template configuration
            available_fix_types = self.aiutility.get_meta_fix(fix_template=fix_template)
            if not available_fix_types:
                logger.warning(f"No fix types found for template {fix_template}")
                return transformed_prompt
            
            # Track which components have been replaced to avoid applying prefix/postfix to them
            replaced_components = set()
            
            # Process fix types in the predefined order (replace, prefix, postfix)
            for fix_type in self.fix_type_order:
                # Skip if this fix type is not available in the template
                if fix_type not in available_fix_types:
                    logger.debug(f"Fix type '{fix_type}' not available for template '{fix_template}'")
                    continue
                
                # Get the modifications for this fix type
                fix_mods = self.aiutility.get_meta_fix(fix_template, fix_type)
                
                # Apply modifications to each specified component
                for component in components:
                    # Skip if component doesn't exist in prompt
                    if component not in transformed_prompt:
                        continue
                        
                    # Skip if component has been replaced and we're trying to apply prefix/postfix
                    if component in replaced_components and fix_type in ["prefix", "postfix"]:
                        continue
                    
                    # Get the modification for this component
                    modification = fix_mods.get(component)
                    
                    # Skip if no modification exists or if it's None
                    if modification is None or modification == "":
                        logger.debug(f"No modification found for component '{component}' with fix type '{fix_type}'")
                        continue
                    
                    # Apply the appropriate modification based on fix_type
                    if fix_type == "prefix":
                        transformed_prompt[component] = modification + transformed_prompt[component]
                        logger.debug(f"Added prefix to component '{component}'")
                    elif fix_type == "postfix":
                        transformed_prompt[component] = transformed_prompt[component] + modification
                        logger.debug(f"Added postfix to component '{component}'")
                    elif fix_type == "replace":
                        transformed_prompt[component] = modification
                        replaced_components.add(component)
                        logger.debug(f"Replaced component '{component}' with template content")
            
            logger.info(f"Successfully transformed prompt with {fix_template} reasoning")
            return transformed_prompt
            
        except Exception as e:
            logger.error(f"Error transforming prompt with {fix_template}: {str(e)}")
            raise RuntimeError(f"Failed to transform prompt: {str(e)}") from e
