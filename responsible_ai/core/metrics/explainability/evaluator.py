"""
Explainability metric evaluator.
"""

from typing import Dict, Any, List, Optional
from core.evaluators.base_evaluator import BaseEvaluator
from core.llm.llm_client import LLMClient
from utils.helpers import parse_json_from_string, preserve_item_context
from utils.errors import EvaluationError


class ExplainabilityEvaluator(BaseEvaluator):
    """
    Evaluator for generating explanations of AI model responses.
    """

    def __init__(self):
        """Initialize the explainability evaluator."""
        super().__init__("explainability")
        self.llm_client = LLMClient()

    def _format_prompt(self, prompt: str, response: str, **kwargs) -> str:
        """
        Format the evaluation prompt with provided context.

        Args:
            prompt: The original prompt (user query)
            response: The model's response to evaluate
            **kwargs: Additional context parameters

        Returns:
            Formatted prompt string
        """
        # Get optional context parameters
        system_instructions = kwargs.get("system_instructions") or "None"
        conversation_history = kwargs.get("conversation_history") or "None"
        retrieved_contexts = kwargs.get("retrieved_contexts") or "None"

        # Format the prompt template with all parameters

        evaluation_prompt = self.prompt_template.replace("{{system_instructions}}", system_instructions)
        evaluation_prompt = evaluation_prompt.replace("{{conversation_history}}", conversation_history)
        evaluation_prompt = evaluation_prompt.replace("{{user_query}}", prompt)
        evaluation_prompt = evaluation_prompt.replace("{{retrieved_contexts}}", retrieved_contexts)
        evaluation_prompt = evaluation_prompt.replace("{{llm_response}}", response)

        return evaluation_prompt

    def evaluate(self, prompt: str, response: str, api_request_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate an explanation for a model's response.
        Unlike other metrics, this doesn't produce a score but a detailed explanation.

        Args:
            prompt: The original prompt
            response: The model's response to evaluate
            api_request_id: Optional API request ID for tracing
            **kwargs: Additional parameters for evaluation including optional context

        Returns:
            Explanation as a dictionary with various explanation components
        """
        self.logger.info("Generating explanation for model response")

        try:
            # Format the evaluation prompt with context
            evaluation_prompt = self._format_prompt(prompt, response, **kwargs)

            # Get LLM explanation - pass api_request_id
            llm_response, usage_data = self.llm_client.get_completion(evaluation_prompt, api_request_id=api_request_id)

            # Parse the JSON response
            explanation = parse_json_from_string(llm_response)

            # Add metadata to the explanation
            result = {"prompt": prompt, "response": response, "explanation": explanation}

            self.logger.info("Explanation generation complete")
            return result, usage_data

        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            raise EvaluationError(f"Explanation generation failed: {str(e)}")

    def batch_evaluate(self, data: List[Dict[str, Any]], api_request_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple prompt-response pairs.

        Args:
            data: List of dictionaries containing prompts and responses
            api_request_id: Optional API request ID for tracing
            **kwargs: Additional parameters for evaluation

        Returns:
            List of explanation results
        """
        self.logger.info(f"Generating explanations for {len(data)} items")

        results = []
        for i, item in enumerate(data):
            try:
                prompt = item.get("prompt", "")
                response = item.get("response", "")

                if not prompt or not response:
                    self.logger.warning(f"Skipping item {i}: missing prompt or response")
                    continue

                # Extract additional context from item if available
                context_kwargs = {
                    "system_instructions": item.get("system_instructions"),
                    "conversation_history": item.get("conversation_history"),
                    "retrieved_contexts": item.get("retrieved_contexts"),
                }

                result = self.evaluate(prompt, response, api_request_id=api_request_id, **context_kwargs)

                result = preserve_item_context(item, result)

                results.append(result)
                self.logger.info(f"Processed explanation generation for item {i+1}/{len(data)}")
            except Exception as e:
                self.logger.error(f"Error processing item {i}: {str(e)}")
                results.append({"error": str(e), "id": item.get("id", f"item-{i}")})

        return results