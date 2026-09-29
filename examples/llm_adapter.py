"""Skeleton LLM adapter. Copy, fill in `_call`, and point kgqa.toml at it:

    [llm]
    backend = "python:examples.llm_adapter:make"
    model = "your-model-id"        # any extra keys are passed to make(**kwargs)

The pipeline calls `complete(system, prompt)` for constrained generation (arm B, and arm C when no
template applies) and for the fallback path (arm A). Return token counts if your API reports them;
the eval report uses them for cost per question.
"""

from __future__ import annotations

from kgqa.llm import LLMResponse


class MyLLM:
    name = "my-llm"

    def __init__(self, model: str = "", temperature: float = 0.0, max_tokens: int = 800) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call(self, system: str, prompt: str) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens). Replace with your provider's SDK call."""
        raise NotImplementedError("Wire your LLM provider in examples/llm_adapter.py")

    def complete(self, system: str, prompt: str) -> LLMResponse:
        text, n_in, n_out = self._call(system, prompt)
        return LLMResponse(text, n_in, n_out)


def make(**kwargs) -> MyLLM:
    return MyLLM(**kwargs)
