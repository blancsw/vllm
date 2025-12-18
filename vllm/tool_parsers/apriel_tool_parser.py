from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_json_tool_parser import AbstractJSONToolParser


class AprielToolParser(AbstractJSONToolParser):
    """
    Tool call parser for Apriel models (e.g., Apriel 1.6).

    Extracts tool calls from the format:
    <tool_calls>[{"name": "func_name", "arguments": {"arg1": "value1"}}, ...]</tool_calls>

    Used when --enable-auto-tool-choice --tool-call-parser apriel are set.
    """

    def __init__(self, tokenizer: TokenizerLike) -> None:
        super().__init__(
                tokenizer,
                tool_calls_prefix="<tool_calls>",
                tool_calls_suffix="</tool_calls>",
                )

    def _extract_tool_call_data(self, tool_call_obj: dict) -> tuple[str | None, dict | None]:
        """
        Apriel format: {"name": "function_name", "arguments": {"arg": "val"}}
        """
        function_name = tool_call_obj.get("name")
        arguments = tool_call_obj.get("arguments")

        # In streaming contexts, "name" might be present while "arguments" is missing or None
        # In strictly complete JSON, both should be present.
        if not isinstance(function_name, str):
            return None, None

        return function_name, arguments