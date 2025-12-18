from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_json_tool_parser import AbstractJSONToolParser


class ApertusToolParser(AbstractJSONToolParser):
    """
    Tool call parser for Apriel models (e.g., Apriel 1.6).

    Extracts tool calls from the format:
    <tool_calls>[{"name": "func_name", "arguments": {"arg1": "value1"}}, ...]</tool_calls>

    Used when --enable-auto-tool-choice --tool-call-parser apriel are set.
    """

    def __init__(self, tokenizer: TokenizerLike) -> None:
        super().__init__(
                tokenizer,
                tool_calls_prefix="<|tools_prefix|>",
                tool_calls_suffix="<|tools_suffix|>",
                )

    def _extract_tool_call_data(self, tool_call_obj: dict) -> tuple[str | None, dict | None]:
        """
        Apertus format: {"function_name": {"arg": "val"}}
        The function name is the key, the arguments are the value.
        """
        if len(tool_call_obj) != 1:
            return None, None

        function_name = next(iter(tool_call_obj))
        arguments = tool_call_obj[function_name]
        return function_name, arguments
