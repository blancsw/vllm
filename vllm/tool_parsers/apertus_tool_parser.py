from vllm.entrypoints.openai.protocol import ChatCompletionRequest
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

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because the tool_call tokens are
            # marked "special" in some models. Since they are skipped
            # prior to the call to the tool parser, it breaks tool calling.
            request.skip_special_tokens = False
        return request

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
