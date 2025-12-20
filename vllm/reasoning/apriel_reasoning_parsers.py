# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    pass
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any

logger = init_logger(__name__)


class AprielReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Apriel models.

    This parser treats all text before the "[BEGIN FINAL RESPONSE]" marker as
    thinking/reasoning content. There is no explicit start token.
    """

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content and starts the final response."""
        return "[BEGIN FINAL RESPONSE]"

    @property
    def start_token(self) -> str:
        return "Here are my reasoning steps:\n"

    @property
    def strict_start_end_token(self):
        return False
