# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ResponsesRequest,
        )
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any


class AprielReasoningParser(ReasoningParser):
    """
    Reasoning parser for Apriel models.

    This parser treats all text before the "[BEGIN FINAL RESPONSE]" marker as
    thinking/reasoning content. There is no explicit start token.
    """

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content and starts the final response."""
        return "[BEGIN FINAL RESPONSE]"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                    "The model tokenizer must be passed to the ReasoningParser "
                    "constructor during construction."
                    )

        # Resolve the ID for the separator marker
        self.end_token_id = self.vocab.get(self.end_token)

        self.special_token_clean = ["<|end|>", "[END FINAL RESPONSE]", "</s>"]

        if self.end_token_id is None:
            # Attempt to find the ID by encoding if not in vocab directly
            encoded = self.tokenizer.encode(self.end_token, add_special_tokens=False)
            if len(encoded) == 1:
                self.end_token_id = encoded[0]

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        Reasoning has ended if the end_token_id is present in the sequence.
        """
        return self.end_token_id in input_ids

    def is_reasoning_end_streaming(
            self, input_ids: list[int], delta_ids: list[int]
            ) -> bool:
        """
        In streaming, reasoning ends when the end_token_id appears in the delta.
        """
        return self.end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content IDs that appear strictly after the end token.
        """
        if self.end_token_id not in input_ids:
            return []

        index = input_ids.index(self.end_token_id)
        return input_ids[index + 1:]

    def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
            ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message.
        Since there is no start token, everything before end_token is reasoning.
        """
        # Skip the end token itself to avoid it appearing in text
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.end_token_id:
            return None

        # If the end token has already been seen in previous tokens, everything is content
        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)

        # If the end token appears in the current delta
        if self.end_token_id in delta_token_ids:
            end_index = delta_text.find(self.end_token)
            if end_index != -1:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                        reasoning=reasoning if reasoning else None,
                        content=content if content else None
                        )

        # Default: end token not yet reached, so delta is reasoning
        return DeltaMessage(reasoning=delta_text)

    def extract_reasoning(
            self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
            ) -> tuple[str | None, str | None]:
        """
        Extract reasoning and content from the full model output.
        Everything before the marker is reasoning; everything after is content.
        """
        if self.end_token not in model_output:
            # Still in reasoning phase
            return model_output, None

        reasoning, _, content = model_output.partition(self.end_token)

        # Clean up any potential artifacts like <|end|> from the final content
        final_content = content
        for stop_signal in self.special_token_clean:
            if stop_signal in final_content:
                final_content = final_content.split(stop_signal)[0]

        return reasoning.strip(), final_content.strip() if final_content.strip() else None
