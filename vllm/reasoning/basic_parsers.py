# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Sequence, Tuple, List

from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.logger import init_logger
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

logger = init_logger(__name__)

class BaseThinkingReasoningParser(ReasoningParser):
    """Base class for reasoning parsers that use thinking tokens.

    This class provides common functionality for parsers that use start and end
    tokens to delimit reasoning content (
        e.g., <think>...</think>, <seed:think>...</seed:think>).

    Subclasses must implement the start and end tokens via abstract
    properties.
    If need strict_start_end_token to True reasoning is only extracted if the start token is explicitly.
    default: False
    """

    @property
    @abstractmethod
    def strict_start_end_token(self) -> bool:
        """Whether the parser requires both start and end tokens to be present.

        If True, reasoning is only extracted if the start token is explicitly
        found. If False, models that output reasoning at the start without a
        start token (implicit start) are supported.

        Returns:
            True if strict start/end tokens are required, False otherwise.
        """
        raise False

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token that starts reasoning content.

        Returns:
            The string representation of the start token.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token that ends reasoning content.

        Returns:
            The string representation of the end token.
        """
        raise NotImplementedError

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        if not self.start_token or not self.end_token:
            raise ValueError("start_token and end_token must be defined in subclasses")

        # Encode tokens to IDs. Supports multi-token IDs.
        # We use add_special_tokens=False to ensure we get the raw IDs for the string
        # without adding BOS/EOS tokens.
        start_ids = self.model_tokenizer.encode(
            self.start_token, add_special_tokens=False
        )
        end_ids = self.model_tokenizer.encode(
            self.end_token, add_special_tokens=False
        )

        self.start_token_ids: Tuple[int, ...] = tuple(start_ids)
        self.end_token_ids: Tuple[int, ...] = tuple(end_ids)

        if not self.start_token_ids or not self.end_token_ids:
            raise RuntimeError(
                f"{self.__class__.__name__} reasoning parser could not locate "
                "think start/end tokens in the tokenizer!"
            )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """Checks if the reasoning phase has ended based on input IDs.

        Scans the input IDs backwards to determine if the most recent delimiter
        encountered is the end token, indicating that reasoning has concluded.

        Args:
            input_ids: A list of token IDs representing the generated text so far.

        Returns:
            True if the reasoning phase has ended, False otherwise.
        """
        input_len = len(input_ids)
        start_len = len(self.start_token_ids)
        end_len = len(self.end_token_ids)

        # Iterate backwards to find the most recent delimiter
        i = input_len
        while i > 0:
            # Check for end token match ending at i
            if i >= end_len:
                if tuple(input_ids[i - end_len : i]) == self.end_token_ids:
                    return True

            # Check for start token match ending at i
            if i >= start_len:
                if tuple(input_ids[i - start_len : i]) == self.start_token_ids:
                    return False

            i -= 1

        return False

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        """Checks if the reasoning end token is present in the newly generated delta.

        Args:
            input_ids: The full list of generated token IDs (unused in this logic
                but kept for interface compatibility).
            delta_ids: The list of token IDs generated in the most recent step.

        Returns:
            True if the end token sequence appears in the delta, False otherwise.
        """
        return self._contains_sequence(delta_ids, self.end_token_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extracts the content token IDs that follow the reasoning end token.

        Args:
            input_ids: A list of token IDs containing both reasoning and content.

        Returns:
            A list of token IDs representing the content after the reasoning block.
            Returns an empty list if the end token is not found.
        """
        idx = self._find_sequence_index(input_ids, self.end_token_ids)
        if idx == -1:
            return []

        return input_ids[idx + len(self.end_token_ids) :]

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extracts reasoning content from a streaming delta.

        Determines which part of the delta belongs to reasoning and which part
        belongs to the final content, handling cases where tags are split across
        chunks. Respects the strict_start_end_token property.

        Args:
            previous_text: The text generated before the current step.
            current_text: The total text generated so far (previous + delta).
            delta_text: The text generated in the current step.
            previous_token_ids: Token IDs generated before the current step.
            current_token_ids: Total token IDs generated so far.
            delta_token_ids: Token IDs generated in the current step.

        Returns:
            A DeltaMessage containing extracted reasoning and/or content, or None
            if the delta contains only special tokens to be ignored.
        """
        # 1. Skip if the delta is exactly the start or end token sequence.
        # This prevents the tag itself from being emitted as content immediately.
        delta_tuple = tuple(delta_token_ids)
        if delta_tuple == self.start_token_ids or delta_tuple == self.end_token_ids:
            return None

        # 2. Determine reasoning state based on text.
        # Using text search is robust against tokenization boundaries (e.g. split tags).
        has_start = self.start_token in previous_text
        has_end_in_prev = self.end_token in previous_text

        if has_end_in_prev:
            # Reasoning phase already completed in previous steps.
            return DeltaMessage(content=delta_text)

        if has_start:
            # Reasoning started explicitly in previous chunks.
            end_index = delta_text.find(self.end_token)

            if end_index != -1:
                # Case: Start in prev, End in delta.
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning, content=content if content else None
                )
            else:
                # Case: Start in prev, No end yet.
                return DeltaMessage(reasoning=delta_text)

        else:
            # Start token not found in previous text.
            start_index = delta_text.find(self.start_token)

            if start_index != -1:
                # Found explicit start token in delta.
                end_index = delta_text.find(
                    self.end_token, start_index + len(self.start_token)
                )

                if end_index != -1:
                    # Case: Start in delta, End in delta.
                    reasoning = delta_text[
                        start_index + len(self.start_token) : end_index
                    ]
                    content = delta_text[end_index + len(self.end_token) :]
                    return DeltaMessage(
                        reasoning=reasoning, content=content if content else None
                    )
                else:
                    # Case: Start in delta, continues.
                    reasoning = delta_text[start_index + len(self.start_token) :]
                    return DeltaMessage(reasoning=reasoning)

            else:
                # No start token in previous text OR in delta.
                if self.strict_start_end_token:
                    # Strict mode: No start token means no reasoning.
                    return DeltaMessage(content=delta_text)
                else:
                    # Non-strict mode: Assume implicit reasoning start.
                    end_index = delta_text.find(self.end_token)

                    if end_index != -1:
                        # Case: Implicit Start, End in delta.
                        reasoning = delta_text[:end_index]
                        content = delta_text[end_index + len(self.end_token) :]
                        return DeltaMessage(
                            reasoning=reasoning, content=content if content else None
                        )
                    else:
                        # Case: Implicit Start, No end yet.
                        return DeltaMessage(reasoning=delta_text)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """Extracts reasoning and content from the full model output string.

        Args:
            model_output: The complete generated text from the model.
            request: The request object (unused in base implementation).

        Returns:
            A tuple containing:
                - The extracted reasoning text (or None if not found/empty).
                - The extracted content text (or None if not found/empty).
        """
        logger.warning(f"---- extract_reasoning\n{model_output}")
        # If strict requirements are enabled, reasoning MUST have a start token.
        if self.strict_start_end_token and self.start_token not in model_output:
            return None, model_output

        # String-based extraction is robust regardless of tokenization.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # Start token is detected, but not find the end token
        if self.end_token not in model_output and self.strict_start_end_token:
            return None, model_output
        # For models that may not generate start token,
        # assume the reasoning content is always at the start.
        elif self.end_token not in model_output:
            return model_output, None
        else:
            reasoning, _, content = model_output.partition(self.end_token)
            final_content = content or None
            return reasoning, final_content

    def _contains_sequence(
        self, haystack: Sequence[int], needle: Tuple[int, ...]
    ) -> bool:
        """Checks if a sequence of IDs exists within another sequence.

        Args:
            haystack: The sequence of IDs to search within.
            needle: The sequence of IDs to search for.

        Returns:
            True if the needle sequence is found in the haystack, False otherwise.
        """
        if not needle:
            return False
        n_len = len(needle)
        h_len = len(haystack)
        if n_len > h_len:
            return False

        # Optimization for single-token needle (common case)
        if n_len == 1:
            return needle[0] in haystack

        # Search for sequence
        for i in range(h_len - n_len + 1):
            if haystack[i] == needle[0]:
                if tuple(haystack[i : i + n_len]) == needle:
                    return True
        return False

    def _find_sequence_index(
        self, haystack: Sequence[int], needle: Tuple[int, ...]
    ) -> int:
        """Finds the starting index of a sequence of IDs within another sequence.

        Args:
            haystack: The sequence of IDs to search within.
            needle: The sequence of IDs to search for.

        Returns:
            The index where the needle sequence starts in the haystack, or -1 if not found.
        """
        if not needle:
            return -1
        n_len = len(needle)
        h_len = len(haystack)
        if n_len > h_len:
            return -1

        if n_len == 1:
            try:
                return haystack.index(needle[0])
            except ValueError:
                return -1

        for i in range(h_len - n_len + 1):
            if haystack[i] == needle[0]:
                if tuple(haystack[i : i + n_len]) == needle:
                    return i
        return -1