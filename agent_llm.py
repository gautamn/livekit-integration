# Copyright 2023 LiveKit, Inc.
#

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

import openai, json, requests
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import ToolChoice, utils as llm_utils
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import FunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,
)
from openai.types.chat.chat_completion_chunk import Choice

from livekit.plugins.openai.models import (
    CerebrasChatModels,
    ChatModels,
    DeepSeekChatModels,
    OctoChatModels,
    PerplexityChatModels,
    TelnyxChatModels,
    TogetherChatModels,
    XAIChatModels,
)
from livekit.plugins.openai.utils import AsyncAzureADTokenProvider, to_chat_ctx, to_fnc_ctx

from agent_completion_steam import AgentStreamWrapper
from agent_api import AgentAPIClient

import aiohttp
import asyncio

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))

openai_api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class _LLMOptions:
    model: str | ChatModels
    user: NotGivenOr[str]
    temperature: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]
    store: NotGivenOr[bool]
    metadata: NotGivenOr[dict[str, str]]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ChatModels = "gpt-4o",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        store: NotGivenOr[bool] = NOT_GIVEN,
        metadata: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of OpenAI LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """
        super().__init__()
        self._opts = _LLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
        )
        self._client = client or openai.AsyncClient(
            api_key=api_key if is_given(api_key) else None,
            base_url=base_url if is_given(base_url) else None,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=timeout
                if timeout
                else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    @staticmethod
    def with_azure(
        *,
        model: str | ChatModels = "gpt-4o",
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> LLM:
        """
        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `AZURE_OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
        - `api_version` from `OPENAI_API_VERSION`
        - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`
        """  # noqa: E501

        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout
            if timeout
            else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        )  # type: ignore

        return LLM(
            model=model,
            client=azure_client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    @staticmethod
    def with_cerebras(
        *,
        model: str | CerebrasChatModels = "llama3.1-8b",
        api_key: str | None = None,
        base_url: str = "https://api.cerebras.ai/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ) -> LLM:
        """
        Create a new instance of Cerebras LLM.

        ``api_key`` must be set to your Cerebras API key, either using the argument or by setting
        the ``CEREBRAS_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if api_key is None:
            raise ValueError(
                "Cerebras API key is required, either as argument or set CEREBAAS_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    @staticmethod
    def with_fireworks(
        *,
        model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
        api_key: str | None = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
    ) -> LLM:
        """
        Create a new instance of Fireworks LLM.

        ``api_key`` must be set to your Fireworks API key, either using the argument or by setting
        the ``FIREWORKS_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if api_key is None:
            raise ValueError(
                "Fireworks API key is required, either as argument or set FIREWORKS_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    @staticmethod
    def with_x_ai(
        *,
        model: str | XAIChatModels = "grok-2-public",
        api_key: str | None = None,
        base_url: str = "https://api.x.ai/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
    ):
        """
        Create a new instance of XAI LLM.

        ``api_key`` must be set to your XAI API key, either using the argument or by setting
        the ``XAI_API_KEY`` environmental variable.
        """
        api_key = api_key or os.environ.get("XAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "XAI API key is required, either as argument or set XAI_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    @staticmethod
    def with_deepseek(
        *,
        model: str | DeepSeekChatModels = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
    ) -> LLM:
        """
        Create a new instance of DeepSeek LLM.

        ``api_key`` must be set to your DeepSeek API key, either using the argument or by setting
        the ``DEEPSEEK_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError(
                "DeepSeek API key is required, either as argument or set DEEPSEEK_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    @staticmethod
    def with_octo(
        *,
        model: str | OctoChatModels = "llama-2-13b-chat",
        api_key: str | None = None,
        base_url: str = "https://text.octoai.run/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
    ) -> LLM:
        """
        Create a new instance of OctoAI LLM.

        ``api_key`` must be set to your OctoAI API key, either using the argument or by setting
        the ``OCTOAI_TOKEN`` environmental variable.
        """

        api_key = api_key or os.environ.get("OCTOAI_TOKEN")
        if api_key is None:
            raise ValueError(
                "OctoAI API key is required, either as argument or set OCTOAI_TOKEN environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    @staticmethod
    def with_ollama(
        *,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434/v1",
        client: openai.AsyncClient | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
    ) -> LLM:
        """
        Create a new instance of Ollama LLM.
        """

        return LLM(
            model=model,
            api_key="ollama",
            base_url=base_url,
            client=client,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    @staticmethod
    def with_perplexity(
        *,
        model: str | PerplexityChatModels = "llama-3.1-sonar-small-128k-chat",
        api_key: str | None = None,
        base_url: str = "https://api.perplexity.ai",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
    ) -> LLM:
        """
        Create a new instance of PerplexityAI LLM.

        ``api_key`` must be set to your TogetherAI API key, either using the argument or by setting
        the ``PERPLEXITY_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if api_key is None:
            raise ValueError(
                "Perplexity AI API key is required, either as argument or set PERPLEXITY_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    @staticmethod
    def with_together(
        *,
        model: str | TogetherChatModels = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key: str | None = None,
        base_url: str = "https://api.together.xyz/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
    ) -> LLM:
        """
        Create a new instance of TogetherAI LLM.

        ``api_key`` must be set to your TogetherAI API key, either using the argument or by setting
        the ``TOGETHER_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError(
                "Together AI API key is required, either as argument or set TOGETHER_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    @staticmethod
    def with_telnyx(
        *,
        model: str | TelnyxChatModels = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        api_key: str | None = None,
        base_url: str = "https://api.telnyx.com/v2/ai",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: ToolChoice = "auto",
    ) -> LLM:
        """
        Create a new instance of Telnyx LLM.

        ``api_key`` must be set to your Telnyx API key, either using the argument or by setting
        the ``TELNYX_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("TELNYX_API_KEY")
        if api_key is None:
            raise ValueError(
                "Telnyx AI API key is required, either as argument or set TELNYX_API_KEY environmental variable"  # noqa: E501
            )

        return LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

    async def create_my_stream(self, messages):
        """Create a stream using OpenAI API for streaming completions"""
        import openai
        import os
        from agent_completion_steam import AgentStreamWrapper

        # Use the OpenAI client directly
        # You can customize these settings as needed
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Create an OpenAI client
        client = openai.AsyncClient(api_key=openai_api_key)
        
        try:
            # Make the API call to OpenAI with streaming enabled
            # This matches the behavior of the original OpenAI streaming call
            stream = await client.chat.completions.create(
                model="gpt-4",  # You can change this to your preferred model
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}
            )
            
            # Return the stream directly - no need for AgentStreamWrapper
            # since OpenAI's stream already implements the async iterator protocol
            return stream
            
        except Exception as e:
            print(f"OpenAI API Error: {str(e)}")
            # Create a fallback response
            raise e
    
    async def simulate_agent_stream(self, messages=None, conversation_id=None):
        """Simulate a text stream that connects to the third-party agent API.
        
        This function creates an OpenAI-compatible stream that connects to the agent API
        and streams back responses in a format compatible with OpenAI's streaming platform.
        
        Args:
            messages: Optional messages from the user. If provided, the last user message
                     will be sent to the agent as the query.
            conversation_id: Optional ID for continuing a conversation with the agent.
                     
        Returns:
            An asynchronous stream compatible with OpenAI's streaming format.
        """
        # Set this flag to true to enable a fallback response if the agent API doesn't return any text
        USE_FALLBACK_RESPONSE = False
        # Import OpenAI types to create proper objects
        import openai
        import os
        import time
        import uuid
        
        # Extract the query from messages if provided
        query = "Hello, how are you?"
        
        # Handle different message formats
        if messages:
            # Check if messages is a list of dictionaries (standard format)
            if isinstance(messages, list) and len(messages) > 0:
                # Find the last user message
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        query = msg.get('content', query)
                        break
            # Check if messages is a list of OpenAI message objects
            elif hasattr(messages, '__iter__'):
                # Convert to list to be able to iterate in reverse
                messages_list = list(messages)
                for msg in reversed(messages_list):
                    # Check if it's an OpenAI message object with role and content attributes
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        if msg.role == 'user':
                            query = msg.content
                            break
                    # Check if it's a dict-like object
                    elif hasattr(msg, 'get'):
                        if msg.get('role') == 'user':
                            query = msg.get('content', query)
                            break
        
        print(f"[Agent Stream] Extracted query: {query}")
        
        # Create a custom stream that returns proper OpenAI-compatible objects
        try:
            # First, let's try to import the necessary OpenAI types
            try:
                from openai.types.chat import ChatCompletionChunk
                from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
                
                # Create a class that returns proper OpenAI objects by wrapping the agent API
                class AgentAPIStream:
                    def __init__(self, query, conversation_id=None):
                        self.query = query
                        self.conversation_id = conversation_id or str(uuid.uuid4())
                        self.agent_client = AgentAPIClient()
                        self.agent_stream = None
                        self.finished = False
                        self.stream_id = f"agent-{uuid.uuid4()}"
                        self.buffer = ""  # Buffer for accumulating characters
                        self.word_buffer = ""  # Buffer for accumulating words
                        self.received_text = False  # Flag to track if we've received any text
                        self.fallback_response = "I'm sorry, I couldn't process your request at this time. Please try again later."  # Fallback response
                        print(f"[Agent Stream] Initializing with query: {query}, conversation_id: {self.conversation_id}")
                    
                    def __aiter__(self):
                        return self
                    
                    async def __anext__(self):
                        print("[AgentAPIStream.__anext__] Method called")
                        # Initialize the agent stream if not already done
                        if self.agent_stream is None:
                            print("[AgentAPIStream.__anext__] Initializing agent_stream")
                            self.agent_stream = self.agent_client.call_agent(
                                query=self.query,
                                conversation_id=self.conversation_id
                            )
                            print("[AgentAPIStream.__anext__] agent_stream initialized")
                        
                        # If we've finished, stop iteration
                        if self.finished:
                            print("[AgentAPIStream.__anext__] Stream finished, raising StopAsyncIteration")
                            raise StopAsyncIteration
                        
                        try:
                            print("[AgentAPIStream.__anext__] Awaiting next chunk from agent_stream")
                            # Get the next chunk from the agent
                            agent_chunk = await anext(self.agent_stream)
                            print(f"[AgentAPIStream.__anext__] Received agent chunk: {agent_chunk}")
                            
                            # Extract text from the agent's response
                            #text = agent_chunk.get("answer", "")
                            text = agent_chunk
                            print(f"[AgentAPIStream.__anext__] Extracted text: '{text}'")
                            
                            # If we got text, add it to our buffer and immediately send it to TTS
                            # This ensures the TTS layer receives content as soon as possible
                            if text:
                                print(f"[AgentAPIStream.__anext__] Sending text directly to TTS: '{text}'")
                                # Mark that we've received text from the agent API
                                self.received_text = True
                                # Create a chunk with the text and return it immediately
                                return self._create_chunk(text)
                            
                            # If no text, add empty string to buffer
                            self.buffer += text
                            
                            # If we got an empty chunk and we're at the end of the stream
                            if not text:
                                print("[AgentAPIStream.__anext__] Empty text received")
                                # If we have any remaining text in the buffer, send it
                                if self.word_buffer:
                                    text_to_send = self.word_buffer
                                    self.word_buffer = ""
                                    self.finished = True
                                    return self._create_chunk(text_to_send)
                                elif self.buffer:
                                    text_to_send = self.buffer
                                    self.buffer = ""
                                    self.finished = True
                                    return self._create_chunk(text_to_send)
                                else:
                                    # No more text, send the final stop chunk
                                    final_chunk = ChatCompletionChunk(
                                        id=f"chatcmpl-{uuid.uuid4()}",  # Use chatcmpl- prefix to match OpenAI format
                                        choices=[Choice(
                                            index=0,
                                            delta=ChoiceDelta(tool_calls=[]),  # Include empty tool_calls
                                            finish_reason="stop"
                                        )],
                                        model="gpt-4",  # Use a model name that the TTS system recognizes
                                        object="chat.completion.chunk",
                                        created=int(time.time()),
                                        usage=None
                                    )
                                    print(f"[Agent] Created final chunk: id='{final_chunk.id}' with finish_reason='stop'")
                                    return final_chunk
                            
                            # Check if we have complete words (space or punctuation)
                            words = []
                            i = 0
                            while i < len(self.buffer):
                                if self.buffer[i] in " ,.!?;:\n":
                                    # We found a word boundary
                                    if self.word_buffer:
                                        # Add the accumulated word
                                        words.append(self.word_buffer)
                                        self.word_buffer = ""
                                    # Add the space or punctuation as its own word
                                    words.append(self.buffer[i])
                                else:
                                    # Add character to the current word
                                    self.word_buffer += self.buffer[i]
                                i += 1
                            
                            # Update the buffer to contain only the partial word
                            self.buffer = self.word_buffer
                            self.word_buffer = ""
                            
                            # If we have words to send, join them and return
                            if words:
                                text_to_send = "".join(words)
                                return self._create_chunk(text_to_send)
                            
                            # If we don't have any words to send yet, continue to the next chunk
                            return await self.__anext__()
                                    
                        except StopAsyncIteration:
                            # If the agent stream is done, send any remaining text
                            self.finished = True
                            
                            # If we haven't received any text from the agent API, use the fallback response
                            if not self.received_text and USE_FALLBACK_RESPONSE:
                                print(f"[AgentAPIStream.__anext__] No text received from agent API, using fallback response: {self.fallback_response}")
                                return self._create_chunk(self.fallback_response)
                            
                            if self.word_buffer:
                                text_to_send = self.word_buffer
                                self.word_buffer = ""
                                return self._create_chunk(text_to_send)
                            elif self.buffer:
                                text_to_send = self.buffer
                                self.buffer = ""
                                return self._create_chunk(text_to_send)
                            else:
                                # No more text, send the final stop chunk
                                final_chunk = ChatCompletionChunk(
                                    id=f"chatcmpl-{uuid.uuid4()}",  # Use chatcmpl- prefix to match OpenAI format
                                    choices=[Choice(
                                        index=0,
                                        delta=ChoiceDelta(tool_calls=[]),  # Include empty tool_calls
                                        finish_reason="stop"
                                    )],
                                    model="gpt-4",  # Use a model name that the TTS system recognizes
                                    object="chat.completion.chunk",
                                    created=int(time.time()),
                                    usage=None
                                )
                                print(f"[Agent] Created final chunk: id='{final_chunk.id}' with finish_reason='stop'")
                                return final_chunk
                    
                    def _create_chunk(self, text):
                        """Helper method to create a ChatCompletionChunk with the given text"""
                        # Print the text being sent to TTS for debugging
                        print(f"[TTS Input] Sending text to TTS: {text}")
                        
                        # Create a chunk with the proper format for TTS processing
                        # This format matches what OpenAI returns: id='chatcmpl-XXX' delta=ChoiceDelta(role='assistant', content='text', tool_calls=[])
                        chunk = ChatCompletionChunk(
                            id=f"chatcmpl-{uuid.uuid4()}",  # Use chatcmpl- prefix to match OpenAI format
                            choices=[Choice(
                                index=0,
                                delta=ChoiceDelta(content=text, role="assistant", tool_calls=[]),
                                finish_reason=None
                            )],
                            model="gpt-4",  # Use a model name that the TTS system recognizes
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            usage=None
                        )
                        
                        # Log the chunk in the expected format
                        print(f"[Agent] Created chunk: id='{chunk.id}' delta=ChoiceDelta(role='assistant', content='{text}', tool_calls=[])")
                        
                        return chunk
                    
                    async def __aenter__(self):
                        return self
                        
                    async def __aexit__(self, exc_type, exc_val, exc_tb):
                        pass
                
                # Return the agent API stream
                return AgentAPIStream(query, conversation_id)
                
            except ImportError as ie:
                print(f"Import error: {str(ie)}. Falling back to create_my_stream.")
                # If we can't import the OpenAI types, fall back to create_my_stream
                return await self.create_my_stream(messages)
            
        except Exception as e:
            print(f"Error in simulate_agent_stream: {str(e)}")
            # If there's an error, fall back to create_my_stream
            return await self.create_my_stream(messages)
    
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:

        print("#####################################chat()#####################################################")
        extra = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.metadata):
            extra["metadata"] = self._opts.metadata

        if is_given(self._opts.user):
            extra["user"] = self._opts.user

        parallel_tool_calls = (
            parallel_tool_calls if is_given(parallel_tool_calls) else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice  # type: ignore
        if is_given(tool_choice):
            oai_tool_choice: ChatCompletionToolChoiceOptionParam
            if isinstance(tool_choice, dict):
                oai_tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice["function"]["name"]},
                }
                extra["tool_choice"] = oai_tool_choice
            elif tool_choice in ("auto", "required", "none"):
                oai_tool_choice = tool_choice
                extra["tool_choice"] = oai_tool_choice

        if is_given(response_format):
            extra["response_format"] = llm_utils.to_openai_response_format(response_format)

        return LLMStream(
            self,
            model=self._opts.model,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        model: str | ChatModels,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._client = client
        self._llm = llm
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        # current function call that we're waiting for full completion (args are streamed)
        # (defined inside the _run method to make sure the state is reset for each run/attempt)
        self._oai_stream: openai.AsyncStream[ChatCompletionChunk] | None = None
        self._tool_call_id: str | None = None
        self._fnc_name: str | None = None
        self._fnc_raw_arguments: str | None = None
        self._tool_index: int | None = None
        self._agent_stream: AgentStreamWrapper | None = None
        retryable = True

        try:
            print("#####################################_run()#####################################################")
            chat_ctx = to_chat_ctx(self._chat_ctx, id(self._llm))
            fnc_ctx = to_fnc_ctx(self._tools) if self._tools else openai.NOT_GIVEN
            
            # self._oai_stream = stream = await self._client.chat.completions.create(
            #     messages=chat_ctx,
            #     tools=fnc_ctx,
            #     model=self._model,
            #     stream_options={"include_usage": True},
            #     stream=True,
            #     **self._extra_kwargs,
            # )

            # Use the custom chatbot stream implementation
            # self._agent_stream = stream = await self.create_my_stream(
            #     messages=chat_ctx
            # )
            # self._agent_stream = stream = await self.create_chatbot_stream(
            #     messages=chat_ctx
            # )

            # self._agent_stream = stream = await self.simulate_text_stream(
            #     messages=chat_ctx
            # )

            # Use the agent API integration exclusively since that's what we need
            
            use_agent_api = os.getenv("USE_AGENT_API", "false").lower() == "true"
            stream = None
            print(f"\n[Agent] Using Agent API integration: {use_agent_api}")    
            if use_agent_api:
                print("Using Agent API integration")
                # Get the agent stream from the LLM instance
                print("Before calling simulate_agent_stream")
                stream = await self._llm.simulate_agent_stream(
                    messages=chat_ctx
                )
                print(f"After calling simulate_agent_stream, got stream of type: {type(stream)}")
                self._agent_stream = stream
            else:
                print("Using OpenAI API integration")
                stream = await self._client.chat.completions.create(
                    messages=chat_ctx,
                    tools=fnc_ctx,
                    model=self._model,
                    stream_options={"include_usage": True},
                    stream=True,
                    **self._extra_kwargs,
                )
                print(f"After calling chat.completions.create, got stream of type: {type(stream)}")

            print(f"Setting self._oai_stream to stream of type: {type(stream)}")
            self._oai_stream = stream  

            async with stream:
                async for chunk in stream:
                    for choice in chunk.choices:
                        chat_chunk = self._parse_choice(chunk.id, choice)
                        print(f"\n[Agent] Received chunk: {chat_chunk}")
                        if chat_chunk is not None:
                            retryable = False
                            self._event_ch.send_nowait(chat_chunk)

                    if chunk.usage is not None:
                        retryable = False
                        tokens_details = chunk.usage.prompt_tokens_details
                        cached_tokens = tokens_details.cached_tokens if tokens_details else 0
                        chunk = llm.ChatChunk(
                            id=chunk.id,
                            usage=llm.CompletionUsage(
                                completion_tokens=chunk.usage.completion_tokens,
                                prompt_tokens=chunk.usage.prompt_tokens,
                                prompt_cached_tokens=cached_tokens or 0,
                                total_tokens=chunk.usage.total_tokens,
                            ),
                        )
                        self._event_ch.send_nowait(chunk)

        except openai.APITimeoutError:
            raise APITimeoutError(retryable=retryable) from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            ) from None
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_choice(self, id: str, choice: Choice) -> llm.ChatChunk | None:
        delta = choice.delta
        print(f"\n[Agent] _parse_choice called with id: {id}, delta: {delta}")

        if delta is None:
            print("[Agent] Delta is None, returning None")
            return None
        
        # Create a new ID with the chatcmpl- prefix to match OpenAI's format
        # This ensures the TTS layer processes the chunk correctly
        new_id = id
        if not id.startswith("chatcmpl-"):
            import uuid
            new_id = f"chatcmpl-{uuid.uuid4()}"
            print(f"[Agent] Changing ID from {id} to {new_id} to match OpenAI format")
        
        # Ensure the delta has the correct structure
        if delta.content is None:
            print("[Agent] Adding dummy content to response with None content")
            delta.content = " "  # Single space to trigger TTS
        
        # Ensure tool_calls is present (even if empty)
        if not hasattr(delta, 'tool_calls') or delta.tool_calls is None:
            print("[Agent] Adding empty tool_calls to delta")
            delta.tool_calls = []
        
        # Ensure role is set to assistant if not already set
        if delta.role is None:
            print("[Agent] Setting role to assistant")
            delta.role = "assistant"

        # print("#####################################_parse_choice()#####################################################")
        if delta.tool_calls:
            for tool in delta.tool_calls:
                if not tool.function:
                    continue

                call_chunk = None
                if self._tool_call_id and tool.id and tool.index != self._tool_index:
                    call_chunk = llm.ChatChunk(
                        id=id,
                        delta=llm.ChoiceDelta(
                            role="assistant",
                            content=delta.content,
                            tool_calls=[
                                llm.FunctionToolCall(
                                    arguments=self._fnc_raw_arguments or "",
                                    name=self._fnc_name or "",
                                    call_id=self._tool_call_id or "",
                                )
                            ],
                        ),
                    )
                    self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None

                if tool.function.name:
                    self._tool_index = tool.index
                    self._tool_call_id = tool.id
                    self._fnc_name = tool.function.name
                    self._fnc_raw_arguments = tool.function.arguments or ""
                elif tool.function.arguments:
                    self._fnc_raw_arguments += tool.function.arguments  # type: ignore

                if call_chunk is not None:
                    return call_chunk

        if choice.finish_reason in ("tool_calls", "stop") and self._tool_call_id:
            call_chunk = llm.ChatChunk(
                id=id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content=delta.content,
                    tool_calls=[
                        llm.FunctionToolCall(
                            arguments=self._fnc_raw_arguments or "",
                            name=self._fnc_name or "",
                            call_id=self._tool_call_id or "",
                        )
                    ],
                ),
            )
            self._tool_call_id = self._fnc_name = self._fnc_raw_arguments = None
            return call_chunk

        return llm.ChatChunk(
            id=new_id,  # Use the new ID with chatcmpl- prefix
            delta=llm.ChoiceDelta(content=delta.content, role="assistant"),
        )

   

    async def create_my_stream(self, messages):
        """Create a stream using OpenAI API for streaming completions"""
        import openai
        import os
        from agent_completion_steam import AgentStreamWrapper

        # Use the OpenAI client directly
        # You can customize these settings as needed
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Create an OpenAI client
        client = openai.AsyncClient(api_key=openai_api_key)
        
        try:
            # Make the API call to OpenAI with streaming enabled
            # This matches the behavior of the original OpenAI streaming call
            stream = await client.chat.completions.create(
                model="gpt-4",  # You can change this to your preferred model
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}
            )
            
            # Return the stream directly - no need for AgentStreamWrapper
            # since OpenAI's stream already implements the async iterator protocol
            return stream
            
        except Exception as e:
            print(f"OpenAI API Error: {str(e)}")
            # Create a fallback response
            # This simulates an OpenAI streaming response with an error message
        # from openai.types.chat import ChatCompletionChunk, ChatCompletionChunkChoice
        # from openai.types.chat.chat_completion_chunk import ChunkChoice, ChoiceDelta
            
            # Create a mock OpenAI stream
            # class MockOpenAIStream:
            #     def __init__(self, error_message):
            #         self.error_message = error_message
            #         self._done = False
                
            #     async def __aiter__(self):
            #         return self
                
            #     async def __anext__(self):
            #         if self._done:
            #             raise StopAsyncIteration
                    
            #         self._done = True
                    
            #         # Create a chunk that mimics OpenAI's format
            #         chunk = ChatCompletionChunk(
            #             id="error-response",
            #             choices=[ChatCompletionChunkChoice(
            #                 index=0,
            #                 delta=ChoiceDelta(content=f"Error: {self.error_message}", role="assistant"),
            #                 finish_reason="stop"
            #             )],
            #             model="gpt-4",
            #             object="chat.completion.chunk",
            #             created=1234567890
            #         )
            #         return chunk
                
            #     async def __aenter__(self):
            #         return self
                
            #     async def __aexit__(self, exc_type, exc_val, exc_tb):
            #         pass
            
            # # Return our mock stream
            # return MockOpenAIStream(str(e))
            
    async def simulate_text_stream(self, messages=None):
        """Simulate a text stream that acts as a custom brain for the STT to LLM to TTS pipeline.
        
        This function creates a simple OpenAI-compatible stream with predefined sample texts.
        Each text is sent as a separate chunk with a delay to simulate real-time processing.
        
        Args:
            messages: Optional messages from the user (not used in this implementation).
                     
        Returns:
            An asynchronous stream compatible with OpenAI's streaming format.
        """
        # Import OpenAI types to create proper objects
        import openai
        import os
        import time
        import uuid
        
        # Create a custom stream that returns proper OpenAI-compatible objects
        try:
            # First, let's try to import the necessary OpenAI types
            try:
                from openai.types.chat import ChatCompletionChunk
                from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
                
                # Create a class that returns proper OpenAI objects
                class CustomBrainStream:
                    def __init__(self):
                        self.sample_texts = [
                            "Hello, this is a test of the text-to-speech system."
                        ]
                        self.position = 0
                        self.finished = False
                        self.stream_id = f"brain-{uuid.uuid4()}"
                    
                    def __aiter__(self):
                        return self
                    
                    async def __anext__(self):
                        # If we've gone through all texts and sent the final chunk, stop iteration
                        if self.finished:
                            raise StopAsyncIteration
                            
                        # If we got an empty chunk and we're at the end of the stream
                        if not text:
                            print(f"[Agent Stream] Received empty text, buffer: '{self.buffer}'")
                            # If we have any remaining text in the buffer, send it
                            if self.buffer:
                                print(f"[Agent Stream] Sending remaining buffer: '{self.buffer}'")
                                # Create a chunk with the remaining text
                                chunk = ChatCompletionChunk(
                                    id=self.stream_id,
                                    choices=[Choice(
                                        index=0,
                                        delta=ChoiceDelta(content=self.buffer, role="assistant"),
                                        finish_reason=None
                                    )],
                                    model="gpt-4",
                                    object="chat.completion.chunk",
                                    created=int(time.time()),
                                    usage=None
                                )
                                self.buffer = ""
                                print(f"[Agent Stream] Returning chunk with remaining buffer")
                                return chunk
                            

                            # Mark as finished
                            self.finished = True
                            print("[Agent Stream] Finished, sending final chunk")
                            
                            # No more text, send the final stop chunk
                            final_chunk = ChatCompletionChunk(
                                id=f"chatcmpl-{uuid.uuid4()}",  # Use chatcmpl- prefix to match OpenAI format
                                choices=[Choice(
                                    index=0,
                                    delta=ChoiceDelta(tool_calls=[]),  # Include empty tool_calls
                                    finish_reason="stop"
                                )],
                                model="gpt-4",  # Use a model name that the TTS system recognizes
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                usage=None
                            )
                            print(f"[Agent] Created final chunk: id='{final_chunk.id}' with finish_reason='stop'")
                            return final_chunk                      
                        # Get the current text and create a chunk
                        text = self.sample_texts[self.position]
                        print(f"[TTS Input] Sending text: {text}")
                        
                        # Create a chunk with the current text
                        chunk = ChatCompletionChunk(
                            id=self.stream_id,
                            choices=[Choice(
                                index=0,
                                delta=ChoiceDelta(content=text, role="assistant"),
                                finish_reason=None
                            )],
                            model="custom-brain",
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            usage=None  # No usage information for our custom brain
                        )
                        
                        # Move to next position and add delay
                        self.position += 1
                        await asyncio.sleep(1)  # 1 second delay between texts
                        
                        return chunk
                    
                    async def __aenter__(self):
                        return self
                        
                    async def __aexit__(self, exc_type, exc_val, exc_tb):
                        pass
                
                # Return the custom brain stream
                return CustomBrainStream()
                
            except ImportError as ie:
                print(f"Import error: {str(ie)}. Falling back to create_my_stream.")
                # If we can't import the OpenAI types, fall back to create_my_stream
                return await self.create_my_stream(messages)
            
        except Exception as e:
            print(f"Error in simulate_text_stream: {str(e)}")
            # If there's an error, fall back to create_my_stream
            return await self.create_my_stream(messages)
    

            
    
