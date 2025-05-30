import requests
import json
import asyncio
import aiohttp
import time
import uuid
import logging
from typing import Optional, Dict, Any, AsyncIterator
import os
from dotenv import load_dotenv
from utils.logging_utils import get_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = get_logger("agent_api")


class AgentAPIClient:
    """Client for interacting with the third-party agent API."""

    bearer_token = os.getenv("AGENT_API_BEARER_TOKEN")
    
    def __init__(self, base_url: str = os.getenv("AGENT_API_BASE_URL", "https://app.eng.quant.ai/api/chat-messages")):
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.bearer_token}'
        }
    
    async def call_agent(self, 
                         query: str, 
                         call_id: Optional[str] = None, 
                         from_number: Optional[str] = None, 
                         conversation_id: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Call the agent API and return a streaming response.
        
        Args:
            query: The user's query to send to the agent
            call_id: A unique ID for the call (generated if not provided)
            from_number: The phone number the call is coming from
            conversation_id: ID for continuing a conversation (generated if not provided)
            
        Returns:
            An async iterator yielding response chunks from the agent
        """
            
        payload = json.dumps({
            "query": query,
            "conversation_id": "3766fcb0-884a-4ab9-8717-ca39d0230c8d",
        print(f"****** [Agent] Calling agent with query: {query}, conversation_id: {conversation_id}")    
        
        payload = json.dumps({
            "query": query,
            "conversation_id": os.getenv("CONVERSATION_ID"),
            "response_mode": "streaming",
            "channel": "web",
            "inputs": {}
        })
        
        logger.info(f"Making POST request to {self.base_url}")
        logger.debug(f"Request payload: {payload}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers, data=payload) as response:
                logger.info(f"Got response with status: {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error response: {response.status} - {error_text}")
                    raise Exception(f"Agent API returned status {response.status}: {error_text}")
                
                logger.info("Starting to process streaming response")
                # Flag to track if we've yielded any chunks with text
                has_yielded_text = False
                
                # The response is a text/event-stream with chunks in JSON format
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line:
                        logger.debug(f"Received line: {line[:100]}..." if len(line) > 100 else f"Received line: {line}")
                        try:
                            # Parse the JSON chunk
                            json_part = line.split("data: ", 1)[1]
                            chunk = json.loads(json_part)
                            logger.debug(f"Yielding chunk: {chunk}")

                            # if chunk.get("event") == "message_end" or chunk.get("event") == "agent_thought":
                            #     pass
                            
                            # Check if this chunk has text content
                            if chunk.get("event") == "agent_message" and "answer" in chunk and chunk["answer"]:
                                has_yielded_text = True
                                yield chunk["answer"]

                        except json.JSONDecodeError:
                            logger.warning(f"JSON decode error for line: {line}")
                            # Skip lines that aren't valid JSON
                            continue
                
                # If we haven't yielded any chunks with text, yield a default response
                if not has_yielded_text:
                    logger.warning("No text content received from API, yielding default response")
                    default_response = {
                        "text": "I'm sorry, I couldn't process your request properly. Please try again."
                    }
                    yield default_response
                
                logger.info("Finished processing streaming response")

    async def call_agent_sync(self, 
                             query: str, 
                             call_id: Optional[str] = None, 
                             conversation_id: Optional[str] = None) -> str:
        """
        Call the agent API and return the complete response as a string.
        This is a non-streaming version that collects all chunks.
        
        Args:
            query: The user's query to send to the agent
            call_id: A unique ID for the call (generated if not provided)
            from_number: The phone number the call is coming from
            conversation_id: ID for continuing a conversation (generated if not provided)
            
        Returns:
            The complete text response from the agent
        """
        full_response = ""
        async for chunk in self.call_agent(query, call_id, from_number, conversation_id):
            if "text" in chunk:
                full_response += chunk["text"]
        
        return full_response


# Example usage
async def example():
    agent = AgentAPIClient()
    async for chunk in agent.call_agent("Hello, how are you?"):
        print(chunk)

if __name__ == "__main__":
    asyncio.run(example())
