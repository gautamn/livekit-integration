import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the main package
sys.path.append(str(Path(__file__).parent.parent))

from agent_api import AgentAPIClient
from utils.logging_utils import get_logger

# Set up logger
logger = get_logger("test_agent_api")

async def test_agent_api():
    """Test the agent API integration directly."""
    # Create an agent client
    agent_client = AgentAPIClient()
    
    # Test query
    query = "Hello, how are you?"
    conversation_id = "test-conversation-id"
    
    logger.info(f"Sending query to agent API: {query}")
    
    # Call the agent API and print each chunk
    async for chunk in agent_client.call_agent(query=query, conversation_id=conversation_id):
        logger.info(f"Received chunk: {chunk}")
        if "text" in chunk:
            logger.info(f"Text: {chunk['text']}")
    
    logger.info("Agent API test complete")

if __name__ == "__main__":
    asyncio.run(test_agent_api())
