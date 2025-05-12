import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the main package
sys.path.append(str(Path(__file__).parent.parent))

from agent_llm import LLM
from dotenv import load_dotenv
from utils.logging_utils import get_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = get_logger("test_agent_integration")

async def test_agent_stream():
    """Test the agent stream integration."""
    # Create an LLM instance
    openai_api_key = os.getenv("OPENAI_API_KEY")
    logger.info(f"Using OpenAI API key: {openai_api_key[:4]}..." if openai_api_key else "OpenAI API key not found")
    llm = LLM(api_key=openai_api_key)
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # Use a specific conversation ID for testing (or None to generate a new one)
    conversation_id = "2f652be8-9cfe-4e88-8acd-eb785d94699b"
    
    logger.info(f"Testing agent stream with conversation ID: {conversation_id}")
    logger.info(f"Query: {messages[-1]['content']}")
    logger.info("Response:")
    
    # Now we can call the simulate_agent_stream method directly on the LLM object
    stream = await llm.simulate_agent_stream(messages, conversation_id)
    
    # Process the stream
    full_response = ""
    async for chunk in stream:
        # Extract the content from the chunk
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            # Print to console for real-time feedback
            print(content, end="", flush=True)
    
    logger.info(f"\nFull response: {full_response}")
    logger.info("Stream completed.")

if __name__ == "__main__":
    asyncio.run(test_agent_stream())
