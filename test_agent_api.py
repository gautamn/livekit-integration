import asyncio
import os
from agent_api import AgentAPIClient

async def test_agent_api():
    """Test the agent API integration directly."""
    # Create an agent client
    agent_client = AgentAPIClient()
    
    # Test query
    query = "Hello, how are you?"
    conversation_id = "test-conversation-id"
    
    print(f"Sending query to agent API: {query}")
    
    # Call the agent API and print each chunk
    async for chunk in agent_client.call_agent(query=query, conversation_id=conversation_id):
        print(f"Received chunk: {chunk}")
        if "text" in chunk:
            print(f"Text: {chunk['text']}")
    
    print("Agent API test complete")

if __name__ == "__main__":
    asyncio.run(test_agent_api())
