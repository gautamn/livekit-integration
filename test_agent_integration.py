import asyncio, os
from agent_llm import LLM
from dotenv import load_dotenv
load_dotenv()

async def test_agent_stream():
    """Test the agent stream integration."""
    # Create an LLM instance
    openai_api_key = os.getenv("OPENAI_API_KEY")
    print(f"Using OpenAI API key: {openai_api_key}")
    llm = LLM(api_key=openai_api_key)
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # Use a specific conversation ID for testing (or None to generate a new one)
    #conversation_id = "test-conversation-" + asyncio.current_task().get_name()
    conversation_id = "2f652be8-9cfe-4e88-8acd-eb785d94699b"
    
    print(f"Testing agent stream with conversation ID: {conversation_id}")
    print(f"Query: {messages[-1]['content']}")
    print("Response:")
    
    # Now we can call the simulate_agent_stream method directly on the LLM object
    stream = await llm.simulate_agent_stream(messages, conversation_id)
    
    # Process the stream
    full_response = ""
    async for chunk in stream:
        # Extract the content from the chunk
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)
    
    print("\n\nFull response:", full_response)
    print("Stream completed.")

if __name__ == "__main__":
    asyncio.run(test_agent_stream())
