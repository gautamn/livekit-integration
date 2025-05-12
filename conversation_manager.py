import os
import json
import aiohttp
from dotenv import load_dotenv

load_dotenv()

class ConversationManager:
    _conversation_id = None

    @classmethod
    async def initialize(cls):
        """Call API and store the conversation_id once."""
        if cls._conversation_id is not None:
            return  # Already initialized

        query = "Hello"

        try:
            # Create the payload and headers similar to how it's done in agent_api.py
            url = os.getenv("AGENT_API_BASE_URL", "https://app.eng.quant.ai/api/chat-messages")
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {os.getenv("AGENT_API_BEARER_TOKEN")}'
            }
            payload = json.dumps({
                "query": query,
                "conversation_id": "",
                "response_mode": "streaming",
                "channel": "web",
                "inputs": {}
            })

            print(f"[ConversationManager] Making POST request to {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=payload) as response:
                    print(f"[ConversationManager] Got response with status: {response.status}")
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"[ConversationManager] Error response: {response.status} - {error_text}")
                        raise Exception(f"Agent API returned status {response.status}: {error_text}")
                    
                    print("[ConversationManager] Starting to process streaming response")
                    
                    # The response is a text/event-stream with chunks in JSON format
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line:
                            try:
                                # Parse the JSON chunk
                                json_part = line.split("data: ", 1)[1]
                                chunk = json.loads(json_part)
                                
                                # Check if this chunk has conversation_id
                                if "conversation_id" in chunk and chunk["conversation_id"]:
                                    cls._conversation_id = chunk["conversation_id"]
                                    print(f"[ConversationManager] Set conversation_id: {cls._conversation_id}")
                                    # Exit the loop as soon as we get the conversation_id
                                    cls._conversation_id = data.get("conversation_id")
                                    if cls._conversation_id:
                                        print(f"[ConversationManager] Set conversation_id: {cls._conversation_id}")
                                    else:
                                        print("[ConversationManager] conversation_id missing in response")
                                    break

                            except (json.JSONDecodeError, IndexError):
                                print(f"[ConversationManager] Error parsing line: {line}")
                                continue

        except Exception as e:
            print(f"[ConversationManager] Failed to fetch conversation_id: {e}")

    @classmethod
    def get_conversation_id(cls):
        if cls._conversation_id is None:
            raise ValueError("ConversationManager is not initialized. Call initialize() first.")
        return cls._conversation_id
        
    @classmethod
    def set_conversation_id(cls, conversation_id):
        """Manually set the conversation_id."""
        cls._conversation_id = conversation_id
        print(f"[ConversationManager] Manually set conversation_id: {cls._conversation_id}")
