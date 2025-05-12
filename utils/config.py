import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """
    Centralized configuration management for the application.
    Loads configuration from environment variables with fallbacks.
    """
    
    # API Configuration
    AGENT_API_BASE_URL = os.getenv("AGENT_API_BASE_URL", "https://app.eng.quant.ai/api/chat-messages")
    AGENT_API_BEARER_TOKEN = os.getenv("AGENT_API_BEARER_TOKEN", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # LiveKit Configuration
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
    
    # Feature Flags
    USE_AGENT_API = os.getenv("USE_AGENT_API", "false").lower() == "true"
    
    # Default conversation ID (should be moved to a database in production)
    DEFAULT_CONVERSATION_ID = os.getenv("DEFAULT_CONVERSATION_ID", "")
    
    # HTTP Client Configuration
    HTTP_TIMEOUT_CONNECT = float(os.getenv("HTTP_TIMEOUT_CONNECT", "15.0"))
    HTTP_TIMEOUT_READ = float(os.getenv("HTTP_TIMEOUT_READ", "30.0"))
    HTTP_TIMEOUT_WRITE = float(os.getenv("HTTP_TIMEOUT_WRITE", "10.0"))
    HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
    
    # Model Configuration
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    DEFAULT_STT_MODEL = os.getenv("DEFAULT_STT_MODEL", "nova-3")
    DEFAULT_STT_LANGUAGE = os.getenv("DEFAULT_STT_LANGUAGE", "multi")
    
    @classmethod
    def get_http_timeouts(cls) -> Dict[str, float]:
        """
        Get HTTP timeout configuration.
        
        Returns:
            Dictionary with timeout configurations
        """
        return {
            "connect": cls.HTTP_TIMEOUT_CONNECT,
            "read": cls.HTTP_TIMEOUT_READ,
            "write": cls.HTTP_TIMEOUT_WRITE,
        }
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            True if all required configuration is present, False otherwise
        """
        required_vars = [
            "AGENT_API_BEARER_TOKEN" if cls.USE_AGENT_API else None,
            "OPENAI_API_KEY",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET"
        ]
        
        # Filter out None values
        required_vars = [var for var in required_vars if var is not None]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var, None):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"Missing required configuration: {', '.join(missing_vars)}")
            return False
        
        return True

# Singleton instance
config = Config()
