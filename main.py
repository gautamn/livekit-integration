from dotenv import load_dotenv

from livekit import agents
import os

from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from conversation_manager import ConversationManager


from agent_llm import LLM
from utils.logging_utils import get_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = get_logger("main")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    logger.info("Starting agent entrypoint")
    try:
        logger.info("Connecting to room")
        await ctx.connect()
        logger.info("Successfully connected to room")

        logger.info("Initializing agent session with components")
        session = AgentSession(
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )
        logger.info("Agent session initialized successfully")

        logger.info("Initializing conversation manager")
        await ConversationManager.initialize()
        logger.info("Conversation manager initialized successfully")

        cid = ConversationManager.get_conversation_id()
        print("#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Using conversation_id:", cid)

        logger.info("Starting agent session")
        await session.start(
            room=ctx.room,
            agent=Assistant(),
            room_input_options=RoomInputOptions(
                # LiveKit Cloud enhanced noise cancellation
                # - If self-hosting, omit this parameter
                # - For telephony applications, use `BVCTelephony` for best results
                noise_cancellation=noise_cancellation.BVC(), 
            ),
        )
        logger.info("Agent session started successfully")

        logger.info("Generating initial greeting")
        await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )
        logger.info("Initial greeting generated")
    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}")
        raise


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))