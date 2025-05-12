import asyncio
import sys
from pathlib import Path
from typing import AsyncIterable

# Add the parent directory to sys.path to allow imports from the main package
sys.path.append(str(Path(__file__).parent.parent))

from livekit import agents, rtc
from livekit.agents.tts import SynthesizedAudio
from livekit.plugins import cartesia
import os
from dotenv import load_dotenv
from utils.logging_utils import get_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = get_logger("test_tts")

async def simulate_text_stream() -> AsyncIterable[str]:
    sample_texts = [
        "Hello, this is a test of the text-to-speech system.",
        "This stream simulates live input from a user.",
        "Each line will be converted to speech and streamed.",
    ]
    for text in sample_texts:
        logger.info(f"Sending text to TTS: {text}")
        await asyncio.sleep(1)
        yield text

async def entrypoint(ctx: agents.JobContext):
    logger.info("Starting TTS test")
    try:
        await ctx.connect()
        logger.info("Connected to room")

        if not os.getenv("CARTESIA_API_KEY"):
            raise ValueError("CARTESIA_API_KEY must be set")

        text_stream = simulate_text_stream()
        audio_source = rtc.AudioSource(44100, 1)

        track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
        await ctx.room.local_participant.publish_track(track)
        logger.info("Audio track published")

        tts = cartesia.TTS(model="sonic-english")
        tts_stream = tts.stream()  # use `await` if needed

        async def send_audio(audio_stream: AsyncIterable[SynthesizedAudio]):
            async for a in audio_stream:
                if not a.audio or not a.audio.frame:
                    logger.warning("Empty audio frame received!")
                else:
                    logger.debug(f"Capturing audio frame: {len(a.audio.frame)} bytes")
                await audio_source.capture_frame(a.audio.frame)

        asyncio.create_task(send_audio(tts_stream))

        async for text in text_stream:
            tts_stream.push_text(text)
        tts_stream.end_input()
        logger.info("TTS test completed")
    except Exception as e:
        logger.error(f"Error in TTS test: {str(e)}")
        raise

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
