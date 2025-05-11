import asyncio
from livekit import agents, rtc
from livekit.agents.tts import SynthesizedAudio
from livekit.plugins import cartesia
from typing import AsyncIterable
import os
from dotenv import load_dotenv
load_dotenv()

async def simulate_text_stream() -> AsyncIterable[str]:
    sample_texts = [
        "Hello, this is a test of the text-to-speech system.",
        "This stream simulates live input from a user.",
        "Each line will be converted to speech and streamed.",
    ]
    for text in sample_texts:
        print(f"[TTS Input] Sending text: {text}")
        await asyncio.sleep(1)
        yield text

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    print("[Agent] Connected to room")

    if not os.getenv("CARTESIA_API_KEY"):
        raise ValueError("CARTESIA_API_KEY must be set")

    text_stream = simulate_text_stream()
    audio_source = rtc.AudioSource(44100, 1)

    track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
    await ctx.room.local_participant.publish_track(track)
    print("[Audio] Track published")

    tts = cartesia.TTS(model="sonic-english")
    tts_stream = tts.stream()  # use `await` if needed

    async def send_audio(audio_stream: AsyncIterable[SynthesizedAudio]):
        async for a in audio_stream:
            if not a.audio or not a.audio.frame:
                print("[Audio] Empty frame received!")
            else:
                print(f"[Audio] Capturing frame: {len(a.audio.frame)} bytes")
            await audio_source.capture_frame(a.audio.frame)

    asyncio.create_task(send_audio(tts_stream))

    async for text in text_stream:
        tts_stream.push_text(text)
    tts_stream.end_input()

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

