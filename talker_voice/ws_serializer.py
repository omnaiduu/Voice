"""Minimal WebSocket protocol for the browser client.

Browser -> server:  binary = raw PCM16 mono 16 kHz mic audio; text "ping" ignored.
Server -> browser:  binary = raw PCM16 mono 24 kHz bot audio;
                    text   = JSON events ({"type": ...}); {"type": "clear"} on barge-in
                    tells the browser to drop audio it has buffered but not played.
"""

from __future__ import annotations

import json

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

IN_RATE = 16000


class RawPCMSerializer(FrameSerializer):
    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message)
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "clear"})
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            if len(data) % 2:
                data = data[:-1]
            return InputAudioRawFrame(audio=bytes(data), sample_rate=IN_RATE, num_channels=1)
        return None  # text frames (pings) carry nothing for the pipeline
