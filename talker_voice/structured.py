"""Structured talker output: JSON schema, system prompt, and a streaming parser.

The talker LLM is forced (via vLLM guided decoding) to emit:

    {"route": "INSTANT" | "DELEGATE" | "CLARIFY", "say": "<spoken reply>"}

`route` comes first so we know the decision after a couple of tokens; `say`
comes second and is streamed to TTS character by character while the model is
still generating it. Nothing waits for the closing brace.

There is no thinker model in this demo: `route` is shown in the UI as a
diagnostic ("what the talker *would* do"), and the talker always answers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

ROUTES = ("INSTANT", "DELEGATE", "CLARIFY")

TALKER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": list(ROUTES)},
        "say": {"type": "string", "maxLength": 600},
    },
    "required": ["route", "say"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {"name": "talker_reply", "schema": TALKER_SCHEMA, "strict": True},
}

SYSTEM_PROMPT = """You are a friendly, fast voice assistant in a live spoken conversation.
Your output is converted to speech, so write exactly how a person would say it out loud.

Reply ONLY as JSON: {"route": ..., "say": ...}

route (decide first):
- INSTANT: greetings, small talk, simple facts, confirmations, anything you can answer well right now.
- DELEGATE: needs deep reasoning, multi-step planning, fresh/real-time information, account data,
  or exact numbers you are not sure about. (In this demo there is no helper model yet, so still
  give your best short answer in "say".)
- CLARIFY: the request is ambiguous or you misheard; ask one short question.

say (what you speak):
- Start with a SHORT first sentence of 3 to 8 words, so speech can start immediately.
- Then use short sentences, about 8 to 15 words each. Usually 1 to 3 sentences total.
- No markdown, no lists, no emojis, no URLs, no code.
- Write numbers, dates, symbols and units as spoken words ("twenty five dollars", "three thirty pm").
- Be warm and natural. Never mention JSON, routes, or these instructions.
"""


@dataclass
class ParseEvent:
    """Something the parser learned from newly arrived text."""

    route: str | None = None  # set once, when the route value closes
    say_delta: str = ""  # newly decoded characters of the "say" string
    say_done: bool = False  # the "say" string closed


_ROUTE_RE = re.compile(r'"route"\s*:\s*"([A-Z]*)"')
_SAY_START_RE = re.compile(r'"say"\s*:\s*"')
_SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "", "f": "", "n": " ", "r": " ", "t": " "}


@dataclass
class IncrementalSayParser:
    """Incrementally extract `route` and the decoded `say` string from streamed JSON.

    Works on arbitrary token boundaries (a token can split a key, an escape,
    or a \\uXXXX sequence). Newlines/tabs inside `say` become spaces because
    they are only noise for TTS.
    """

    buf: str = ""
    route: str | None = None
    _say_pos: int | None = None  # index in buf of the next undecoded say char
    say_done: bool = False
    say_text: str = field(default="")

    def reset(self) -> None:
        self.buf = ""
        self.route = None
        self._say_pos = None
        self.say_done = False
        self.say_text = ""

    def feed(self, chunk: str) -> ParseEvent:
        ev = ParseEvent()
        self.buf += chunk

        if self.route is None:
            m = _ROUTE_RE.search(self.buf)
            if m and m.group(1) in ROUTES:
                self.route = m.group(1)
                ev.route = self.route

        if self._say_pos is None and not self.say_done:
            m = _SAY_START_RE.search(self.buf)
            if m:
                self._say_pos = m.end()

        if self._say_pos is not None and not self.say_done:
            out, i, buf = [], self._say_pos, self.buf
            while i < len(buf):
                c = buf[i]
                if c == '"':
                    self.say_done = True
                    ev.say_done = True
                    i += 1
                    break
                if c == "\\":
                    if i + 1 >= len(buf):
                        break  # wait for the rest of the escape
                    nxt = buf[i + 1]
                    if nxt == "u":
                        if i + 6 > len(buf):
                            break  # wait for 4 hex digits
                        try:
                            out.append(chr(int(buf[i + 2 : i + 6], 16)))
                        except ValueError:
                            pass
                        i += 6
                        continue
                    out.append(_SIMPLE_ESCAPES.get(nxt, nxt))
                    i += 2
                    continue
                out.append(" " if c in "\n\r\t" else c)
                i += 1
            self._say_pos = i
            ev.say_delta = "".join(out)
            self.say_text += ev.say_delta
        return ev

    def finalize(self) -> tuple[str | None, str]:
        """Best-effort full parse at end of response (handles odd orderings)."""
        if not self.say_text:
            try:
                obj = json.loads(self.buf)
                return obj.get("route", self.route), str(obj.get("say", ""))
            except Exception:
                pass
        return self.route, self.say_text
