import json
import random

from talker_voice.structured import IncrementalSayParser


def _stream(text: str, parser: IncrementalSayParser, sizes=(1, 2, 3, 5, 8)):
    rnd = random.Random(0)
    said, route = "", None
    i = 0
    while i < len(text):
        n = rnd.choice(sizes)
        ev = parser.feed(text[i : i + n])
        i += n
        said += ev.say_delta
        route = route or ev.route
    return route, said


def test_basic_streaming_any_chunking():
    obj = {"route": "INSTANT", "say": "Sure thing. It is twenty five degrees today."}
    for sizes in [(1,), (2,), (7,), (1, 4, 9)]:
        p = IncrementalSayParser()
        route, said = _stream(json.dumps(obj), p, sizes)
        assert route == "INSTANT"
        assert said == obj["say"]
        assert p.say_done


def test_escapes_and_unicode_split_across_tokens():
    obj = {"route": "CLARIFY", "say": 'He said "hi"\nthen left é \\ ok'}
    raw = json.dumps(obj)  # contains \" \n é \\
    p = IncrementalSayParser()
    route, said = _stream(raw, p, (1,))
    assert route == "CLARIFY"
    assert said == 'He said "hi" then left é \\ ok'


def test_say_streams_before_json_closes():
    p = IncrementalSayParser()
    ev = p.feed('{"route": "DELEGATE", "say": "Let me ch')
    assert ev.route == "DELEGATE"
    assert ev.say_delta == "Let me ch"
    assert not p.say_done


def test_reset():
    p = IncrementalSayParser()
    p.feed('{"route":"INSTANT","say":"a"}')
    p.reset()
    ev = p.feed('{"route":"CLARIFY","say":"b"}')
    assert ev.route == "CLARIFY" and ev.say_delta == "b"


def test_tone_parsed_before_say():
    p = IncrementalSayParser()
    ev = p.feed('{"route": "INSTANT", "tone": "cheerful", "say": "Hi')
    assert ev.route == "INSTANT" and ev.tone == "cheerful" and ev.say_delta == "Hi"


def test_schema_order_route_tone_say():
    from talker_voice.structured import TALKER_SCHEMA
    assert list(TALKER_SCHEMA["properties"]) == ["route", "tone", "say"]
