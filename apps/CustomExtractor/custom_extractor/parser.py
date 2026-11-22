"""Pd binbuf / cyclone coll syntax. Record numbers are keys, not timestamps."""
from dataclasses import dataclass
import math
import re


class FormatError(ValueError):
    pass


NUMBER = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
SPACE = " \t\r\n"


@dataclass(frozen=True)
class Atom:
    value: int | float | str
    raw: str


@dataclass(frozen=True)
class Record:
    keys: tuple[Atom, ...]
    atoms: tuple[Atom, ...]
    start: int
    end: int

    @property
    def values(self):
        return [a.value for a in self.atoms]


def _atom(raw):
    decoded, escaped, i = [], False, 0
    while i < len(raw):
        if raw[i] == "\\":
            escaped = True
            i += 1
            if i == len(raw):
                raise FormatError("dangling backslash in atom")
        decoded.append(raw[i])
        i += 1
    value = "".join(decoded)
    if not escaped and NUMBER.fullmatch(value):
        number = float(value)
        if not math.isfinite(number):
            raise FormatError(f"non-finite native number: {value!r}")
        value = int(value) if re.fullmatch(r"-?[0-9]+", value) else number
    return Atom(value, raw)


def parse(text: str) -> list[Record]:
    """Preserve offsets/lexemes; backslash escapes *any* next character in Pd."""
    if not isinstance(text, str) or "\x00" in text:
        raise FormatError("native text must be a string without NUL bytes")
    records, tokens, keys = [], [], None
    start = 1 if text.startswith("\ufeff") else 0
    i, token_start = start, None
    while i < len(text):
        char = text[i]
        if char == "\\":
            if i + 1 == len(text):
                raise FormatError(f"dangling escape at character {i}")
            if token_start is None:
                token_start = i
            i += 2
            continue
        if char in SPACE + ",;":
            if token_start is not None:
                tokens.append(_atom(text[token_start:i]))
                token_start = None
            if char == ",":
                if keys is not None or not 1 <= len(tokens) <= 2:
                    raise FormatError(f"invalid coll key/comma at character {i}")
                # Native coll supports integer keys, symbolic keys, or both.
                numeric = [a.value for a in tokens if not isinstance(a.value, str)]
                if len(numeric) > 1 or any(int(n) != n for n in numeric):
                    raise FormatError(f"coll key must be an integer or symbol at character {i}")
                if len(tokens) == 2 and len(numeric) != 1:
                    raise FormatError(f"invalid coll key pair at character {i}")
                keys, tokens = tuple(tokens), []
            elif char == ";":
                if keys is None or not tokens:
                    raise FormatError(f"missing coll key, comma, or payload at character {i}")
                records.append(Record(keys, tuple(tokens), start, i + 1))
                keys, tokens, start = None, [], i + 1
        elif token_start is None:
            token_start = i
        i += 1
    if token_start is not None or tokens or keys is not None:
        raise FormatError("unterminated native coll record (expected ';')")
    if not records:
        raise FormatError("empty file: no native Tweeter coll records")
    return records


def atom_text(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise FormatError(f"unsupported native atom: {value!r}")
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or abs(value) > 3.402823466e38:
            raise FormatError(f"number outside native float32 range: {value!r}")
        return repr(value)
    if not value or "\x00" in value or len(value.encode("utf-8", errors="surrogateescape")) >= 1000:
        raise FormatError("native symbols must contain 1..999 UTF-8 bytes and no NUL")
    result = "".join("\\" + c if c in SPACE + "\\,;$" else c for c in value)
    # Pd uses an escape to distinguish a numeric-looking symbol from a float.
    if NUMBER.fullmatch(value) and "\\" not in result:
        result = "\\" + result
    return result


def record_text(keys, values):
    return " ".join(atom_text(v) for v in keys) + ", " + " ".join(atom_text(v) for v in values) + ";"
