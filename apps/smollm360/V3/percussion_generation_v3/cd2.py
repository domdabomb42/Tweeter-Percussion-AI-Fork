"""CD2: exact-count decoding over distinct tracker steps in any order."""
from .data import END_MARKER


class PositionConstraintCD2:
    """Allow any unused position 0..63 next, then end marker and EOS.

    State tracks completed positions as a set. Partial tokenizer pieces are
    retained so multi-token numbers cannot bypass the no-duplicate rule.
    """

    def __init__(self, tokenizer, count):
        if type(count) is not int or not 1 <= count <= 64:
            raise ValueError("count must be 1..64")
        self.count = count
        self.eos = tokenizer.eos_token_id
        self.pieces = {(first, step): tuple(tokenizer.encode(
            ("" if first else " ") + str(step), add_special_tokens=False))
            for first in (False, True) for step in range(64)}
        self.end = tuple(tokenizer.encode("\n" + END_MARKER,
                                          add_special_tokens=False))
        if self.eos is None or not self.end or any(not piece for piece in self.pieces.values()):
            raise ValueError("tokenizer cannot encode CD2 rhythm grammar")
        self.cache = {(): frozenset({(frozenset(), (), -1, False)})}

    def _options(self, used):
        if len(used) == self.count:
            return [(self.end, -1, True)]
        first = not used
        return [(self.pieces[(first, step)], step, False)
                for step in range(64) if step not in used]

    def _advance(self, states, token):
        result = set()
        for used, pending, selected, ending in states:
            if ending and not pending:
                if token == self.eos:
                    result.add((used, (), -1, True))
                continue
            options = [(pending, selected, ending)] if pending else self._options(used)
            for ids, step, end_flag in options:
                if ids and ids[0] == token:
                    tail = ids[1:]
                    if tail:
                        result.add((used, tail, step, end_flag))
                    elif end_flag:
                        result.add((used, (), -1, True))
                    else:
                        result.add((used | {step}, (), -1, False))
        return frozenset(result)

    def states(self, generated):
        key = tuple(generated)
        if key not in self.cache:
            self.cache[key] = self._advance(self.states(key[:-1]), key[-1])
        return self.cache[key]

    def allowed(self, generated):
        allowed = set()
        for used, pending, _, ending in self.states(generated):
            if ending and not pending:
                allowed.add(self.eos)
            elif pending:
                allowed.add(pending[0])
            else:
                allowed.update(ids[0] for ids, _, _ in self._options(used))
        if not allowed:
            raise ValueError("CD2 has no legal next token")
        return sorted(allowed)
