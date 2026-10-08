"""MiniMax reasoning/content stream primitive; not registered as an API profile.

The audited embedded template already supplies the opening <think> marker.
The caller decodes token bytes with a stateful UTF-8 decoder and handles EOS
by token ID. Tool parsing/history normalization remain separate work.
"""
from serve.frontend import Event


class MiniMaxReasoningParser:
    """Split the first closing reasoning marker without losing whitespace.

Only a possible marker suffix is buffered (at most seven characters). On a
length limit/cancel/EOS before the closing marker, finish preserves the partial
reasoning and reasoning_complete stays false. It never invents final content.
Literal EOS spellings and tool-like text are preserved, not executed.
"""
    END = '</think>'

    def __init__(self):
        self.reasoning_complete = False
        self._pending = ''
        self._finished = False

    def feed(self, delta):
        if self._finished:
            raise ValueError('MiniMax stream is already finished')
        if not isinstance(delta, str):
            raise TypeError('feed decoded text; use an incremental UTF-8 decoder for token bytes')
        if not delta:
            return []
        if self.reasoning_complete:
            return [Event('content', delta)]
        text = self._pending+delta
        at = text.find(self.END)
        if at >= 0:
            self.reasoning_complete = True
            self._pending = ''
            events = [Event('reasoning', text[:at])] if at else []
            final = text[at+len(self.END):]
            if final:
                events.append(Event('content', final))
            return events
        hold = 0
        for length in range(1, len(self.END)):
            if text.endswith(self.END[:length]):
                hold = length
        end = len(text)-hold
        self._pending = text[end:]
        return [Event('reasoning', text[:end])] if end else []

    def finish(self):
        if self._finished:
            return []
        self._finished = True
        text, self._pending = self._pending, ''
        return [Event('reasoning', text)] if text else []
