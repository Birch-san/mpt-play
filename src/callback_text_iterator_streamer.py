from transformers import AutoTokenizer, TextIteratorStreamer
from typing import Optional, Protocol

class TextCallback(Protocol):
  def __call__(self, text: str, stream_end: bool = False) -> None: ...


class CallbackTextIteratorStreamer(TextIteratorStreamer):
  callback: TextCallback
  def __init__(
      self, tokenizer: AutoTokenizer, callback: TextCallback, skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
    ):
    super().__init__(tokenizer, skip_prompt, **decode_kwargs)
    self.callback = callback

  def on_finalized_text(self, text: str, stream_end: bool = False):
    self.callback(text, stream_end=stream_end)
