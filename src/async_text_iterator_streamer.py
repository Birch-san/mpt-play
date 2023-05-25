from transformers import TextStreamer, AutoTokenizer
import asyncio
from typing import Optional

class AsyncTextIteratorStreamer(TextStreamer):
  def __init__(
      self, tokenizer: AutoTokenizer, skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
    ):
    super().__init__(tokenizer, skip_prompt, **decode_kwargs)
    self.text_queue = asyncio.Queue()
    self.stop_signal = '<|im_end|>'
    self.timeout = timeout
  
  async def on_finalized_text_async(self, text: str, stream_end: bool = False):
    async with asyncio.timeout(self.timeout):
      await self.text_queue.put(text)
    if stream_end:
      async with asyncio.timeout(self.timeout):
        await self.text_queue.put(self.stop_signal)
  
  def on_finalized_text(self, text: str, stream_end: bool = False):
    """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
    loop = asyncio.get_event_loop()
    loop.create_task(self.on_finalized_text_async(text, stream_end))
  
  async def gen(self):
    while True:
      async with asyncio.timeout(self.timeout):
        # TODO: for some reason this await doesn't complete until the queue is completely full.
        #       I don't know why the event loop doesn't return control to this statement.
        value = await self.text_queue.get()
        if value == self.stop_signal:
          break
        yield value