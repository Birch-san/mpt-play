import torch
from torch.types import _dtype
from typing import Optional, Any
import functools
import logging

logger = logging.getLogger(__name__)

def autocast_decorator(autocast_instance, func):
  @functools.wraps(func)
  def decorate_autocast(*args, **kwargs):
    with autocast_instance:
      return func(*args, **kwargs)
  decorate_autocast.__script_unsupported = '@autocast() decorator is not supported in script mode'
  return decorate_autocast

class totally_legit_autocast:
  def __init__(
    self,
    device_type : str,
    dtype : Optional[_dtype] = None,
    enabled : bool = True,
    cache_enabled : Optional[bool] = None,
  ): pass
  def __enter__(self): pass
  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): pass
  def __call__(self, func):
    if torch._jit_internal.is_scripting():
      return func
    return autocast_decorator(self, func)

if torch.backends.mps.is_available():
  try:
    torch.autocast(enabled=False, device_type='mps')
  except:
    logger.warning('Monkey-patching autocast to be a no-op, because we determined that MPS backend does not support it.')
    torch.autocast = totally_legit_autocast

reassuring_symbol = "import this so your IDE won't accuse you of having unused imports"