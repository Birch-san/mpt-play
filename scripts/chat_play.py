from dataclasses import dataclass, field
from typing import Optional, TypedDict, NamedTuple, List
# we gotta stub torch.autocast before the MPT-7B model imports it
from src.mps_autocast_stub import reassuring_symbol
import torch
from torch import LongTensor
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  GenerationConfig,
  HfArgumentParser,
  set_seed,
  StoppingCriteria,
  StoppingCriteriaList,
)
from src.device_map import device_map
from src.callback_text_iterator_streamer import CallbackTextIteratorStreamer
import logging
from enum import Enum
import sys

# no unused imports for me
reassuring_symbol

logger = logging.getLogger(__name__)

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor

class Participant(Enum):
  User = 'user'
  Assistant = 'assistant'
  System = 'system'

class Message(NamedTuple):
  participant: Participant
  message: str

@dataclass
class StopOnTokens(StoppingCriteria):
  stop_token_ids: List[int]
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    for stop_id in self.stop_token_ids:
      if input_ids[0][-1] == stop_id:
        return True
    return False

@dataclass
class ModelArguments:
  model_name_or_path: Optional[str] = field(
    default="mosaicml/mpt-7b-chat"
  )
  trust_remote_code: Optional[bool] = field(
    default=False,
    metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
  )
  double_quant: bool = field(
    default=True,
    metadata={"help": "Compress the quantization statistics through double quantization."}
  )
  quant_type: str = field(
    default="nf4",
    metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
  )
  bits: int = field(
    default=4,
    metadata={"help": "How many bits to use."}
  )
  bf16: Optional[bool] = field(
    default=False,
    metadata={"help": "Compute type of the model. If quantizing: this is also the compute type used for quantized computations. Prefer to turn this on if you are quantizing and your GPU supports it. You probably also want it even if you're not quantizing."}
  )
  context_length: int = field(
    default=2048,
    metadata={"help": "How many bits to use."}
  )

@dataclass
class MiscArguments:
  seed: Optional[int] = field(
    default=64,
    metadata={"help": "Random seed, for deterministic generation."}
  )
  compile: bool = field(
    default=False,
    metadata={"help": "Invoke torch.compile() on the model, with mode='max-autotune'. Requires PyTorch 2, CUDA, and either Python 3.10 or Python 3.11 with a recent torch nightly. Will make the first inference from the model take a bit longer, but subsequent inferences will be faster."}
  )
  use_system_prompt: bool = field(
    default=False,
    metadata={"help": "There is a system prompt used in MosaicML's MPT-7B-Chat demo, but in my brief testing it seemed that the model didn't listen to what I wrote in the system prompt… so I disable it by default, to save you some context length."}
  )

@dataclass
class GenerationArguments:
  # For more hyperparameters check:
  # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
  # Length arguments
  max_new_tokens: Optional[int] = field(
    default=256,
    metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                      "if predict_with_generate is set."}
  )
  min_new_tokens : Optional[int] = field(
    default=None,
    metadata={"help": "Minimum number of new tokens to generate."}
  )

  # Generation strategy
  do_sample: Optional[bool] = field(default=False)
  num_beams: Optional[int] = field(default=1)
  num_beam_groups: Optional[int] = field(default=1)
  penalty_alpha: Optional[float] = field(default=None)
  use_cache: Optional[bool] = field(default=True)

  # Hyperparameters for logit manipulation
  temperature: Optional[float] = field(default=1.0)
  top_k: Optional[int] = field(default=50)
  top_p: Optional[float] = field(default=1.0)
  typical_p: Optional[float] = field(default=1.0)
  diversity_penalty: Optional[float] = field(default=0.0)
  repetition_penalty: Optional[float] = field(default=1.0)
  length_penalty: Optional[float] = field(default=1.0)
  no_repeat_ngram_size: Optional[int] = field(default=0)

def get_model(args: ModelArguments) -> AutoModelForCausalLM:
  config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=args.trust_remote_code,
  )
  config.update({"max_seq_len": args.context_length}) # was originally trained on 2048
  cuda_avail = torch.cuda.is_available()
  compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
  load_in_4bit = args.bits == 4 and cuda_avail
  load_in_8bit = args.bits == 8 and cuda_avail

  quantization_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.double_quant,
    bnb_4bit_quant_type=args.quant_type,
  ) if cuda_avail else None

  if not cuda_avail:
    logger.warning("You don't have CUDA, so we have turned off quantization. If you happen to be on a Mac: you probably have enough unified memory to run in fp16 anyway…")

  if compute_dtype == torch.float16 and cuda_avail and torch.cuda.is_bf16_supported():
    print("Your GPU supports bfloat16; you may want to try it with --bf16 (note: I'm not sure how important this is for inference, but it's certainly preferred when training with 4-bit quantization.)")
  
  device_map_ = { key: 'mps' for key in device_map } if torch.backends.mps.is_available() else device_map

  model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    device_map=device_map_,
    quantization_config=quantization_config,
    torch_dtype=compute_dtype,
    trust_remote_code=args.trust_remote_code,
  ).eval()
  model.config.torch_dtype=compute_dtype

  return model

def main():
  hfparser = HfArgumentParser((ModelArguments, GenerationArguments, MiscArguments))
  model_args, generation_args, misc_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
  if extra_args:
    raise f"Received unsupported command-line args: {extra_args}"
  generation_config = GenerationConfig(**vars(generation_args))
  model: AutoModelForCausalLM = get_model(model_args)
  set_seed(misc_args.seed)
  if misc_args.compile:
    torch.compile(model, mode='max-autotune')

  tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    trust_remote_code=model_args.trust_remote_code,
    cache_dir=None,
    padding_side="right",
    use_fast=True,
  )

  stop_token_ids: List[int] = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
  stop = StopOnTokens(stop_token_ids)
  stopping_criteria=StoppingCriteriaList([stop])

  system_prompt = """- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.
"""

  history: List[Message] = [Message(Participant.System, system_prompt)] if misc_args.use_system_prompt else []

  reset_ansi='\x1b[0m'
  blue_ansi='\x1b[31;34m'
  green_ansi='\x1b[31;32m'
  purple_ansi='\x1b[31;35m'
  prompt=f'{purple_ansi}$ '

  first = True
  while True:
    try:
      user_input = input(f'{blue_ansi}Type a message to begin the conversation…{reset_ansi}\n{prompt}' if first else prompt)
    except (KeyboardInterrupt, EOFError):
      sys.exit(0)
    print(reset_ansi, end='')

    first = False
    history += [Message(Participant.User, user_input)]
  
    history_str: str = ''.join([
      f"<|im_start|>{participant.value}\n{message}<|im_end|>"
      for participant, message in history
    ])
    chat_to_complete = f"{history_str}<|im_start|>{Participant.Assistant.value}\n"

    tokenized_prompts: TokenizerOutput = tokenizer([chat_to_complete], return_tensors='pt', truncation=True)
    tokenized_prompts: TokenizerOutput = tokenized_prompts.to(model.device)
    
    print(green_ansi, end='', flush=True)

    response = ''
    def on_text(message: str, stream_end = False):
      nonlocal response
      response += message
      print(message, end='', flush=True)

    streamer = CallbackTextIteratorStreamer(tokenizer, callback=on_text, skip_prompt=True, skip_special_tokens=True)

    try:
      prediction: LongTensor = model.generate(
        **tokenized_prompts,
        generation_config=generation_config,
        do_sample=generation_config.temperature > 0.,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
      )
      # if you wanted to see the result, you can do so like this:
      #   decode: List[str] = tokenizer.batch_decode(prediction, skip_special_tokens=True, clean_up_tokenization_spaces=True)
      # but we're already streaming it to the console via our callback
    except KeyboardInterrupt:
      pass

    # reset ANSI control sequence (plus line break)
    print(reset_ansi)

    # TODO: cull older history, otherwise context will just keep growing larger.
    #       ideally by measuring each message to work out the smallest cull possible.
    history += [Message(Participant.Assistant, response)]

if __name__ == "__main__":
  main()