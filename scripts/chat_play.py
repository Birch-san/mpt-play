from dataclasses import dataclass, field
from typing import Optional
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  BitsAndBytesConfig,
  GenerationConfig,
  HfArgumentParser,
  set_seed,
)
from argparse import Namespace
import torch
from src.device_map import device_map
import logging

logger = logging.getLogger(__name__)

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

@dataclass
class GenerationArguments:
  # For more hyperparameters check:
  # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
  seed: Optional[int] = field(
    default=64,
    metadata={"help": "Random seed, for deterministic generation."}
  )

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
  cuda_avail = torch.cuda.is_available()
  compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
  load_in_4bit = args.bits == 4
  load_in_8bit = args.bits == 8

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
    logger.warn("You don't have CUDA, so we have turned off quantization. If happen to be on a Mac: you probably have enough unified memory to run in fp16 anyway…")

  if compute_dtype == torch.float16 and cuda_avail and torch.cuda.is_bf16_supported():
    print("Your GPU supports bfloat16; you may want to try it with --bf16 (note: I'm not sure how important this is for inference, but it's certainly preferred when training with 4-bit quantization.)")

  model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    device_map=device_map,
    quantization_config=quantization_config,
    torch_dtype=compute_dtype,
    trust_remote_code=args.trust_remote_code,
  ).eval()
  model.config.torch_dtype=compute_dtype

  return model

def main():
  hfparser = HfArgumentParser((ModelArguments, GenerationArguments))
  model_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
  if extra_args:
    raise f"Received unsupported command-line args: {extra_args}"
  generation_config = GenerationConfig(**vars(generation_args))
  args = Namespace(**vars(model_args))
  model: AutoModelForCausalLM = get_model(model_args)
  set_seed(generation_args.seed)
  ## TODO: inference the model

if __name__ == "__main__":
  main()