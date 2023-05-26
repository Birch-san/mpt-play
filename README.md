# MPT-Play

<img src="https://github.com/Birch-san/mpt-play/assets/6141784/da9cbf86-bc1c-499d-9d9e-7eff3a210696" alt="Type a message to begin the conversation…/$ Hey, my name is Reimu. I maintain the barrier to Gensokyo./Ah, the legendary Reimu. I'm sure your barrier is the most powerful in Gensokyo. That's a lot of responsibility for someone your age!/$ it's not that much really /Anyway, I was thinking that I might try to visit Gensokyo someday. Do you think it will be okay if I just show up one day?/$ Probably fine./But there could be consequences to my actions. I don't know exactly how the barrier works./$ The barrier only affects living things and their creations./I'm glad to hear that. I'll try to plan my visit carefully then.">

Python script to demonstrate how to invoke models such as MPT-7B-Chat from the command-line.

## Setup

All instructions are written assuming your command-line shell is bash.

Clone repository:

```bash
git clone https://github.com/Birch-san/mpt-play.git
cd mpt-play
```

### Create + activate a new virtual environment

This is to avoid interfering with your current Python environment (other Python scripts on your computer might not appreciate it if you update a bunch of packages they were relying on).

Follow the instructions for virtualenv, or conda, or neither (if you don't care what happens to other Python scripts on your computer).

#### Using `venv`

**Create environment**:

```bash
. ./venv/bin/activate
pip install --upgrade pip
```

**Activate environment**:

```bash
. ./venv/bin/activate
```

**(First-time) update environment's `pip`**:

```bash
pip install --upgrade pip
```

#### Using `conda`

**Download [conda](https://www.anaconda.com/products/distribution).**

_Skip this step if you already have conda._

**Install conda**:

_Skip this step if you already have conda._

Assuming you're using a `bash` shell:

```bash
# Linux installs Anaconda via this shell script. Mac installs by running a .pkg installer.
bash Anaconda-latest-Linux-x86_64.sh
# this step probably works on both Linux and Mac.
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda config --set auto_activate_base false
conda init
```

**Create environment**:

```bash
conda create -n p311-mpt python=3.11
```

**Activate environment**:

```bash
conda activate p311-mpt
```

### Install package dependencies

**Ensure you have activated the environment you created above.**

(Optional) treat yourself to latest nightly of PyTorch, with support for Python 3.11 and CUDA 12.1:

```bash
# CUDA
pip install --upgrade --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu121
# Mac
pip install --upgrade --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run:

From root of repository:

```bash
python -m scripts.chat_play --trust_remote_code --bf16
```

On Mac you'll need to disable bfloat16. `PYTORCH_ENABLE_MPS_FALLBACK=1` is not necessary in current PyTorch nightly, but there's no harm keeping it (and it may help on older PyTorch):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.chat_play --trust_remote_code
```

## License

This repository is itself MIT-licensed.

Includes MIT-licensed code copied from Artidoro Pagnoni's [qlora](https://github.com/artidoro/qlora), and [Apache-licensed](licenses/MosaicML-mpt-7b-chat-hf-space.Apache.LICENSE.txt) code copied from MosaicML's [mpt-7b-chat](https://huggingface.co/spaces/mosaicml/mpt-7b-chat/blob/main/app.py) Huggingface Space.