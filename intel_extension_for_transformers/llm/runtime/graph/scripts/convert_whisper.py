#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Convert Hugging Face fine-tuned models to ggml format
#
# Usage:
#
#   git clone https://github.com/openai/whisper
#   git clone https://github.com/ggerganov/whisper.cpp
#   git clone https://huggingface.co/openai/whisper-medium
#
#   python3 ./whisper.cpp/models/convert-h5-to-ggml.py ./whisper-medium/ ./whisper .
#
# This script is similar to "convert-pt-to-ggml.py"
#
# For more info:
#
#   https://github.com/ggerganov/whisper.cpp/issues/157
#

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np
from pathlib import Path
import argparse
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypeVar,
                    Union)

from transformers import WhisperForConditionalGeneration

conv_map = {
    'self_attn.k_proj': 'attn.key',
    'self_attn.q_proj': 'attn.query',
    'self_attn.v_proj': 'attn.value',
    'self_attn.out_proj': 'attn.out',
    'self_attn_layer_norm': 'attn_ln',
    'encoder_attn.q_proj': 'cross_attn.query',
    'encoder_attn.v_proj': 'cross_attn.value',
    'encoder_attn.out_proj': 'cross_attn.out',
    'encoder_attn_layer_norm': 'cross_attn_ln',
    'fc1': 'mlp.0',
    'fc2': 'mlp.2',
    'final_layer_norm': 'mlp_ln',
    'encoder.layer_norm.bias': 'encoder.ln_post.bias',
    'encoder.layer_norm.weight': 'encoder.ln_post.weight',
    'encoder.embed_positions.weight': 'encoder.positional_embedding',
    'decoder.layer_norm.bias': 'decoder.ln.bias',
    'decoder.layer_norm.weight': 'decoder.ln.weight',
    'decoder.embed_positions.weight': 'decoder.positional_embedding',
    'decoder.embed_tokens.weight': 'decoder.token_embedding.weight',
    'proj_out.weight': 'decoder.proj.weight',
}


# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a model to a NE compatible file")
    parser.add_argument("--outtype",
                        choices=["f32", "f16"],
                        default="fp32",
                        help="output format (default: based on input)")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument("model", type=Path, help="directory containing model file")
    args = parser.parse_args(args_in)
    dir_model = args.model
    dir_out = args.outfile
    out_type = args.outtype


    encoder = json.load((dir_model / "vocab.json").open("r", encoding="utf8"))
    encoder_added = json.load((dir_model / "added_tokens.json").open("r", encoding="utf8"))
    hparams = json.load((dir_model / "config.json").open("r", encoding="utf8"))

    model = WhisperForConditionalGeneration.from_pretrained(dir_model)

    #code.interact(local=locals())
    path = os.getcwd()
    path = path+'/whisper'
    if os.path.exists(path)== False:
        os.system('git clone https://github.com/openai/whisper.git')
    n_mels = hparams["num_mel_bins"]
    mel_path = path+'/whisper/assets/mel_filters.npz'
    with np.load(mel_path) as f:
        filters = torch.from_numpy(f[f"mel_{n_mels}"])

    dir_tokenizer = dir_model

    fname_out = dir_out

    tokens = json.load(open(dir_tokenizer / "vocab.json", "r", encoding="utf8"))

    # Default use 16-bit
    
    use_f16 = True

    fout = open(fname_out, "wb")

    fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["max_source_positions"]))
    fout.write(struct.pack("i", hparams["d_model"]))
    fout.write(struct.pack("i", hparams["encoder_attention_heads"]))
    fout.write(struct.pack("i", hparams["encoder_layers"]))
    fout.write(struct.pack("i", hparams["max_length"]))
    fout.write(struct.pack("i", hparams["d_model"]))
    fout.write(struct.pack("i", hparams["decoder_attention_heads"]))
    fout.write(struct.pack("i", hparams["decoder_layers"]))
    fout.write(struct.pack("i", hparams["num_mel_bins"]))
    fout.write(struct.pack("i", use_f16))

    fout.write(struct.pack("i", filters.shape[0]))
    fout.write(struct.pack("i", filters.shape[1]))
    for i in range(filters.shape[0]):
        for j in range(filters.shape[1]):
            fout.write(struct.pack("f", filters[i][j]))

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    fout.write(struct.pack("i", len(tokens)))

    tokens = sorted(tokens.items(), key=lambda x: x[1])
    for key in tokens:
        text = bytearray([byte_decoder[c] for c in key[0]])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

    list_vars = model.state_dict()
    for name in list_vars.keys():
        # this seems to not be used
        # ref: https://github.com/huggingface/transformers/blob/9a5b84a0076a04fe9596da72e8668069d4f09ea0/src
        #      /transformers/models/whisper/modeling_whisper.py#L1099-L1106
        if name == "proj_out.weight":
            print('Skipping', name)
            continue

        src = name

        nn = name
        if name != "proj_out.weight":
            nn = nn.split(".")[1:]
        else:
            nn = nn.split(".")

        if nn[1] == "layers":
            nn[1] = "blocks"
            if ".".join(nn[3:-1]) == "encoder_attn.k_proj":
                mapped = "attn.key" if nn[0] == "encoder" else "cross_attn.key"
            else:
                mapped = conv_map[".".join(nn[3:-1])]
            name = ".".join(nn[:3] + [mapped] + nn[-1:])
        else:
            name = ".".join(nn)
            name = conv_map[name] if name in conv_map else name

        print(src, ' -> ', name)
        data = list_vars[src].squeeze().numpy()
        data = data.astype(np.float16)

        # reshape conv bias from [n] to [n, 1]
        if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
            data = data.reshape(data.shape[0], 1)
            print("  Reshaped variable: ", name, " to shape: ", data.shape)

        n_dims = len(data.shape)
        print(name, n_dims, data.shape)

        # looks like the whisper models are in f16 by default
        # so we need to convert the small tensors to f32 until we fully support f16 in ggml
        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype = 1
        if use_f16:
            if n_dims < 2 or \
                    name == "encoder.conv1.bias"   or \
                    name == "encoder.conv2.bias"   or \
                    name == "encoder.positional_embedding" or \
                    name == "decoder.positional_embedding":
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype = 0
        else:
            data = data.astype(np.float32)
            ftype = 0

        # header
        str_ = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str_), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str_)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: ", fname_out)
    print("")


   

if __name__ == "__main__":
    main()
