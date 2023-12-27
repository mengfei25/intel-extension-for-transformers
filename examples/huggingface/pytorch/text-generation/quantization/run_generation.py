import argparse
import os
import re
import time
import json
import torch
import logging
from transformers import AutoConfig, AutoTokenizer
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM,
    AutoModel,
)
from transformers.utils import check_min_version
from optimum.intel.generation.modeling import TSModelForCausalLM
from intel_extension_for_transformers.transformers import (
    MixedPrecisionConfig,
    WeightOnlyQuantConfig,
    SmoothQuantConfig,
    BitsAndBytesConfig,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None)
parser.add_argument("--bf16", action="store_true")
parser.add_argument(
    "--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
)
parser.add_argument(
    "--max_new_tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument(
    "--restore",
    action="store_true",
    help="restore ipex quantized model from output_dir/best_configure.json",
)
parser.add_argument(
    "--peft_model_id", type=str, default=None, help="model_name_or_path of peft model"
)
# ============Benchmark configs==============
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--iters", default=100, type=int, help="num iter")
parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
# ============Accuracy configs==============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=56, type=int, help="batch size num.")
parser.add_argument(
    "--save_accuracy_path", default=None, help="Save accuracy results path."
)
parser.add_argument(
    "--tasks",
    nargs="+",
    default=["lambada_openai", "hellaswag","winogrande","piqa","hendrycksTest-abstract_algebra","hendrycksTest-anatomy","hendrycksTest-astronomy","hendrycksTest-business_ethics","hendrycksTest-clinical_knowledge","hendrycksTest-college_biology","hendrycksTest-college_chemistry","hendrycksTest-college_computer_science","hendrycksTest-college_mathematics","hendrycksTest-college_medicine","hendrycksTest-college_physics","hendrycksTest-computer_security","hendrycksTest-conceptual_physics","hendrycksTest-econometrics","hendrycksTest-electrical_engineering","hendrycksTest-elementary_mathematics","hendrycksTest-formal_logic","hendrycksTest-global_facts","hendrycksTest-high_school_biology","hendrycksTest-high_school_chemistry","hendrycksTest-high_school_computer_science","hendrycksTest-high_school_european_history","hendrycksTest-high_school_geography","hendrycksTest-high_school_government_and_politics","hendrycksTest-high_school_macroeconomics","hendrycksTest-high_school_mathematics","hendrycksTest-high_school_microeconomics","hendrycksTest-high_school_physics","hendrycksTest-high_school_psychology","hendrycksTest-high_school_statistics","hendrycksTest-high_school_us_history","hendrycksTest-high_school_world_history","hendrycksTest-human_aging","hendrycksTest-human_sexuality","hendrycksTest-international_law","hendrycksTest-jurisprudence","hendrycksTest-logical_fallacies","hendrycksTest-machine_learning","hendrycksTest-management","hendrycksTest-marketing","hendrycksTest-medical_genetics","hendrycksTest-miscellaneous","hendrycksTest-moral_disputes","hendrycksTest-moral_scenarios","hendrycksTest-nutrition","hendrycksTest-philosophy","hendrycksTest-prehistory","hendrycksTest-professional_accounting","hendrycksTest-professional_law","hendrycksTest-professional_medicine","hendrycksTest-professional_psychology","hendrycksTest-public_relations","hendrycksTest-security_studies","hendrycksTest-sociology","hendrycksTest-us_foreign_policy","hendrycksTest-virology","hendrycksTest-world_religions","truthfulqa_mc","arc_challenge","wikitext"],
    type=str,
    help="tasks list for accuracy validation",
)
# ============MixedPrecision configs==============
parser.add_argument("--mixed_precision", action="store_true")
# ============SmoothQuant configs==============
parser.add_argument("--sq", action="store_true")
parser.add_argument("--calib_iters", default=100, type=int, help="Calibration iters.")
parser.add_argument(
    "--calib_padding", action="store_true", help="Calibration dataset do padding."
)
parser.add_argument(
    "--calib_pad_val", default=1, type=int, help="Calibration dataset padding value."
)
parser.add_argument(
    "--calib_len",
    default=512,
    type=int,
    help="Calibration dataset max or padding max length.",
)
parser.add_argument(
    "--recipes", type=str, help="A dictionary as a string, recipes for smoothquant."
)
parser.add_argument("--alpha", default="0.5", help="Smooth quant parameter.")
parser.add_argument(
    "--fallback_add", action="store_true", help="Whether to fallback add ops to FP32"
)
# ============WeightOnlyQuant configs===============
parser.add_argument("--woq", action="store_true")
parser.add_argument(
    "--woq_algo",
    default="RTN",
    choices=["RTN", "AWQ", "TEQ"],
    help="Weight-only parameter.",
)
parser.add_argument(
    "--woq_weight_dtype",
    type=str,
    default="int8",
    choices=[
        "int8",
        "int4_clip",
        "int4_fullrange",
        "fp4_e2m1_bnb",
        "fp4_e2m1",
        "nf4",
        "fp8_e5m2",
        "fp8_e4m3",
    ],
)
parser.add_argument(
    "--woq_scale_dtype",
    type=str,
    default="fp32",
    choices=["fp32", "fp8"],
)
parser.add_argument(
    "--woq_compute_dtype",
    type=str,
    default="fp32",
    choices=["fp32", "bf16", "int8"],
)
parser.add_argument("--woq_group_size", type=int, default=32)
parser.add_argument("--woq_scheme", default="sym")
# ============BitsAndBytes configs==============
parser.add_argument("--bitsandbytes", action="store_true")
# ============AutoModel parameters==============
parser.add_argument("--load_in_4bit", type=bool, default=False)
parser.add_argument("--load_in_8bit", type=bool, default=False)
parser.add_argument("--_commit_hash", default="main", type=str)
parser.add_argument("--trust_remote_code", default=False)
parser.add_argument("--use_llm_runtime", action="store_true")
# =======================================
args = parser.parse_args()

# transformers version >= 4.32.0 contained the mpt modeling definition.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpt/modeling_mpt.py
# 4.31.0 for ipex.optimize_transformers
check_min_version("4.31.0")

# get model config
if args.peft_model_id:
    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(args.peft_model_id)
    if args.model is None:
        args.model = peft_config.base_model_name_or_path
        print("we will use peft base_model_name_or_path to get tokenizer.")

config = AutoConfig.from_pretrained(
    args.model,
    torchscript=True
    if (
        args.sq
        or args.woq_algo in ["AWQ", "TEQ"]
        or (args.int8 or args.int8_bf16_mixed)
        or args.bf16
    )
    else False,  # torchscript will force `return_dict=False` to avoid jit errors
    use_cache=True,  # to use kv cache.
    trust_remote_code=args.trust_remote_code,
    _commit_hash=args._commit_hash,
)

# chatglm
if config.model_type == "chatglm":
    AutoModelForCausalLM = AutoModel
# tokenizer
if config.model_type == "llama":
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(args.model)
else:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

# use peft
args.model = args.peft_model_id if args.peft_model_id is not None else args.model

# Generation
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

# mp/sq/woq/bitsandbytes config setting
quantization_config = None
if args.mixed_precision:
    quantization_config = MixedPrecisionConfig(dtype="bfloat16")  # default is bfloat16
elif args.sq:
    if re.search("gptj", config.model_type) or re.search("gpt_neox", config.model_type):
        op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        }
    elif re.search("mpt", config.model_type):
        op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
            "<built-in function linear>": {
                "weight": {"dtype": ["fp32"]},
                "activation": {"dtype": ["fp32"]},
            },
        }
    elif re.search("mistral", config.model_type) or re.search(
        "baichuan", config.model_type
    ):
        op_type_dict = {".*": {"activation": {"algorithm": "minmax"}}}
    else:
        op_type_dict = {}
    if args.fallback_add:
        op_type_dict["add"] = {
            "weight": {"dtype": ["fp32"]},
            "activation": {"dtype": ["fp32"]},
        }
    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    if args.recipes:
        try:
            import ast

            recipes = ast.literal_eval(args.recipes)
            print("Parsed recipes dictionary:", recipes)
        except ValueError as e:
            print("Error parsing recipes dictionary:", e)
    else:
        recipes = {
            "smooth_quant": True,
            "smooth_quant_args": {
                "alpha": args.alpha if args.alpha == "auto" else float(args.alpha)
            },
        }
    quantization_config = SmoothQuantConfig(
        tokenizer=tokenizer,  # either two of one, tokenizer or calib_func
        recipes=recipes,
        op_type_dict=op_type_dict,  # default is {}
        excluded_precisions=excluded_precisions,  # default is []
        num_beams=generate_kwargs["num_beams"],
        calib_iters=args.calib_iters,
        calib_padding=args.calib_padding,
        calib_len=args.calib_len,
        calib_pad_val=args.calib_pad_val,
    )
elif args.woq:
    quantization_config = WeightOnlyQuantConfig(
        compute_dtype=args.woq_compute_dtype,
        scale_dtype=args.woq_scale_dtype,
        weight_dtype=args.woq_weight_dtype,
        scheme=args.woq_scheme,
        group_size=args.woq_group_size,
    )  # default is A32W4G32
# bitsandbytes
elif args.bitsandbytes:
    # GPU device is need for `load_in_4bit` and `load_in_8bit`.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

# get optimized model
if quantization_config is not None:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
        _commit_hash=args._commit_hash,
        use_llm_runtime=args.use_llm_runtime,

    )
elif args.load_in_4bit or args.load_in_8bit:
    # CPU device usage is provided by intel-extension-for-transformers.
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        _commit_hash=args._commit_hash,
        use_llm_runtime=args.use_llm_runtime,
    )
elif (not args.int8 and not args.int8_bf16_mixed) or args.restore:
    if args.peft_model_id is not None:
        user_model = AutoModelForCausalLM.from_pretrained(
            args.peft_model_id,
            trust_remote_code=args.trust_remote_code,
            _commit_hash=args._commit_hash,
            use_llm_runtime=args.use_llm_runtime,
        )
    else:
        user_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=args.trust_remote_code,
            _commit_hash=args._commit_hash,
            use_llm_runtime=args.use_llm_runtime,
        )

# save model
if args.output_dir:
    if args.sq:
        config.save_pretrained(args.output_dir)
        user_model.save(args.output_dir)
    elif args.mixed_precision:
        user_model.config.save_pretrained(args.output_dir)
        torch.save(
            user_model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin")
        )

# int8 model loading
if args.int8 or args.int8_bf16_mixed:
    # TorchScript model don't attribute generate method, the wrapper is provided.
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_transformers.llm.evaluation.models import (
        TSModelCausalLMForITREX,
    )

    if args.restore:
        from intel_extension_for_transformers.transformers.utils.utility import (
            recover_model_from_json,
        )

        user_model = recover_model_from_json(
            user_model,
            os.path.join(args.output_dir, "best_configure.json"),
            args.trust_remote_code,
        )
        user_model = TSModelCausalLMForITREX(user_model, config=config)
    else:
        user_model = TSModelCausalLMForITREX.from_pretrained(
            args.output_dir,
            file_name="best_model.pt",
            trust_remote_code=args.trust_remote_code,
        )

if args.benchmark:
    prompt = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    # start
    total_time = 0.0
    num_iter = args.iters
    num_warmup = args.num_warmup
    total_token_num = 0
    eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode(), torch.no_grad():
        for i in range(num_iter):
            tic = time.time()
            if hasattr(tokenizer, "build_chat_input"):
                input_ids = tokenizer.build_chat_input(prompt)["input_ids"]
                input_ids = input_ids.repeat(args.batch_size, 1)
                eos_token_id = [
                    tokenizer.eos_token_id,
                    tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>"),
                ]
            elif hasattr(tokenizer, "build_prompt"):
                build_prompt = tokenizer.build_prompt(prompt)
                input_ids = tokenizer(
                    [build_prompt] * args.batch_size, return_tensors="pt"
                ).input_ids
            else:
                input_ids = tokenizer(
                    [prompt] * args.batch_size, return_tensors="pt"
                ).input_ids
            gen_ids = user_model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                **generate_kwargs,
                eos_token_id=eos_token_id
            )
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            toc = time.time()
            # please check the gen_ids if include input_ids.
            input_tokens_num = input_ids.numel()
            output_tokens_num = gen_ids.numel() - input_tokens_num
            print(gen_text, flush=True)
            if i >= num_warmup:
                total_time += toc - tic
                total_token_num += output_tokens_num

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / total_token_num
    print("Inference latency: %.3f sec." % latency)
    throughput = total_token_num / total_time
    print("Throughput: {} samples/sec".format(throughput))

if args.accuracy:
    args.model = (
        peft_config.base_model_name_or_path if args.peft_model_id else args.model
    )
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    if args.bf16:
        try:
            import intel_extension_for_pytorch as ipex
            user_model = ipex.optimize_transformers(user_model.eval(), dtype=torch.bfloat16, inplace=True, deployment_mode=True)
        except:
            from intel_extension_for_transformers.transformers.utils.utility import get_example_inputs
            from intel_extension_for_transformers.llm.evaluation.models import TSModelCausalLMForITREX
            example_inputs = get_example_inputs(user_model.config, tokenizer=tokenizer)
            with torch.no_grad(), torch.cpu.amp.autocast():
                user_model = torch.jit.trace(user_model, example_kwarg_inputs=example_inputs, check_trace=False, strict=False)
                user_model = torch.jit.trace(user_model.eval())
            user_model = TSModelCausalLMForITREX(user_model, config)

    with torch.autocast('cpu', enabled=args.bf16, dtype=torch.bfloat16 if args.bf16 else None):
        results = evaluate(
            model="hf-causal",
            model_args="pretrained="
            + args.model
            + ",tokenizer="
            + args.model
            + ",dtype=float32"
            + ",_commit_hash="
            + args._commit_hash
            + ",trust_remote_code="
            + str(args.trust_remote_code),
            user_model=user_model,
            batch_size=args.batch_size,
            tasks=args.tasks,
        )
    dumped = json.dumps(results, indent=2)
    if args.save_accuracy_path:
        with open(args.save_accuracy_path, "w") as f:
            f.write(dumped)

