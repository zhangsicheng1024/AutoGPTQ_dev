import os
import sys
sys.path.append("/workspace/v-leiwang3/lowbit_workspace/AutoGPTQ_nf4")

from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import numpy as np
import torch
import torch.nn as nn
import time
from logging import getLogger

enable_quant = True
export_nnfusion = True
# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = "save/opt125m_nf4"

# pretrained_model_dir = "meta-llama/Llama-2-7b-hf"
# quantized_model_dir = "save/llama2-7b_nf4_g128"

# pretrained_model_dir = "meta-llama/Llama-2-13b-hf"
# quantized_model_dir = "save/llama2-13b_nf4_g128"

pretrained_model_dir = "/workspace/v-leiwang3/lowbit_workspace/AutoGPTQ_nf4/models/Llama-2-70b-hf"
quantized_model_dir = "save/llama2-70b_nf4_g128"


def main():
    logger = getLogger(__name__)
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    # traindataset,testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)
    if enable_quant:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
        examples = [
            tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]
        quantize_config = BaseQuantizeConfig(
            bits=4,  # quantize model to 4-bit
            format='nf', # quantize model to int / nf / fp
            group_size=128,  # it is recommended to set the value to 128
            desc_act=False,  # desc_act and group size only works on triton
        )

        # load un-quantized model, the model will always be force loaded into cpu
        model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, cache_dir='/gptq_hub')
        # model.model.model.decoder.layers = model.model.model.decoder.layers[:1]

        # quantize model
        time_start = time.time()
        model.quantize(examples, use_triton=False)
        logger.info('quant time: %ds' % (time.time() - time_start))

        model.save_quantized(quantized_model_dir)

    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, 
        inject_fused_attention=False, inject_fused_mlp=False
    )

    # export 2 onnx
    batch_size = 1
    seq_length = 1
    input_shape = (batch_size, seq_length)
    onnx_name = f"qmodel_b{batch_size}s{seq_length}.onnx"
    output_path = os.path.join(quantized_model_dir, onnx_name)
    input_ids = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
    attention_mask = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
    
    if not export_nnfusion:
        start = time.time()
        for i in range(100):
            outputs = model(input_ids=input_ids)
        end = time.time()
        print("time", end - start)
    else:
        import onnx
        from onnxsim import simplify
        model = model.half().cuda()
        torch.onnx.export(      
            model,  
            input_ids,  
            f=output_path,  
            opset_version=11, 
        )  
        # load your predefined ONNX model
        model = onnx.load(output_path)
        # convert model
        model_simp, check = simplify(model)
        sim_output_path = os.path.join(quantized_model_dir, f"qmodel_b{batch_size}s{seq_length}_sim.onnx")
        onnx.save(model_simp, sim_output_path)
    # print(outputs.logits)
    

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
