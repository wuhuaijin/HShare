# HShare

## Install

1. Clone this repo and setup the environment

```
git clone xxx/HShare.git
cd HShare
conda create -n HShare python=3.9 -y
conda activate HShare
pip install -r requirement.txt
```

2. Install the lm-eval framework

```
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Accuracy Evaluation

You can change the sharing ratio and the number of critical token through `config.json`

```
"heavy_const": 88,
"sink_const": 8,
"local_const": 32,
"share_query_const": 2,
"share_head_const": 16,
"share_layer_const": 16
```

**GSM8K, COQA accuracy**

1. add the following to `lm-evaluation-harness/lm_eval/models/huggingface.py`
```
from evaluation.replace_for_lmeval import replace_llama_modules, replace_mistral_modules
replace_llama_modules()
replace_mistral_modules()
```

2. run the following
```
lm_eval --model hf --model_args pretrained=your_model_path --tasks gsm8k,coqa --device cuda:1 --batch_size 8
```



**Longbench**

1. run the following to generate results

```
python pred.py --model llama2-7b-chat-4k --dataset multi_news,multifieldqa_en
```

2. run the following to evaluate results

```
python eval.py --model llama2-7b-chat-4k --method HShare
```

## Speed Evaluation

1. Prepare weight

```
python convert_hf.py --checkpoint_dir you_model_path --model_name meta-llama/Llama-2-7b-chat-hf
```

2. Attention Operator Speedup

```
python triton_kernels/attention_HShare.py 
```

3. End-to-End Inference Speedup

```
python generate.py --checkpoint_path you_model_path.pth --max_new_tokens 2048 --batch_size 16
```



## Citation
If you use our code or method in your work, please consider citing the following:
```

```

## Acknowledgement
We appreciate the following works for their valuable code and data:

https://github.com/THUDM/LongBench

https://github.com/EleutherAI/lm-evaluation-harness

https://github.com/andy-yang-1/DoubleSparse
