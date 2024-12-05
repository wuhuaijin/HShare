from transformers.models import llama
from transformers.models import mistral

from models.HShare_llama import LlamaForCausalLM, LlamaAttention
from models.HShare_mistral import MistralForCausalLM, MistralAttention

def replace_llama_modules():
    llama.LlamaForCausalLM = LlamaForCausalLM
    llama.modeling_llama.LlamaAttention = LlamaAttention

def replace_mistral_modules():
    mistral.MistralForCausalLM = MistralForCausalLM
    mistral.modeling_mistral.MistralAttention = MistralAttention

