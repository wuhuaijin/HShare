import torch
import triton
import triton.language as tl
@triton.jit
def mask_kernel(
    topk_mask_ptr,
    global_mask_ptr,
    local_mask_ptr,
    output_ptr,
    kv_seq_len,
    topk_len,
    global_len,
    local_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    local_mask = tl.load(local_mask_ptr + offsets, mask=(offsets<local_len))
    local_mask += kv_seq_len - local_len
    global_mask = tl.load(global_mask_ptr + offsets, mask=offsets<global_len)
    topk_mask = tl.load(topk_mask_ptr + pid * topk_len + offsets, mask=offsets<topk_len)
    output_stride = topk_len + local_len + global_len
    tl.store(output_ptr + pid * output_stride + offsets, global_mask, mask=offsets < global_len)
    tl.store(output_ptr + pid * output_stride + global_len + offsets, topk_mask, mask=offsets<topk_len)
    tl.store(output_ptr + pid * output_stride + global_len + topk_len + offsets, local_mask, mask=offsets<local_len)
def make_mask(
    topk_mask,
    global_mask,
    local_mask,
    batch_size,
    q_head_num,
    kv_seq_len,
):
    topk_len = topk_mask.shape[-1]
    global_len = global_mask.shape[-1]
    local_len = local_mask.shape[-1]
    mask_len = topk_len + global_len + local_len
    output = torch.empty((batch_size, q_head_num, mask_len), dtype=torch.int64, device='cuda')
    grid = lambda meta: (batch_size * q_head_num, )
    mask_kernel[grid](
        topk_mask,
        global_mask,
        local_mask,
        output,
        kv_seq_len,
        topk_len,
        global_len,
        local_len,
        BLOCK_SIZE=128,
    )
    print(output)
    print(output.dtype)
    return output
batch_size = 1
q_head_num = 1
kv_seq_len = 1024
tok_mask = torch.ones(batch_size, q_head_num, 56, dtype=torch.int64, device='cuda')
local_mask = torch.tensor([i for i in range(64)], dtype=torch.int64, device='cuda')
global_mask = torch.tensor([i for i in range(8)], dtype=torch.int64, device='cuda')
make_mask(tok_mask, global_mask, local_mask, batch_size, q_head_num, kv_seq_len)