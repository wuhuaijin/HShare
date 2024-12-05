import torch
import math

from channel import get_label_tensor
from sparse import fwd_sparse



def att(Q, K, V, Out, q_label, k_label, label_scores, channel, heavy_const, heavy_channel_num, label_mask, attn_mask, label_index=None):

    get_label_tensor(Q, channel, q_label, heavy_channel_num)

    tmp_scores = torch.matmul(q_label.view(Q.shape[0], 1, Q.shape[1], heavy_channel_num).transpose(1,2), k_label.view(Q.shape[0], K.shape[0] // Q.shape[0], K.shape[1], heavy_channel_num).transpose(1,2).transpose(2, 3)).view(Q.shape[0], K.shape[1], K.shape[0] // Q.shape[0])

    _, label_index = torch.topk(tmp_scores, heavy_const, dim=-1)

    fwd_sparse(Q, K, V, Out, label_index, attn_mask)

    return Out

class SharedAttn(torch.nn.Module):
    def __init__(self, token_shared_num, layer_shared_num, bs, global_const=8, local_const=64, num_layer=32, num_head=32):
        super().__init__()
        self._token_shared_num = token_shared_num # 4x
        self._layer_shared_num = num_layer - layer_shared_num # 4x
        self._label_bucket = []
        self._global_const = global_const
        self._local_const = local_const
        self._global_token_index = torch.tensor([i for i in range(0, self._global_const)] * bs * num_head, dtype=torch.int64, device='cuda').view(bs, num_head, -1)
        self._local_token_index = torch.tensor([i for i in range(0, self._local_const)] * bs * num_head, dtype=torch.int64, device='cuda').view(bs, num_head, -1)

        self._compiled_mask_func = None

    def _get_mask_func(self):
        if not self._compiled_mask_func:
            self._compiled_mask_func = torch.compile(self.get_stream_mask, mode='reduce-overhead')
        return self._compiled_mask_func
    
    def get_stream_mask(self, kv_seq_len):
        local_token_index = self._local_token_index + kv_seq_len - self._local_const
        numbers_to_cat = torch.cat((self._global_token_index, local_token_index), dim=-1)
        return numbers_to_cat
            
    def forward(self, Q, K, V, Out, q_label, k_label, label_scores, channel, heavy_const, heavy_channel_num, label_mask, attn_mask, token_id, layer_id):

        # Every token-group/layer_group start, we need to recompute the heavy label
        if token_id % self._token_shared_num == 0 and layer_id < self._layer_shared_num :


            get_label_tensor(Q, channel, q_label, heavy_channel_num)

            kv_seq_len, batch = K.shape[0] // Q.shape[0], Q.shape[0]
            q_head_num, k_head_num = Q.shape[1], K.shape[1]
            tmp_scores = torch.matmul(q_label.view(batch, 1, q_head_num, heavy_channel_num).transpose(1,2), k_label.view(batch, kv_seq_len, k_head_num, heavy_channel_num).transpose(1,2).transpose(2, 3)).view(batch, k_head_num, kv_seq_len)
            _, label_index = torch.topk(tmp_scores[..., self._global_const:kv_seq_len - self._local_const], heavy_const-self._local_const-self._global_const, dim=-1)
            label_index += self._global_const

            # compiled_stream_mask_func = self._get_mask_func()
            # stream_mask = compiled_stream_mask_func(kv_seq_len, q_head_num, self._local_token_index, self._local_const)
            stream_mask = self.get_stream_mask(kv_seq_len)
            label_index = torch.cat((label_index, stream_mask), dim=-1)

            if len(self._label_bucket) < self._layer_shared_num:
                self._label_bucket.append(label_index)
            else:
                self._label_bucket[layer_id] = label_index
        else:
            label_index = self._label_bucket[layer_id % self._layer_shared_num]

        fwd_sparse(Q, K, V, Out, label_index, attn_mask)

        return Out


def test_shared_att(B, N_CTX, H, D, HEAVY_CHANNEL_NUM, HEAVY_CONST):
    import time

    layer_num = 32
    shared_attn = SharedAttn(2, 8, B)

    print(f"B: {B}, N_CTX: {N_CTX}, H: {H}, D: {D}, HEAVY_CHANNEL_NUM: {HEAVY_CHANNEL_NUM}, HEAVY_CONST: {HEAVY_CONST}")

    dtype = torch.float16

    q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)

    channel = torch.zeros(H, HEAVY_CHANNEL_NUM, dtype=torch.int64, device='cuda')
    for h in range(H):
        channel[h] = torch.randperm(D, device='cuda')[:HEAVY_CHANNEL_NUM]

    out = torch.empty((B, H, D), dtype=dtype, device="cuda")

    q_label = torch.empty((B, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda")
    k_label = torch.empty((B * N_CTX, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda")

    label_scores = torch.empty((B, H, N_CTX), dtype=dtype, device="cuda")
    get_label_tensor(k, channel, k_label, HEAVY_CHANNEL_NUM)

    label_mask = torch.zeros((B, N_CTX), dtype=dtype, device="cuda")
    attn_mask = torch.zeros((B, HEAVY_CONST), dtype=dtype, device="cuda")

    # Warm up
    for _ in range(10):
        shared_attn(q, k, v, out, q_label, k_label, label_scores, channel, HEAVY_CONST, HEAVY_CHANNEL_NUM, label_mask, attn_mask, 0, 0)

    
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(run_iter):
        shared_attn(q, k, v, out, q_label, k_label, label_scores, channel, HEAVY_CONST, HEAVY_CHANNEL_NUM, label_mask, attn_mask, i // layer_num, i % layer_num)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))

    return (t2 - t1) / run_iter

def test_att(B, N_CTX, H, D, HEAVY_CHANNEL_NUM, HEAVY_CONST):
    import time

    print(f"B: {B}, N_CTX: {N_CTX}, H: {H}, D: {D}, HEAVY_CHANNEL_NUM: {HEAVY_CHANNEL_NUM}, HEAVY_CONST: {HEAVY_CONST}")

    dtype = torch.float16

    q = torch.empty((B, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B * N_CTX, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)

    channel = torch.zeros(H, HEAVY_CHANNEL_NUM, dtype=torch.int64, device='cuda')
    for h in range(H):
        channel[h] = torch.randperm(D, device='cuda')[:HEAVY_CHANNEL_NUM]

    out = torch.empty((B, H, D), dtype=dtype, device="cuda")

    q_label = torch.empty((B, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda")
    k_label = torch.empty((B * N_CTX, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda")

    label_scores = torch.empty((B, H, N_CTX), dtype=dtype, device="cuda")

    get_label_tensor(k, channel, k_label, HEAVY_CHANNEL_NUM)

    label_mask = torch.zeros((B, N_CTX), dtype=dtype, device="cuda")
    attn_mask = torch.zeros((B, HEAVY_CONST), dtype=dtype, device="cuda")

    index_matrix = torch.randint(0, 1025, (B, H, HEAVY_CONST), device="cuda").long()
    # Warm up
    for _ in range(10):
        att(q, k, v, out, q_label, k_label, label_scores, channel, HEAVY_CONST, HEAVY_CHANNEL_NUM, label_mask, attn_mask, index_matrix)

    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        att(q, k, v, out, q_label, k_label, label_scores, channel, HEAVY_CONST, HEAVY_CHANNEL_NUM, label_mask, attn_mask, index_matrix)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))

    return (t2 - t1) / run_iter


if __name__ == '__main__':

    bszs = [8,16]
    ctxs = [1024,2048,4096]

    sparsity_level = 8
    h = 32
    d = 128


    times = []
    share_times = []

    for bsz in bszs:
        for ctx in ctxs:

            heavy_channel_num = d // sparsity_level
            heavy_const = ctx // sparsity_level
            times.append([bsz, ctx, test_att(bsz, ctx, h, d, heavy_channel_num, heavy_const)])
            share_times.append([bsz, ctx, test_shared_att(bsz, ctx, h, d, heavy_channel_num, heavy_const)])

    print(times)
    print(share_times)
