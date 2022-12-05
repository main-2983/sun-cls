import torch
import torch.nn as nn

from mmcls.models.utils import MultiheadAttention


class SingleHeadAttention(nn.Module):
    def __init__(self,
                 input_dims,
                 embed_dims):
        super(SingleHeadAttention, self).__init__()

        self.input_dims = input_dims
        self.embed_dims = embed_dims

        self.q_matrix = nn.Linear(in_features=input_dims,
                                  out_features=embed_dims)
        self.k_matrix = nn.Linear(in_features=input_dims,
                                  out_features=embed_dims)
        self.v_matrix = nn.Linear(in_features=input_dims,
                                  out_features=embed_dims)
        self.scale    = self.embed_dims ** (-0.5)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward function of Single Head Attention
        Args:
            x: input tensor of shape (B, L, C) with:
                - B being batch size
                - L being num tokens
                - C being num representation channels of single token
        """
        # register shape of input x
        B, L, _ = x.shape
        # generate Q, K, V, each of shape (B, L, embed_dims)
        q = self.q_matrix(x)
        # print(f"Slow Q: {self.q_matrix.weight}")
        print(f"Slow Q val: {q}")
        k = self.k_matrix(x)
        # print(f"Slow K: {self.k_matrix.weight}")
        v = self.v_matrix(x)
        # print(f"Slow V: {self.v_matrix.weight}")

        # perform Q.K^{T}, get matrix of shape (B, L, L)
        q_times_k = torch.matmul(q, k.mT)
        # perform softmax on channel dims
        attention = self.softmax(q_times_k * self.scale)
        # matmul attention with v to create output of shape (B, L, embed_dims)
        out = torch.matmul(attention, v)

        return out


class FastSingleHeadAttn(nn.Module):
    def __init__(self,
                 input_dims,
                 embed_dims):
        super(FastSingleHeadAttn, self).__init__()
        self.input_dims = input_dims
        self.embed_dims = embed_dims

        self.qkv_matrix = nn.Linear(in_features=input_dims,
                                    out_features=embed_dims * 3)
        self.scale = self.embed_dims ** (-0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, L, _ = x.shape
        # this will have shape (B, L, embed_dím * 3)
        qkv = self.qkv_matrix(x)
        # print(f"Fast QKV before view: {qkv}")
        qkv = qkv.view(B, L, 3, self.embed_dims)
        # to (3, B, L, embed_dims)
        qkv = qkv.permute(2, 0, 1, 3)
        # each of shape (B, L, embed_dims)
        q, k, v = qkv[0], qkv[1], qkv[2]
        print(f"Fast Q val: {q}")
        split_weight = torch.tensor_split(self.qkv_matrix.weight, 3)
        # print(f"Fast Q: {split_weight[0]}")
        # print(f"Fast K: {split_weight[1]}")
        # print(f"Fast V: {split_weight[2]}")

        # perform Q.K^{T}, get matrix of shape (B, L, L)
        q_times_k = torch.matmul(q, k.mT)
        # perform softmax on channel dims
        attention = self.softmax(q_times_k * self.scale)
        # matmul attention with v to create output of shape (B, L, embed_dims)
        out = torch.matmul(attention, v)

        return out


if __name__ == '__main__':
    torch.manual_seed(0)
    input = torch.rand((1, 2*2, 3))
    # init FSHSA
    fast_SHSA = FastSingleHeadAttn(input_dims=3,
                                   embed_dims=4)
    # grab qkv_matrix weight
    qkv_weight = fast_SHSA.qkv_matrix.weight
    #print(qkv_weight.shape)
    # split qkv_matrix weight of FSHSA
    q_weight, k_weight, v_weight =  torch.tensor_split(qkv_weight, 3)
    # init SHSA
    shsa = SingleHeadAttention(input_dims=3,
                               embed_dims=4)
    # set q_matrix weight
    shsa.q_matrix.weight = nn.Parameter(q_weight)
    # do the same for k and v weight
    shsa.k_matrix.weight = nn.Parameter(k_weight)
    shsa.v_matrix.weight = nn.Parameter(v_weight)

    # perform forward to check if identical
    fast_out = fast_SHSA(input)
    slow_out = shsa(input)

    print(fast_out.shape)
    print(slow_out.shape)
    # print(fast_out - slow_out)