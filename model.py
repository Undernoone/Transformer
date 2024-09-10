import torch
from torch import nn

num_hiddens = 256

class Embedding(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Embedding, self).__init__(*args, **kwargs)
        self.word_embedding = nn.Embedding(29, num_hiddens)
        self.pos_embedding = nn.Embedding(12, num_hiddens)
        self.pos_test = torch.arange(0, 12).reshape(1, 12)

    def forward(self, X:torch.tensor):
        return self.word_embedding(X) + self.pos_embedding(self.pos_test[:, :X.shape[-1]].to(X.device))

'''
I omitted Batch size,that X is (2, 12, 24)
EED output X(12, 24)
Then X pass through three linear layers to get Q, K, V.
X(12, 24) @ Wq(24, 24) = Q(12, 24)
X(12, 24) @ Wk(24, 24) = K(12, 24)
X(12, 24) @ Wv(24, 24) = V(12, 24)
Then use Q @ K^T to get Attention Score(weight)
Q(12, 24) @ K^T(12, 24) = A(12, 12)
A = A / sqrt(d_k)
A = softmax(A)
A(12, 12) * V(12, 24) = O(12, 24)
O(12, 24) * Wo(24, 24) = O(12, 24)
Total:O = (Softmax(Q*K^T) * V) * Wo
'''

'''
Four-head attention
Q(12, 24) @ K^T(24, 12) = A(12, 12) , Loss message too much
Q(12, 24)-> Q(12, 4, 6) ， K(12, 24)-> K(12, 4, 6) ， V(12, 24)-> V(12, 4, 6)
If use Q @ K^T, the output shape is (4, 4), But we need (4, 12, 12)，12 is the number of heads.
So we need （4, 12, 6) @ （4, 6, 12) = (4, 12, 12)
So we need to transpose the Q(12, 4, 6) to (4, 12, 6) and K(12, 4, 6) -> (4, 6, 12)
Q(4, 12, 6) @ K^T(4, 6, 12) = A(4, 12, 12) [If the first dimension is the same, look at the following ones]
Now,A(4, 12, 12) @ V(4, 12, 6) = O(4, 12, 6)
Pytorch need input size = output size, so we need to reshape O(4, 12, 6) -> O(12, 4, 6) -> O(12, 24)
The above method creates two new transpose function implementations.
'''

def transpose_for_QKV(QKV:torch.tensor):
    QKV = QKV.reshape(QKV.shape[0], QKV.shape[1], 4, QKV.shape[-1]//4)
    # (2, 12, 24) -> (2, 12, 4, 6) We need to count, so write batch size
    QKV = QKV.transpose(-2, -3)
    # (2, 12, 4, 6) -> (2, 4, 12, 6), I like to transpose from back to front
    return QKV

def transpose_for_O(O):
    O = O.transpose(-2, -3)
    # (2, 4, 12, 6) -> (2, 12, 4, 6)
    O = O.reshape(O.shape[0], O.shape[1], -1)
    return O

def attentionScore(Q, K, V, M):
    SqrtQ = Q.shape[-1] ** 0.5
    A = Q @ K.transpose(-1, -2) / SqrtQ
    M = M.unsqueeze(1)
    A.masked_fill_(M == 0, -torch.tensor(float('inf')))
    A = torch.softmax(A, dim=-1)
    O = A @ V
    return O

class Attention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Attention_block, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wk = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wv = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wo = nn.Linear(num_hiddens, num_hiddens, bias=False)

    ''' 
    # Normal attention
    def forward(self, X: torch.Tensor):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        O = attentionScore(Q, K, V)
        O = O @ self.Wo.weight
    '''

    # Multi-head attention
    def forward(self, X, M:torch.Tensor):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        Q, K, V = transpose_for_QKV(Q), transpose_for_QKV(K), transpose_for_QKV(V)
        O = attentionScore(Q, K, V, M)
        O = transpose_for_O(O)
        # O = O @ self.Wo.weight
        O = self.Wo(O)
        return O

class CrossAttention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CrossAttention_block, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wk = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wv = nn.Linear(num_hiddens, num_hiddens, bias=False)
        self.Wo = nn.Linear(num_hiddens, num_hiddens, bias=False)

    # Multi-head attention
    def forward(self, X, X_encoder, inputMask):
        Q, K, V = self.Wq(X), self.Wk(X_encoder), self.Wv(X_encoder)
        Q, K, V = transpose_for_QKV(Q), transpose_for_QKV(K), transpose_for_QKV(V)
        O = attentionScore(Q, K, V, inputMask)
        O = transpose_for_O(O)
        O = self.Wo(O)

        return O

# Position-wise Feed-Forward Networks
class PositionWiseFFN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(PositionWiseFFN, self).__init__(*args, **kwargs)
        self.lin_1 = nn.Linear(num_hiddens, 1024, bias=False)
        self.relu1 = nn.ReLU()
        self.lin_2 = nn.Linear(1024, num_hiddens, bias=False)
        self.relu2 = nn.ReLU()

    def forward(self, X):
        X = self.lin_1(X)
        X = self.relu1(X)
        X = self.lin_2(X)
        X = self.relu2(X)
        return X

class AddNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AddNorm, self).__init__(*args, **kwargs)
        self.add_norm = nn.LayerNorm(num_hiddens)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X, X1):
        X1 = self.add_norm(X1)
        X = X + X1
        X = self.dropout(X)
        return X

class Encoder_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Encoder_block, self).__init__(*args, **kwargs)
        self.attention = Attention_block()
        self.add_norm_1 = AddNorm()
        self.FFN = PositionWiseFFN()
        self.add_norm_2 = AddNorm()
    def forward(self, X, inputMask):
        inputMask = inputMask.unsqueeze(-2)
        X_1 = self.attention(X, inputMask)
        X = self.add_norm_1(X, X_1)
        X_1 = self.FFN(X)
        X = self.add_norm_2(X, X_1)
        return X

class EncoderLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EncoderLayer, self).__init__(*args, **kwargs)
        self.ebd = Embedding()
        self.encoder_block = nn.Sequential()
        self.encoder_block.append(Encoder_block())
        self.encoder_block.append(Encoder_block())
        self.encoder_block.append(Encoder_block())
        self.encoder_block.append(Encoder_block())

    def forward(self, X, inputMask):
        X = self.ebd(X)
        for encoder_block in self.encoder_block:
            X = encoder_block(X, inputMask)
        return X

class Decoder_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Decoder_block, self).__init__(*args, **kwargs)
        self.attention = Attention_block()
        self.add_norm_1 = AddNorm()
        self.cross_attention = CrossAttention_block()
        self.add_norm_2 = AddNorm()
        self.FFN = PositionWiseFFN()
        self.add_norm_3 = AddNorm()
        mask_matrix = torch.ones(12, 12)
        self.tril_mask = torch.tril(mask_matrix).unsqueeze(0)

    def forward(self, X_target, outputMask, X_encoder, inputMask):
        outputMask = outputMask.unsqueeze(-2)
        inputMask = inputMask.unsqueeze(-2)
        X_1 = self.attention(X_target, outputMask * self.tril_mask[:, :outputMask.shape[-1], :outputMask.shape[-1]].to(X_target.device))
        X_target = self.add_norm_1(X_target, X_1)
        X_1 = self.cross_attention(X_target, X_encoder, inputMask)
        X_target = self.add_norm_2(X_target, X_1)
        X_1 = self.FFN(X_target)
        X_target = self.add_norm_3(X_target, X_1)
        return X_target

class DecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DecoderLayer, self).__init__(*args, **kwargs)
        self.ebd = Embedding()
        self.decoder_block = nn.Sequential()
        self.decoder_block.append(Decoder_block())
        self.decoder_block.append(Decoder_block())
        self.decoder_block.append(Decoder_block())
        self.decoder_block.append(Decoder_block())
        self.dense = nn.Linear(num_hiddens, 29, bias=False)

    def forward(self, X_target, outputMask, X_encoder, inputMask):
        X_target = self.ebd(X_target)
        for decoder_block in self.decoder_block:
            X_target = decoder_block(X_target, outputMask, X_encoder, inputMask)
        X_target = self.dense(X_target)
        return X_target

class Transformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Transformer, self).__init__(*args, **kwargs)
        self.EncoderLayer = EncoderLayer()
        self.DecoderLayer = DecoderLayer()

    def forward(self, X_sourse, inputMask, X_target, outputMask):
        X_encoder = self.EncoderLayer(X_sourse, inputMask)
        X_decoder = self.DecoderLayer(X_target, outputMask, X_encoder, inputMask)
        return X_decoder

if __name__ == "__main__":
    pass
