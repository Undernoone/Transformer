import torch
from torch import nn

class EED(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EED, self).__init__(*args, **kwargs)
        self.word_embedding = nn.Embedding(28, 24)
        self.pos_embedding = nn.Embedding(12, 24)
        self.pos_test = torch.arange(12).reshape(1, 12)

    def forward(self,X: torch.Tensor):
        return self.word_embedding(X) + self.pos_embedding(self.pos_test)

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

def transpose_for_QKV(QKV):
    QKV = QKV.reshape(QKV.shape[0], QKV.shape[1], 4,-1)
    # (2, 12, 24) -> (2, 12, 4, 6) We need to count, so write batch size
    QKV = QKV.transpose(-2, -3)
    # (2, 12, 4, 6) -> (2, 4, 12, 6), I like to transpose from back to front
    return QKV

def transpose_for_O(O):
    O = O.transpose(-2, -3)
    # (2, 4, 12, 6) -> (2, 12, 4, 6)
    O = O.reshape(O.shape[0], O.shape[1], O.shape[2]*O.shape[3])
    return O
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

def attention(Q, K, V):
    A = Q @ K.transpose(-1, -2) / Q.shape[-1] ** 0.5
    A = nn.Softmax(dim=-1)(A)
    O = A @ V
    return O

class Attention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Attention_block, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(24, 24, bias=False)
        self.Wk = nn.Linear(24, 24, bias=False)
        self.Wv = nn.Linear(24, 24, bias=False)
        self.Wo = nn.Linear(24, 24, bias=False)

    ''' 
    # Normal attention
    def forward(self, X: torch.Tensor):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        O = attention(Q, K, V)
        O = O @ self.Wo.weight
    '''

    # Multi-head attention
    def forward(self, X: torch.Tensor):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        Q, K, V = transpose_for_QKV(Q), transpose_for_QKV(K), transpose_for_QKV(V)
        O = attention(Q, K, V)
        O = O @ self.Wo.weight
        return O

if __name__ == '__main__':
    test = torch.ones((2, 12)).long()
    ebd = EED()
    test = ebd(test)
    attention = Attention_block()
    test = attention(test)
    pass