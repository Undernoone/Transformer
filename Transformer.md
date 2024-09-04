1. **实例化 `EncoderLayer`：** 初始化编码器层的实例，包括嵌入、注意力和前馈神经网络模块。
2. **词嵌入和位置嵌入：** 输入首先经过词嵌入（wordEBD）和位置嵌入（posEBD）层，生成词汇和位置的嵌入向量。
3. **顺序模块的构建：** 使用 `nn.Sequential` 构建一个顺序容器，将 `Encoder_block` 模块放入其中。
4. **注意力块（Attention Block）：** 
   - 输入生成查询（Q）、键（K）和值（V）向量。
   - 对 `Q`、`K` 和 `V` 进行 `transpose` 以适应矩阵计算。
   - 使用 `Q` 和 `K` 计算注意力得分，并通过 `softmax` 转化为权重。
   - 注意力权重与 `V` 相乘，得到注意力输出 `O`。
   - 根据需要对 `O` 进行 `transpose` 以匹配后续处理的维度要求。
5. **进入 `ADDNorm` 层：** 将注意力块的输出通过残差连接和层归一化处理。
6. **前馈神经网络（FFN）层：** 输入通过两层全连接网络和激活函数处理。
7. **再次进入 `ADDNorm` 层：** 对 FFN 输出应用残差连接和层归一化处理。
8. **整个编码器层处理完成。**

### 调试时出现循环多次的原因

- **每个 `Encoder_block` 在 `EncoderLayer` 中执行两次**：你在 `EncoderLayer` 中定义了两个 `Encoder_block`，因此当你执行 `EncoderLayer` 的 `forward` 时，`Encoder_block` 的 `forward` 方法会被调用两次。
- **`Encoder_block` 内部的 `Attention_block` 也会执行两次**：由于每个 `Encoder_block` 包含了 `Attention_block`，在执行 `EncoderLayer` 的 `forward` 时，`Attention_block` 的逻辑也会被重复两次。

### 代码结构与执行顺序

1. `Embedding` 层将输入嵌入到高维空间中。

2. 输入经过两个 Encoder_block，每个 Encoder_block

    内部进行：

   - 多头注意力机制 (`Attention_block`)。
   - 残差连接加层归一化 (`AddNorm`)。
   - 前馈神经网络 (`PositionWiseFFN`)。
   - 再次残差连接加层归一化 (`AddNorm`)。

3. `EncoderLayer` 将上述过程执行两次，因为 `Encoder_block` 在 `nn.Sequential` 中被添加了两次。