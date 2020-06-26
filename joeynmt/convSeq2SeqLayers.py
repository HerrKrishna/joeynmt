import torch
import torch.nn as nn
from torch import Tensor
from joeynmt.transformer_layers import MultiHeadedAttention
import torch.nn.functional as F



class AbsolutePositionalEncoding(nn.Module):

    def __init__(self,
                 embedding_size: int = 512,
                 max_len: int = 5000):

        super(AbsolutePositionalEncoding, self).__init__()
        positions = torch.arange(0, max_len)
        self.embed = nn.Embedding(max_len, embedding_size)
        self.embed.weight.requires_grad = False
        #check if model uses GPU and move positions to GPU if necessary
        if next(self.parameters()).is_cuda:
            print('on gpu')
            self.positions = positions.cuda()
        else:
            print('on cpu')
            self.positions = positions
        

    def forward(self, emb):
        """Embed inputs.
        Args:
            :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        """
        ape = self.embed(self.positions)
        return emb + ape[:emb.size(1), :]


class MultiStepAttention(nn.Module):

    def __init__(self,
                 hidden_size: int = 512,
                 embedding_size: int = 512):
        super(MultiStepAttention, self).__init__()

        self.hidden2emb = nn.Linear(hidden_size, embedding_size)

    def forward(self,
                x: Tensor,
                trg_embed: Tensor,
                encoder_out: Tensor,
                src_embed: Tensor) -> (Tensor, Tensor):

        x = x.permute(0, 2, 1)
        x = self.hidden2emb(x)
        queries = x + trg_embed
        keys = encoder_out.permute(0, 2, 1)
        attention_weights = F.softmax(torch.matmul(queries, keys), dim=-1)
        values = encoder_out + src_embed
        attention_out = torch.matmul(attention_weights, values)
        attention_out = (x + attention_out)
        attention_out = attention_out.permute(0, 2, 1)

        return attention_weights, attention_out


class ConvSeq2SeqEncoderLayer(nn.Module):
    """
    A Convolutional Seq2Seq Encoder Layer.
    Also called Block in the paper.
    Applies dropout, a convolution,
    a GLU and a residual connection.
    """

    def __init__(self,
                 hidden_size: int = 0,
                 kernel_size: int = 0,
                 dropout: float = 0.1):
        """
        A single ConvSeq2Seq layer.
        :param hidden_size:
        :param kernel_size:
        :param dropout:
        """
        super(ConvSeq2SeqEncoderLayer, self).__init__()

        self.convolution = nn.Conv1d(in_channels=hidden_size,
                                     out_channels=2 * hidden_size,
                                     kernel_size=kernel_size,
                                     padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(p=dropout)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a single Convolutional Seq2Seq encoder layer.
        Applies dropout, a convolution, a GLU and a residual connection.

        :param x: layer input
        :return: output tensor
        """

        x = self.dropout(x)
        residual = x
        x = self.convolution(x)
        x = F.glu(x, dim=1)
        x += residual
        return x


class ConvSeq2SeqDecoderLayer(nn.Module):
    """
    ConvSeq2Seq decoder layer.

    """

    def __init__(self,
                 hidden_size: int = 512,
                 embedding_size: int = 64,
                 kernel_size: int = 5,
                 use_multi_head: bool = False,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        A single ConvSeq2Seq decoderlayer.
        :param hidden_size:
        :param kernel_size:
        :param dropout:
        """
        super(ConvSeq2SeqDecoderLayer, self).__init__()

        self.convolution = nn.Conv1d(in_channels=hidden_size,
                                     out_channels=2 * hidden_size,
                                     kernel_size=kernel_size)
        self.use_multi_head = use_multi_head
        if use_multi_head:
            self.attention = MultiHeadedAttention(num_heads=num_heads,size=hidden_size)
        else:
            self.attention = MultiStepAttention(hidden_size=hidden_size, embedding_size=embedding_size)
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=dropout)

    # pylint: disable=arguments-differ
    def forward(self,
                x: Tensor,
                trg_embed: Tensor,
                encoder_output: Tensor,
                src_embed: Tensor) -> Tensor:
        """
        Forward pass for a single Convolutional Seq2Seq decoder layer.
        Applies dropout to its input, then convolution,
        then a Gated Linear Unit.

        :param x: layer input [Batch_size, hidden_dim, seq_length]
        :return: output tensor
        """

        x = self.dropout(x)
        residual = x
        x = F.pad(x, (self.kernel_size-1, 0), "constant", 0)
        x = self.convolution(x)
        x = F.glu(x, dim=1)
        if self.use_multi_head:
            x = self.attention(encoder_output, encoder_output, x.permute(0,2,1))
            x = x.permute(0,2,1)
        else:
            attention, x = self.attention(x, trg_embed, encoder_output, src_embed)
        x += residual
        return x

