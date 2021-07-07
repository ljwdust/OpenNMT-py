"""Image Encoder. ONMT deeper network"""
import torch.nn as nn
import torch.nn.functional as F
import torch

from onmt.encoders.encoder import EncoderBase


class Residual_Block(nn.Module):
    def __init__(self, n_feats=64):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(n_feats)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(n_feats)
        self.relu2 = nn.ReLU(True)

    def forward(self, x):
        output = self.batchnorm2(self.conv2(self.relu1(self.batchnorm1(self.conv1(x)))))
        output += x
        output = self.relu2(output)
        return output


class ImageEncoder(EncoderBase):
    """A simple encoder CNN -> RNN for image src.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout,
                 image_chanel_size=3):
        super(ImageEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        self.head = nn.Conv2d(image_chanel_size, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_head = nn.BatchNorm2d(64)

        residual = [Residual_Block(n_feats=64) for _ in range(8)]
        self.residual = nn.Sequential(*residual)

        self.tail = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_tail = nn.BatchNorm2d(64)

        src_size = 64
        dropout = dropout[0] if type(dropout) is list else dropout
        self.rnn = nn.LSTM(src_size, int(rnn_size / self.num_directions),
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.pos_lut = nn.Embedding(1000, src_size)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        if embeddings is not None:
            raise ValueError("Cannot use embeddings with ImageEncoder.")
        # why is the model_opt.__dict__ check necessary?
        if "image_channel_size" not in opt.__dict__:
            image_channel_size = 3
        else:
            image_channel_size = opt.image_channel_size
        return cls(
            opt.enc_layers,
            opt.brnn,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            image_channel_size
        )

    def load_pretrained_vectors(self, opt):
        """Pass in needed options only when modify function definition."""
        pass

    def forward(self, src, lengths=None):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""

        batch_size = src.size(0)
        # (batch_size, 64, imgH, imgW)
        # layer head
        src = F.relu(self.batch_norm_head(self.head(src[:, :, :, :] - 0.5)), True)

        # (batch_size, 64, imgH/2, imgW/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        src = self.residual(src)

        src = F.relu(self.batch_norm_tail(self.tail(src)), True)

        # # (batch_size, 64, H, W)
        all_outputs = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            row_vec = torch.Tensor(batch_size).type_as(inp.data) \
                .long().fill_(row)
            pos_emb = self.pos_lut(row_vec)
            with_pos = torch.cat(
                (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0)
            outputs, hidden_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)

        return hidden_t, out, lengths

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout