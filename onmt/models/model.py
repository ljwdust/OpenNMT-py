""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
from torch.nn import functional as F
import skimage.io
import kornia


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, rectifier=None, tps_inputsize=[32,64]):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rectifier = rectifier
        if rectifier is not None:
            self.model_tps, self.model_stn_head = rectifier
        self.tps_inputsize = tps_inputsize
        self.cnt = 0

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs

        if self.rectifier is not None:
            # input images are downsampled before being fed into stn_head.
            stn_input = F.interpolate(src, self.tps_inputsize, mode='bilinear', align_corners=True)
            stn_img_feat, ctrl_points = self.model_stn_head(stn_input)

            self.cnt += 1
            skimage.io.imsave(str(self.cnt)+'-1.jpg', src[0].squeeze().cpu().numpy())
            # src, _ = self.model_tps(src, ctrl_points)

            dst_h, dst_w = src.shape[2:4]
            points_dst = torch.tensor([[
                [0., 0.], [dst_w - 1., 0.], [dst_w - 1., dst_h - 1.], [0., dst_h - 1.],
            ]])
            points_dst = points_dst.repeat(src.shape[0], 1, 1)
            points_dst = points_dst.cuda()

            # p_dst = torch.cat((points_dst, points_dst), 0)
            # p_dst = p_dst.cuda()

            ctrl_points2 = ctrl_points.clone()
            ctrl_points[:,2,:] = ctrl_points2[:,3,:]
            ctrl_points[:,3,:] = ctrl_points2[:,2,:]
            ctrl_points[:,:,0] *= dst_w
            ctrl_points[:,:,1] *= dst_h

            M = kornia.get_perspective_transform(ctrl_points, points_dst)
            src = kornia.warp_perspective(src, M, dsize=(dst_h, dst_w), border_mode='border')

            skimage.io.imsave(str(self.cnt)+'-2.jpg', src[0].squeeze().cpu().detach().numpy())

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
