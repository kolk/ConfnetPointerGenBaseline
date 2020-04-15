""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, ques_encoder, ans_encoder, decoder, vocab=None):
        super(NMTModel, self).__init__()
        self.ques_encoder = ques_encoder
        self.ans_encoder = ans_encoder
        self.decoder = decoder
        self.vocab_ = vocab

    def forward(self, batch, ques, ques_scores, ans, tgt, ques_lengths, par_arc_lengths, ans_lengths, bptt=False, with_align=False):
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

        ques_enc_state, ques_memory_bank, ques_lengths = self.ques_encoder(ques, ques_scores, ques_lengths,
                                                                           par_arc_lengths)
        #print('ques_lengths', ques_lengths)
        #print('ans_lengths', ans_lengths)

        """
        ans_lens_sorted, idx = torch.sort(ans_lengths, descending=True)
        ans_sorted = ans[:,idx,:]

        ans_enc_state_, ans_memory_bank_, ans_lengths_sorted = self.ans_encoder(ans_sorted, ans_lens_sorted)
        rev_indx = [(idx == i).nonzero().item() for i in range(len(ans_lens_sorted))]
        ans_enc_state = tuple(enc_ans[:, rev_indx, :] for enc_ans in ans_enc_state_)
        ans_memory_bank = ans_memory_bank_[:,rev_indx,:]
        """
        ans_enc_state, ans_memory_bank, ans_lengths = self.ans_encoder(ans, ans_lengths)
        enc_state =  tuple(torch.add(enc_q, enc_ans) for enc_q, enc_ans in zip(ques_enc_state, ans_enc_state))
        memory_bank = torch.cat([ques_memory_bank, ans_memory_bank], 0)
        memory_lengths = torch.add(ques_lengths, ans_lengths)

        if batch.src_map.size()[0] != memory_bank.size()[0]:
            print('src_map size', batch.src_map.size())
            print('memory_bank.size', memory_bank.size())
            for i, (sent, voc) in enumerate(zip(batch.src_map.permute(1,0,2), batch.src_ex_vocab)):
                for j, word in enumerate(sent):
                    par_arcs_ii = (word !=0).nonzero()
                    print(j, [voc.itos[w] for w in par_arcs_ii])
                print('^^^^^^^^^^^^^^^^^^')

            #print('ques size', ques.size())
            for i, sent in enumerate(ques.squeeze(-1)):
                for j, par_arc in enumerate(sent):
                    print(j, [self.vocab_.itos[w] for w in par_arc])
                print('&&&&&&&&&&&&&&&&&&')

            for i, sent in enumerate(tgt.squeeze(-1).permute(1,0)):
                print(i, [self.vocab_.itos[w] for w in sent])
                print('$$$$$$$$$$$$$$$$$$$$$')
            for i, sent in enumerate(ans.squeeze(-1).permute(1,0)):
                print(i, [self.vocab_.itos[w] for w in sent])
                print('##########################')



        if bptt is False:
            self.decoder.init_state(ans, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=memory_lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
