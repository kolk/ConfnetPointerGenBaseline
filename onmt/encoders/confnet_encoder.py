import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from onmt.encoders.encoder import EncoderBase
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.utils.rnn_factory import rnn_factory

class ConfnetEncoder(EncoderBase):
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(ConfnetEncoder, self).__init__()
        assert embeddings is not None

        self.embeddings = embeddings
        self.thetav = nn.Linear(in_features=self.embeddings.embedding_size, out_features=self.embeddings.embedding_size)
        self.v_bar = nn.Linear(in_features=self.embeddings.embedding_size, out_features=1)

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        #print('opt enc rnn size', opt.enc_rnn_size)
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge)

    def forward(self, confnets, scores, src_lengths=None, par_arc_lengths=None):
        """
        Based on the paper NEURAL CONFNET CLASSIFICATION (http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0006039.pdf)
        """
        #self._check_args(confnets, src_lengths)

        #print('confnet size', confnets.size())

        emb = self.embeddings(confnets.permute(1,0,2,3)) #(slen, batch, max_par_arc, emb_dim)
        emb_trans = emb.squeeze(3)#.permute(1,0,2,3) #(batch, slen, max_par_arc, emb_dim)
        output_list = torch.tensor([]).cuda()
        #### FEATS NOT SUPPORTED ###
        confnets_ = confnets.squeeze(-1).permute(1, 0, 2)  # (max_sent_len, batch, max_par_arc_len, emb_dim)
        scores_ = scores.squeeze(-1).permute(1, 0, 2)  # (max_sent_len, batch, max_par_arc_len)
        par_arc_lengths_ = par_arc_lengths.permute(1, 0)  # (max_sent_lens, batch)
        for em, score, lengths in zip(emb_trans, scores_, par_arc_lengths_):
            # word embedding
            # s_len, batch, emb_dim = emb.size()
            # output = self.dropout(output)
            sc = score.unsqueeze(-1).expand(em.size()) #(batch, max_par_arc, emb_sz)

              # confnet score weighted word embedding
            q = em.float() * sc.float() #(batch, max_par_arc, emb_sz)
            batch_size, max_par_arcs, emb_sz = q.size()
            v = torch.tanh(self.thetav(q)) #(batch, max_par_arc, emb_sz)
            v_bar = self.v_bar(v).squeeze(-1)
            #### masking: Mask the padding ####
            mask = torch.arange(max_par_arcs)[None, :].to("cuda") < lengths[:, None].to("cuda").type(torch.float)
            mask = mask.type(torch.float)
            masked_v_bar = torch.where(mask == False, torch.tensor([float("-inf") - 1e-10], device=q.device), v_bar)
            attention = torch.softmax(masked_v_bar, dim=1)
            final_attention = attention.masked_fill(torch.isnan(attention), 0)
            # apply attention weights
            output = q * final_attention.unsqueeze(-1).expand(q.size())

            # most attented words
            most_attentive_arc = torch.argmax(final_attention, dim=1)
            # highest attention weights
            most_attentive_arc_weights, _ = torch.max(final_attention, dim=1)
            # a = output

            a = torch.sum(output, dim=1)
            output_list = torch.cat((output_list, a.unsqueeze(0)), dim=0)
        # a = self.dropout(a)

        output_confnet = output_list.permute(1, 0, 2)  # (batch, max_sent_len, hid_dim)
        #return a, output_confnet, src_lengths #most_attentive_arc, attention, most_attentive_arc_weights  # output, h_output

        packed_emb_ = output_confnet.permute(1,0,2)
        if src_lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = src_lengths.view(-1).tolist()
            packed_emb = pack(packed_emb_, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if src_lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, src_lengths


    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):
        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])


    def _bridge(self, hidden):
        """Forward hidden state through bridge."""

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


    def update_dropout(self, dropout):
        self.rnn.dropout = dropout