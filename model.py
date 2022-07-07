import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 bilstm1_dim=300,
                 bilstm2_dim=300,
                 bilstm3_dim=300,
                 ):
        super(BiLSTMEncoder, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.bilstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=bilstm1_dim,
            batch_first=True,
            bidirectional=True
        )

        self.bilstm2 = nn.LSTM(
            input_size=embedding_dim + (bilstm1_dim * 2),
            hidden_size=bilstm2_dim,
            batch_first=True,
            bidirectional=True
        )

        self.bilstm3 = nn.LSTM(
            input_size=embedding_dim + (bilstm2_dim * 2),
            hidden_size=bilstm3_dim,
            batch_first=True,
            bidirectional=True
        )

    def run_lstm(self, x, x_lens, bilstm):

        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        output_packed, _ = bilstm(x_packed)

        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)

        return output_padded, output_lengths



    def forward(self, sents, lengths):

        emb = self.embeddings(sents)

        out1, out1_lens = self.run_lstm(emb, lengths, self.bilstm1)

        out2, out2_lens = self.run_lstm(torch.cat([emb, out1], dim=2), lengths, self.bilstm2)

        out3, out3_lens = self.run_lstm(torch.cat([emb, out1 + out2], dim=2), lengths, self.bilstm3)
        # max pooling
        out, _ = torch.max(out3, dim=1) #check the dim

        return out

class StackBiLSTMClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 bilstm1_dim=300,
                 bilstm2_dim=300,
                 bilstm3_dim=300,
                 mlp_dim=800,
                 out_dim=3,
                 p=0.1
                 ):
        super(StackBiLSTMClassifier, self).__init__()

        self.encoder = BiLSTMEncoder(
            vocab_size,
            embedding_dim,
            bilstm1_dim,
            bilstm2_dim,
            bilstm3_dim
        )

        self.linear1 = nn.Linear(bilstm3_dim * 2 * 4, mlp_dim)

        self.linear_new = nn.Linear(mlp_dim, mlp_dim)

        self.linear2 = nn.Linear(mlp_dim, out_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=p)

    def forward(self, prem, hyp):

        v_prem = self.encoder(prem[0], prem[1])

        v_hyp = self.encoder(hyp[0], hyp[1])

        v_match = torch.cat([v_prem, v_hyp, torch.abs(v_prem-v_hyp), v_prem*v_hyp], dim=1)

        out = self.relu(self.dropout(self.linear1(v_match)))

        out = self.relu(self.dropout(self.linear_new(out)))

        out = self.dropout(self.linear2(out))

        return F.log_softmax(out, dim=-1)



