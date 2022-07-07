from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from spacy.lang.en import English
nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer


class NLIDataset(Dataset):
  def __init__(self, dataset, w2i):
    self.dataset = dataset
    self.w2i = w2i

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    prem = [token.text for token in tokenizer(self.dataset[idx]['premise'])]
    hyp = [token.text for token in tokenizer(self.dataset[idx]['hypothesis'])]
    prem_idx = [word2idx(word, self.w2i) for word in prem]
    hyp_idx = [word2idx(word, self.w2i) for word in hyp]
    return torch.tensor(prem_idx), torch.tensor(hyp_idx), torch.tensor(self.dataset[idx]['label'])

def word2idx(word, w2i):
  return w2i.get(word, w2i.get(word.lower(), w2i['<UNK>']))

def pad_collate(batch):
  (prem, hyp, y) = zip(*batch)
  prem_lens = torch.tensor([len(p) for p in prem])
  hyp_lens = torch.tensor([len(h) for h in hyp])

  prem_pad = pad_sequence(prem, batch_first=True, padding_value=0)
  hyp_pad = pad_sequence(hyp, batch_first=True, padding_value=0)
  # y_pad = pad_sequence(torch.LongTensor(y), batch_first=True, padding_value=0)

  return (prem_pad, prem_lens), (hyp_pad, hyp_lens), torch.LongTensor(y)



def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = []
    vecs = []
    words.append('<UNK>')
    for line in tqdm(f):
      w_line = line.split(' ')
      words.append(w_line[0])
      vecs.append(np.array(w_line[1:], dtype=np.float64))
  vecs = np.array(vecs)
  unk_vecs = np.mean(vecs, axis=0)
  vecs = np.concatenate((np.array([unk_vecs]), vecs), axis=0)
  return words, vecs
