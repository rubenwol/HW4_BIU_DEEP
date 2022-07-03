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
    prem_idx = [self.w2i.get(word.lower(), self.w2i['unk']) for word in prem]
    hyp_idx = [self.w2i.get(word.lower(), self.w2i['unk']) for word in hyp]
    return torch.tensor(prem_idx), torch.tensor(hyp_idx), torch.LongTensor([self.dataset[idx]['label']])



def pad_collate(batch):
  (prem, hyp, y) = zip(*batch)
  prem_lens = torch.tensor([len(p) for p in prem])
  hyp_lens = torch.tensor([len(h) for h in hyp])

  prem_pad = pad_sequence(prem, batch_first=True, padding_value=0)
  hyp_pad = pad_sequence(hyp, batch_first=True, padding_value=0)
  y_pad = pad_sequence(y, batch_first=True, padding_value=0)

  return (prem_pad, prem_lens), (hyp_pad, hyp_lens), y_pad



def read_glove_vector(glove_vec):
  with open(glove_vec, 'r', encoding='UTF-8') as f:
    words = []
    vecs = []
    for line in tqdm(f):
      w_line = line.split(' ')
      words.append(w_line[0])
      vecs.append(np.array(w_line[1:], dtype=np.float64))
  return words, np.array(vecs)
