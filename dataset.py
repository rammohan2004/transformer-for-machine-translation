"""
dataset.py — Multi30k Dataset Loading & Preprocessing
DA6401 Assignment 3
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy
from collections import Counter


# ── Special token indices (keep consistent across all files) ──────────
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY HELPER
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Maps tokens <-> indices. Special tokens always at indices 0-3."""

    def __init__(self):
        self.stoi = {}  # string → index
        self.itos = {}  # index → string

    def build(self, token_lists, min_freq=2):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        # Special tokens at fixed positions 0,1,2,3
        self.stoi = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.itos = {idx: tok for tok, idx in self.stoi.items()}

        for tok, freq in counter.items():
            if freq >= min_freq and tok not in self.stoi:
                idx = len(self.stoi)
                self.stoi[tok] = idx
                self.itos[idx] = tok

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens):
        return [self.stoi.get(tok, UNK_IDX) for tok in tokens]

    def decode(self, indices):
        return [self.itos.get(idx, '<unk>') for idx in indices]


# ══════════════════════════════════════════════════════════════════════
#  DATASET — strictly matches skeleton signatures
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.split = split

        # Load spaCy tokenizers safely
        import spacy
        from spacy.cli import download

        try:
            self.spacy_de = spacy.load('de_core_news_sm')
        except OSError:
            download('de_core_news_sm')
            self.spacy_de = spacy.load('de_core_news_sm')

        try:
            self.spacy_en = spacy.load('en_core_web_sm')
        except OSError:
            download('en_core_web_sm')
            self.spacy_en = spacy.load('en_core_web_sm')

        # Load raw dataset from HuggingFace
        raw = load_dataset('bentrevett/multi30k')
        self.raw_data = raw[split]

        # Tokenize all sentences (strings → token-string lists)
        # Done here so build_vocab() and process_data() can both use them
        self.src_tokens = [
            [tok.text.lower() for tok in self.spacy_de.tokenizer(ex['de'])]
            for ex in self.raw_data
        ]
        self.tgt_tokens = [
            [tok.text.lower() for tok in self.spacy_en.tokenizer(ex['en'])]
            for ex in self.raw_data
        ]

        # Will be populated by build_vocab() / optionally pre-set externally
        self.src_vocab = None
        self.tgt_vocab = None

        # Will be populated by process_data()
        self.data = []

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        MIN_FREQ = 2  

        if self.src_vocab is None:
            self.src_vocab = Vocabulary()
            self.src_vocab.build(self.src_tokens, min_freq=MIN_FREQ)

        if self.tgt_vocab is None:
            self.tgt_vocab = Vocabulary()
            self.tgt_vocab.build(self.tgt_tokens, min_freq=MIN_FREQ)
        
        

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        assert self.src_vocab is not None and self.tgt_vocab is not None, \
            "Call build_vocab() before process_data()"

        self.data = []
        for src_toks, tgt_toks in zip(self.src_tokens, self.tgt_tokens):
            src_ids = [SOS_IDX] + self.src_vocab.encode(src_toks) + [EOS_IDX]
            tgt_ids = [SOS_IDX] + self.tgt_vocab.encode(tgt_toks) + [EOS_IDX]
            self.data.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



# ══════════════════════════════════════════════════════════════════════
#  COLLATE FUNCTION
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """
    Pads variable-length sequences to longest in batch.

    Returns:
        src_batch : [batch_size, max_src_len]  padded with PAD_IDX=1
        tgt_batch : [batch_size, max_tgt_len]  padded with PAD_IDX=1
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True,
                              padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True,
                              padding_value=PAD_IDX)
    return src_batch, tgt_batch


# ══════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size=128, num_workers=2):
    """
    Builds train/val/test dataloaders with vocab shared from train only.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # Train: builds vocab from scratch
    train_ds = Multi30kDataset('train')
    train_ds.build_vocab()
    train_ds.process_data()

    # Val/Test: inject train vocab, then call build_vocab() (which skips rebuild)
    val_ds = Multi30kDataset('validation')
    val_ds.src_vocab = train_ds.src_vocab
    val_ds.tgt_vocab = train_ds.tgt_vocab
    val_ds.build_vocab()
    val_ds.process_data()

    test_ds = Multi30kDataset('test')
    test_ds.src_vocab = train_ds.src_vocab
    test_ds.tgt_vocab = train_ds.tgt_vocab
    test_ds.build_vocab()
    test_ds.process_data()

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  collate_fn=collate_fn,
                               num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, collate_fn=collate_fn,
                               num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                               shuffle=False, collate_fn=collate_fn,
                               num_workers=num_workers)

    return train_loader, val_loader, test_loader, \
           train_ds.src_vocab, train_ds.tgt_vocab


# ── Sanity check ──────────────────────────────────────────────────────
if __name__ == '__main__':
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(batch_size=4)

    print(f"Src vocab size : {len(src_vocab)}")
    print(f"Tgt vocab size : {len(tgt_vocab)}")
    print(f"Train batches  : {len(train_loader)}")
    print(f"Val   batches  : {len(val_loader)}")
    print(f"Test  batches  : {len(test_loader)}")

    src, tgt = next(iter(train_loader))
    print(f"src shape : {src.shape}")   # [4, src_len]
    print(f"tgt shape : {tgt.shape}")   # [4, tgt_len]
    # First token should be SOS=2, last should be EOS=3
    print(f"src[0]    : {src[0]}")
    print(f"tgt[0]    : {tgt[0]}")


