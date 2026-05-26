import json
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextPreprocessor:
    def __init__(self, vocab_path: str, max_seq_len: int = 350, max_vocab: int = 20000):
        self.max_seq_len = max_seq_len
        self.max_vocab = max_vocab
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.word_to_idx = json.load(f)

    def process(self, text: str) -> np.ndarray:
        words = text_to_word_sequence(str(text))
        
        seq = []
        for word in words:
            idx = self.word_to_idx.get(word, 1)
            seq.append(idx)
            
        padded_seq = pad_sequences(
            [seq], maxlen=self.max_seq_len, dtype='int32', 
            padding='post', truncating='post', value=0
        )
        return padded_seq