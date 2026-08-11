import math
from collections import defaultdict
from rx_recognition.lexicon_trie import MedicalLexicons

class DomainLM:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        
        lexicons = MedicalLexicons()
        
        self.lexicon_words = set()
        
        if lexicons.medicines_trie:
            import os
            if os.path.exists(lexicons.medicines_path):
                with open(lexicons.medicines_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.lexicon_words.add(line.strip().lower())
                            
        if lexicons.symptoms_trie:
            import os
            if os.path.exists(lexicons.symptoms_path):
                with open(lexicons.symptoms_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.lexicon_words.add(line.strip().lower())
                            
        # Character-level n-gram model
        for word in self.lexicon_words:
            padded = "^" * (self.n - 1) + word + "$"
            for char in word:
                self.vocab.add(char)
            self.vocab.add("^")
            self.vocab.add("$")
            
            for i in range(len(padded) - self.n + 1):
                ngram = padded[i:i+self.n]
                context = padded[i:i+self.n-1]
                self.ngrams[ngram] += 1
                self.context_counts[context] += 1
                
        self.vocab_size = len(self.vocab)
        
    def score_sequence(self, text: str) -> float:
        """Returns normalized log probability of the text according to the character n-gram model."""
        if not text:
            return -100.0
            
        padded = "^" * (self.n - 1) + text.lower() + "$"
        log_prob = 0.0
        
        for i in range(len(padded) - self.n + 1):
            ngram = padded[i:i+self.n]
            context = padded[i:i+self.n-1]
            
            count = self.ngrams.get(ngram, 0)
            context_count = self.context_counts.get(context, 0)
            
            # Laplace smoothing
            prob = (count + 1) / (context_count + self.vocab_size + 1)
            log_prob += math.log(prob)
            
        return log_prob / max(1, len(text))
        
    def rescore_beams(self, beams: list[tuple[str, float]], field_type: str = None) -> tuple[str, float]:
        """
        Takes a list of (text, trocr_score) and returns the best (text, final_score).
        trocr_score is assumed to be 0-100 pseudo-confidence.
        """
        best_text = ""
        best_score = -float('inf')
        best_conf = 0.0
        
        w_trocr = 1.0
        w_lm = 0.5
        w_lexicon = 1.0  # Reduced bonus so it doesn't override correct fine-tuned TrOCR predictions
        
        for text, conf in beams:
            lm_score = self.score_sequence(text)
            
            lexicon_bonus = 0.0
            # Clean text for checking
            clean_text = text.lower().strip()
            if field_type in ("medicine", "symptoms") and clean_text in self.lexicon_words:
                lexicon_bonus = 1.0
                
            final_score = w_trocr * (conf / 100.0) + w_lm * lm_score + w_lexicon * lexicon_bonus
            
            if final_score > best_score:
                best_score = final_score
                best_text = text
                best_conf = conf
                
        return best_text, best_conf

_domain_lm_instance = None
def get_domain_lm():
    global _domain_lm_instance
    if _domain_lm_instance is None:
        _domain_lm_instance = DomainLM()
    return _domain_lm_instance
