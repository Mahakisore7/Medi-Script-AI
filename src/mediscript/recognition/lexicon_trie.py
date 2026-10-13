import os
from pathlib import Path
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class LexiconTrie:
    """
    A simple character-level Trie for constraining text generation or 
    fast prefix lookups against a known medical lexicon.
    """
    def __init__(self, words: list[str], case_insensitive: bool = True):
        self.root = TrieNode()
        self.case_insensitive = case_insensitive
        for word in words:
            if word:
                self.insert(word)

    def _normalize(self, text: str) -> str:
        return text.lower() if self.case_insensitive else text

    def insert(self, word: str):
        node = self.root
        for char in self._normalize(word):
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def is_valid_prefix(self, prefix: str) -> bool:
        """Checks if the given string is a valid prefix of any word in the trie."""
        node = self.root
        for char in self._normalize(prefix):
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def is_valid_word(self, word: str) -> bool:
        """Checks if the exact word exists in the trie."""
        node = self.root
        for char in self._normalize(word):
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def get_allowed_next_characters(self, prefix: str) -> list[str]:
        """
        Returns a list of valid next characters given a prefix.
        This will be used to filter allowed tokens during generation.
        """
        node = self.root
        for char in self._normalize(prefix):
            if char not in node.children:
                return []
            node = node.children[char]
        
        allowed = list(node.children.keys())
        if node.is_end_of_word:
            # We use an empty string to denote that stopping here is allowed
            allowed.append("")
        return allowed


def load_csv_lexicon(path: str) -> list[str]:
    """Loads a single-column CSV into a list of strings."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        # Assumes single column, no header or just text per line
        lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]


class MedicalLexicons:
    """
    Singleton-like access to the different lexicon tries.
    """
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default relative to this file
            from mediscript.config import LEXICON_DIR

            data_dir = str(LEXICON_DIR)
            
        self.medicines_path = os.path.join(data_dir, "medicines.csv")
        self.symptoms_path = os.path.join(data_dir, "symptoms.csv")
        
        # Hardcode section tokens per PRD Phase 1:
        # "Section-token lexicon includes the non-standard abbreviations actually observed"
        self.section_tokens = [
            "Rx:", "Erp:", "Px:", "Tx:", "Adv:", "-> Adv:", "Symptoms:", "C/O:", "Name:", "Age:"
        ]
        
        self.medicines_trie = None
        self.symptoms_trie = None
        self.sections_trie = None

    def get_medicines_trie(self) -> LexiconTrie:
        if self.medicines_trie is None:
            words = load_csv_lexicon(self.medicines_path)
            self.medicines_trie = LexiconTrie(words)
        return self.medicines_trie

    def get_symptoms_trie(self) -> LexiconTrie:
        if self.symptoms_trie is None:
            words = load_csv_lexicon(self.symptoms_path)
            self.symptoms_trie = LexiconTrie(words)
        return self.symptoms_trie

    def get_sections_trie(self) -> LexiconTrie:
        if self.sections_trie is None:
            self.sections_trie = LexiconTrie(self.section_tokens, case_insensitive=True)
        return self.sections_trie

if __name__ == "__main__":
    lexicons = MedicalLexicons()
    med_trie = lexicons.get_medicines_trie()
    print("Loaded medicines trie.")
    # Quick test
    prefix = "par"
    print(f"Next chars for '{prefix}': {med_trie.get_allowed_next_characters(prefix)}")
