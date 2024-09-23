import pandas as pd
from typing import Optional, Tuple, List, Dict, Set
from enum import Enum
from pre_train.aligner.transliterate import Ja2Ar
import os 


cwd = os.path.dirname(os.path.realpath(__file__))
CORPUS_PATH = cwd + "/corpora/"


class Lang(Enum):
    AR = 1,
    HE = 2,
    AM = 3


class Corpus:
    _lang: str
    _legal_letters: str
    _translation_rules: Optional[Tuple[str, str]]
    _corpus: pd.DataFrame
    _total_words: int

    def __init__(self, lang: str, legal_letters: str, translation_rules: Optional[Tuple[str, str]] = None):
        self._lang = lang
        self._legal_letters = legal_letters
        self._translation_rules = translation_rules

        self._load()
        self._total_words = self._calc_total_words()

    def _load(self) -> None:
        self._corpus = pd.read_csv(
            filepath_or_buffer=CORPUS_PATH + f"{self._lang}_clear.csv",
            sep=",",
            header=0,
            dtype={"times": int, "word": str, "freq": float},
            index_col="word"
        )

    def _replace_chars(self, word: str) -> str:
        return word if self._translation_rules is None else word.translate(word.maketrans(*self._translation_rules))

    def _clear_word(self, word: str) -> str:
        replaced_word = self._replace_chars(word)
        return ''.join(l for l in replaced_word if l in self._legal_letters)

    def _calc_total_words(self) -> int:
        return self._corpus["times"].sum()

    def _is_word_in_corpus(self, word, replace_chars=True) -> bool:
        searched_word = self._replace_chars(word) if replace_chars is True else word
        return searched_word in self._corpus.index

    def _smallest_freq(self) -> float:
        return 1 / self._total_words

    def _is_legal_word(self, word) -> bool:
        return len(word) == len(self._clear_word(word))

    def find_word_freq(self, word, replace_chars=True) -> float:
        if self._is_legal_word(word) is False:
            raise ValueError(f"The word {word} does not exist in the {self._lang} language")

        searched_word = self._replace_chars(word) if replace_chars is True else word
        if self._is_word_in_corpus(searched_word) is False:
            return 0  # self._smallest_freq()
        return self._corpus.loc[searched_word]["times"].sum() / self._total_words


class CorpusAr(Corpus):
    LANG = Lang.AR.name.lower()
    LEGAL_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهويءةؤئى"
    TRANSLATION_RULES = ("آأإ", "ااا")

    def __init__(self):
        super().__init__(self.LANG, self.LEGAL_LETTERS, translation_rules=self.TRANSLATION_RULES)


class CorpusAm(Corpus):
    LANG = Lang.AM.name.lower()
    LEGAL_LETTERS = "אבגדהוזחטיכלמנסעפצקרשתךםןףץ"

    def __init__(self):
        super().__init__(self.LANG, self.LEGAL_LETTERS)


class CorpusHe(Corpus):
    LANG = Lang.HE.name.lower()
    LEGAL_LETTERS = "אבגדהוזחטיכלמנסעפצקרשתךםןףץ"

    def __init__(self):
        super().__init__(self.LANG, self.LEGAL_LETTERS)


class FreqCalculator:
    _CORPUS_MAP = {
        Lang.AR: CorpusAr(),
        Lang.HE: CorpusHe(),
        Lang.AM: CorpusAm()
    }

    def __init__(self):
        pass

    @staticmethod
    def _get_stem(word: str, lang: Lang, prefix_ar: str, prefix_ja: str) -> str:
        prefix = prefix_ar if lang == Lang.AR else prefix_ja
        if word.startswith(prefix) is False:
            raise ValueError(f"The word {word} does not start with the prefix {prefix}")

        return word[len(prefix):]

    def score(self, word: str, lang: Lang, prefix_ja: str, prefix_ar: Optional[str] = None) -> float:
        corpus = self._CORPUS_MAP[lang]
        stem = self._get_stem(word, lang, prefix_ar, prefix_ja)

        freq_word, freq_stem = corpus.find_word_freq(word), corpus.find_word_freq(stem)
        if lang == Lang.AR:
            return (freq_stem + freq_word) / 2 if (freq_stem * freq_word) > 0 else 0
        else:
            return freq_stem if freq_word == 0 else 0


class FreqComparator:
    LEGAL_AR_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهويءةؤئى"
    FACTOR = 10**2

    _word: Optional[str]
    _prefix_ar: Optional[str]
    _prefix_ja: Optional[str]
    _freq_calculator: Optional[FreqCalculator]

    def __init__(self):
        self._word = None
        self._prefix_ar = None
        self._prefix_ja = None
        self._freq_calculator = FreqCalculator()

    def _is_legal_ar_word(self, word):
        return all(l in self.LEGAL_AR_LETTERS for l in word)

    def _find_ar_transliterated_options(self) -> List[str]:
        return [op.ar for op in Ja2Ar(self._word).get_transliterated_words() if self._is_legal_ar_word(op.ar) and op.ar.startswith(self._prefix_ar)]

    def _get_best_ar_score(self) -> Tuple[str, float]:
        transliterated_options = self._find_ar_transliterated_options()
        transliterated_scores = [(op, self._freq_calculator.score(op, Lang.AR, self._prefix_ja, self._prefix_ar))
                                 for op in transliterated_options]

        return max(transliterated_scores, key=lambda x: x[1])

    def _get_best_nar_score(self, lang: Lang) -> Tuple[str, float]:
        return self._word, self._freq_calculator.score(self._word, lang, self._prefix_ja)

    def is_mixed(self, prefix_ar: str, prefix_ja: str, word: str):
        self._word = word
        self._prefix_ar = prefix_ar
        self._prefix_ja = prefix_ja
        best_scores: Dict[Lang, Tuple[str, float]] = {
            Lang.AR: self._get_best_ar_score(),
            Lang.HE: self._get_best_nar_score(Lang.HE),
            Lang.AM: self._get_best_nar_score(Lang.AM)
        }

        # print(f"word={word} dict={best_scores}")

        if best_scores[Lang.HE][1] > self.FACTOR * best_scores[Lang.AR][1]:
            print(f"The {Lang.HE.name} word {best_scores[Lang.HE][0]} is mixed")
            return True
        if best_scores[Lang.AM][1] / 2050 > self.FACTOR * best_scores[Lang.AR][1]:
            print(f"The {Lang.AM.name} word {best_scores[Lang.AM][0]} is mixed")
            return True
        return False


# print(FreqComparator("ال", "אל").is_mixed("אלסאן"))


# print(self.total_words())
# self._corpus = self._corpus.drop(columns=["freq"])
# self._corpus["word"] = self._corpus["word"].apply(lambda w: self._clear_word(w))
# self._corpus = self._corpus.groupby("word").sum().reset_index().sort_values(by="times", ascending=False)
#
# total_words = self.total_words()
# self._corpus["freq"] = self._corpus["times"] / total_words
