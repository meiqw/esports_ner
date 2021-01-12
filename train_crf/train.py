from typing import Mapping, Sequence, Dict, Optional, List, Iterable
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from pycrfsuite import Trainer, Tagger
from pymagnitude import Magnitude
from .hw3utils import PRF1, FeatureExtractor, EntityEncoder
from .hw3utils import PUNC_REPEAT_RE, DIGIT_RE, UPPERCASE_RE, LOWERCASE_RE
import copy 
import spacy
import json
import  os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class BiasFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            features["bias"] = 1.0


class TokenFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        key = "tok[{}]={}".format(relative_idx, token)
        features[key] = 1.0

class PosFeature(FeatureExtractor):
    def extract(
        self,
        pos_tag: str,
        current_idx: int,
        relative_idx: int,
        pos_tags: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        key = "tok[{}]={}".format(relative_idx, pos_tag)
        features[key] = 1.0


class UppercaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if token.isupper():
            key = "uppercase[{}]".format(relative_idx)
            features[key] = 1.0


class TitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if token.istitle():
            key = "titlecase[{}]".format(relative_idx)
            features[key] = 1.0


class InitialTitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if token.istitle() and current_idx + relative_idx == 0:
            key = "initialtitlecase[{}]".format(relative_idx)
            features[key] = 1.0


class PunctuationFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if PUNC_REPEAT_RE.match(token) is not None:
            key = "punc[{}]".format(relative_idx)
            features[key] = 1.0


class DigitFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if DIGIT_RE.search(token) is not None:
            key = "digit[{}]".format(relative_idx)
            features[key] = 1.0


class WordShapeFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        shape = ""
        for letter in token:
            if UPPERCASE_RE.match(letter) is not None:
                shape += "X"
            elif LOWERCASE_RE.match(letter) is not None:
                shape += "x"
            elif DIGIT_RE.search(letter) is not None:
                shape += "0"
            else:
                shape += letter
        key = "shape[{}]={}".format(relative_idx, shape)
        features[key] = 1.0


class WordVectorFeature(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0) -> None:
        self.vectors_path = vectors_path
        self.scaling = scaling
        self.vectors = Magnitude(vectors_path, normalized = False)

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            word_vector = self.vectors.query(token)
            word_vector = word_vector * self.scaling
            d = len(word_vector)
            keys = ["v{}".format(i) for i in range(d)]
            features.update(zip(keys, word_vector))

class BrownClusterFeature(FeatureExtractor):
    def __init__(
        self,
        clusters_path: str,
        *,
        use_full_paths: bool = False,
        use_prefixes: bool = False,
        prefixes: Optional[Sequence[int]] = None,
    ):
        if use_full_paths == False and use_prefixes == False:
            print("At least one of use_full_paths or use_prefixes should be \
                   set to True")
            raise ValueError
        self.clusters = dict()
        file = open(clusters_path)
        line = file.readline()
        while line:
            id_token = line.split("\t")
            cluster_id = id_token[0]
            token = id_token[1] 
            self.clusters[token] = cluster_id
            line = file.readline()
        file.close()

        self.use_full_paths = use_full_paths
        self.use_prefixes = use_prefixes
        self.prefixes = prefixes

        

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0 and token in self.clusters:
            cluster_id = self.clusters[token]
            if self.use_full_paths == True:
                key = "cpath={}".format(cluster_id)
                features[key] = 1.0
            else:
                if self.prefixes == None:
                    for i in range(len(cluster_id)):
                        key = "cprefix{}={}".format(i+1, cluster_id[:i+1])
                        features[key] = 1.0
                else:
                    for p in self.prefixes:
                        if p <= len(cluster_id):
                            key = "cprefix{}={}".format(p, cluster_id[:p])
                            features[key] = 1.0
                        else:
                            break

class WindowedTokenFeatureExtractor:
    def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
        self.extractors = feature_extractors
        self.window_size = window_size

    def extract(self, tokens: Sequence[str], pos_tags: Sequence[str]) -> List[Dict[str, float]]:
        n = len(tokens)
        feature_list = []
        for i in range(len(tokens)):
            features = dict()
            for j in range(0, self.window_size + 1):
                k = i + j
                if 0 <= k and k < n:
                    for i, extractor in enumerate(self.extractors):
                        if i==2:
                            extractor.extract(pos_tags[k], i, j, pos_tags, features)
                        else:
                            extractor.extract(tokens[k], i, j, tokens, features)
                if j > 0:
                    k = i - j
                    if 0 <= k and k < n:
                        for i, extractor in enumerate(self.extractors):
                            if i == 2:
                                extractor.extract(pos_tags[k], i, j, pos_tags, features)
                            else:
                                extractor.extract(tokens[k], i, j, tokens, features)
            feature_list.append(features)
        return feature_list


class CRFsuiteEntityRecognizer:
    def __init__(
        self, feature_extractor: WindowedTokenFeatureExtractor, encoder: EntityEncoder
    ) -> None:
        self.feature_extractor = feature_extractor
        self.entity_encoder = encoder
        self.train_has_been_called = False
        self.trainer = None
        self.tagger = None

    @property
    def encoder(self) -> EntityEncoder:
        return self.entity_encoder

    def train(self, docs: Iterable[Doc], algorithm: str, params: dict, path: str) -> None:
        self.train_has_been_called = True
        self.trainer = Trainer(algorithm=algorithm, verbose=False)
        self.trainer.set_params(params)
        for doc in docs:
            for sent in doc.sents:
                tokens = []
                pos_tags = []
                str_tokens = []
                for token in sent:
                    tokens.append(token)
                    str_tokens.append(str(token))
                    pos_tags.append(token.tag_)
                features = self.feature_extractor.extract(str_tokens, pos_tags)
                labels = self.entity_encoder.encode(tokens)
                self.trainer.append(features, labels)
        self.trainer.train(path)



class BILOUEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        labels = []
        for token in tokens:
            if token.ent_iob_ == "":
                labels.append("O")
            else:
                labels.append(token.ent_iob_ + "-" + token.ent_type_)

        return bio_to_bilou(labels)


class BIOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        labels = []
        for token in tokens:
            if token.ent_iob_ == "":
                labels.append("O")
            else:
                labels.append(token.ent_iob_ + "-" + token.ent_type_)

        return labels


class IOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        labels = []
        for token in tokens:
            if token.ent_iob_ == "":
                labels.append("O")
            else:
                labels.append("I" + "-" + token.ent_type_)
        return labels


def bio_to_bilou(labels: Sequence[str]) -> List[str]:
    # All the transitions:
    # "O" -> "O"
    # "B-X" -> "U-X" or "B-X"
    # "I-X" -> "I-X" or "L-X"

    # transition conditions:
    # if "B-X" is not followed by "I-X", then "U-X", else "B-X"
    # if "I-X" is not followed by "I-X", then "L-X", else "I-X"
    bilou_labels = []
    n = len(labels)
    for i in range(0, n):
        code_type = labels[i].split("-")
        if code_type[0] == "O":
            bilou_labels.append("O")
        elif code_type[0] == "B":
            if i == n - 1 or labels[i + 1] != "I-" + code_type[1]:
                bilou_labels.append("U-" + code_type[1])
            else:
                bilou_labels.append(labels[i])
        else:
            # (code_type[0] == "I"):
            if i == n - 1 or labels[i + 1] != "I-" + code_type[1]:
                bilou_labels.append("L-" + code_type[1])
            else:
                bilou_labels.append(labels[i])
    return bilou_labels


def decode_bilou(labels: Sequence[str], tokens: Sequence[Token], doc: Doc) -> List[Span]:
    # a span is defined by the start and end of an entity.
    # We identify the start and end of entity from the labels(maybe improper)
    # so what defines the start of entity?
    # 1. Any label that starts with B- or U-
    # 2. I- or L- labels that are either sentence initials or follows
    #    "O" or a different type
    # What defines the end of entity?
    # 1. end of sentence 2. Before "O" or a different type
    n = len(labels)
    i = 0
    j = i + 1
    decoded = []
    while i < n and j <= n:
        code_entity_i = labels[i].split("-")
        code_i = code_entity_i[0]
        entity_prev = ""
        if i > 0:
            code_entity_prev = labels[i - 1].split("-")
            if code_entity_prev[0] != "O":
                entity_prev = code_entity_prev[1]

        # if i is a start position
        if code_i != "O" and (
            code_i == "B"
            or code_i == "U"
            or (
                i == 0
                or (i > 0 and (labels[i - 1] == "O" or entity_prev != code_entity_i[1]))
            )
        ):
            entity_i = code_entity_i[1]
            # if j is an end position
            if j == n or labels[j] == "O":
                decoded.append(Span(doc, i, j, entity_i))
                i = j
            else:
                code_entity_j = labels[j].split("-")
                code_j = code_entity_j[0]
                entity_j = code_entity_j[1]
                if entity_j != entity_i:
                    decoded.append(Span(doc, i, j, entity_i))
                    i = j
            j += 1
        else:
            i += 1
            j = i + 1
    return decoded

def binary_search(token_idx, start):
    left = 0
    right = len(token_idx)-1

    while (left < right):
        mid = (left + right)//2
        mid_val = token_idx[mid][0]
        if mid_val == start:
            return mid
        elif mid_val > start:
            right = mid
        else:
            left = mid+1
    if token_idx[left][0] < start:
        return left
    else:
        return left-1


def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:
    text = doc_json["text"]
    approver = doc_json["annotation_approver"]
    labels = doc_json["labels"]
    doc = nlp(text)
    # collect the token start character offset amd token offset
    token_idx = []

    for tok in doc:
        token_idx.append((tok.idx, tok.idx+len(tok)-1, tok.i))

    token_end_idx = token_idx[-1][1]
    token_start_idx = token_idx[0][0]
    entities = []

    #if len(labels) == 0 and approver == None:
    #    raise ValueError

    for label in labels:
        # binary search for the last tok.idx <= label[0]
        start = label[0]
        end = label[1]-1
        if end > token_end_idx or start < token_start_idx:
            raise ValueError
        idx = binary_search(token_idx, start)
        start_token = token_idx[idx]
        i = start_token[2]

        while start_token[1] < end:
            idx += 1
            start_token = token_idx[idx]
        j = idx

        entities.append(Span(doc, i, j+1, label[2]))

    #doc.ents = entities
    try:
        doc.ents = entities
    except Exception as e:
        doc.ents = []

    return doc

NLP = spacy.load("en_core_web_sm", disable=["ner"])
docs = []
label_mapping = {1 : "DISH", 2 : "INGRED",
                 3 : "QUAL", 4 : "BIZ", 5 : "SERV"}


features = [
        BiasFeature(),
        TokenFeature(),
        #PosFeature(),
        UppercaseFeature(),
        TitlecaseFeature(),
        #InitialTitlecaseFeature(),
        PunctuationFeature(),
        DigitFeature(),
        WordShapeFeature(),
        BrownClusterFeature(os.path.join(CUR_DIR, "rcv1.64M-c10240-p1.paths"),
            use_prefixes=True,
            prefixes=[8, 12, 16, 20]),
        WordVectorFeature(os.path.join(CUR_DIR, "wiki-news-300d-1M-subword.magnitude"), 2.0),
    ]

feature_extractor = WindowedTokenFeatureExtractor(
        features,
        1,
    )

crf = CRFsuiteEntityRecognizer(
    feature_extractor,
    BIOEncoder(),
)
