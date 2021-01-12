from spacy.tokens import Doc, Span, Token
from .feature_extractors import WindowedTokenFeatureExtractor
from abc import ABC, abstractmethod
from typing import Mapping, Sequence, Dict, Optional, List, Iterable

from pycrfsuite import Trainer, Tagger

from .utils import decode_bilou

class EntityEncoder(ABC):
    @abstractmethod
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        raise NotImplementedError

class CRFsuiteEntityRecognizer:
    def __init__(
        self, feature_extractor: WindowedTokenFeatureExtractor, encoder: EntityEncoder
    ) -> None:
        self.feature_extractor = feature_extractor
        self._encoder = encoder
        self.tagger = Tagger()

    @property
    def encoder(self) -> EntityEncoder:
        return self._encoder

    def train(self, docs: Iterable[Doc], algorithm: str, params: dict, path: str) -> None:
        trainer = Trainer(algorithm=algorithm, params=params, verbose=False)
        for doc in docs:
            for sentence in doc.sents:
                tokens = list(sentence)
                features = self.feature_extractor.extract(
                    [str(token) for token in tokens]
                )
                labels = self.encoder.encode(tokens)
                trainer.append(features, labels)
        trainer.train(path)
        self.tagger.close()
        self.tagger.open(path)

    def __call__(self, doc: Doc) -> Doc:
        doc_ent = []
        for sentence in doc.sents:
            tokens = list(sentence)
            labels = self.predict_labels([str(token) for token in tokens])
            entities = decode_bilou(labels, tokens, doc)
            # print("tokens:%s\nfeatures:%s\nlabels:%s\nentities:%s\n"%(str(tokens), str(features), str(labels), str(entities)))
            for entity in entities:
                doc_ent.append(entity)
        doc.ents = doc_ent
        return doc

    def predict_labels(self, tokens: Sequence[str]) -> List[str]:
        features = self.feature_extractor.extract(tokens)
        return self.tagger.tag(features)


class BILOUEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        res = []
        for i, token in enumerate(tokens):
            ent_tag, ent_type = token.ent_iob_, token.ent_type_
            if ent_tag == "":
                res.append("O")
            elif ent_tag == "B":
                res.append("B-" + ent_type)
            elif ent_tag == "I":
                if i + 1 < len(tokens) and tokens[i + 1].ent_iob_ == "I":
                    res.append("I-" + ent_type)
                else:
                    res.append("L-" + ent_type)
        return res

class BIOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        res = []
        for token in tokens:
            ent_tag, ent_type = token.ent_iob_, token.ent_type_
            if ent_tag == "":
                res.append("O")
            else:
                res.append(ent_tag + "-" + ent_type)
        return res