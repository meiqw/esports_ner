from pycrfsuite import Tagger
import os
from crf_predict.models import Doc, Ent
from CRF import CRFsuiteEntityRecognizer
from CRF import BiasFeature, TokenFeature, UppercaseFeature, TitlecaseFeature, DigitFeature, PunctuationFeature, WordShapeFeature
from CRF import WindowedTokenFeatureExtractor
from CRF import BILOUEncoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'CRF')
MODEL_DIR = os.path.join(MODEL_DIR, 'tmp.model')

tagger = Tagger()
tagger.open(MODEL_DIR)


def predict_ents(test_docs):
    entity_dict = {}

    features = [
        BiasFeature(),
        TokenFeature(),
        UppercaseFeature(),
        TitlecaseFeature(),
        # InitialTitlecaseFeature(),
        DigitFeature(),
        PunctuationFeature(),
        WordShapeFeature(),
        # LikelyAdjectiveFeature(),
        # AfterVerbFeature(),
        # WordVectorFeature(word_vector_file_path, 1.0),
        # BrownClusterFeature(
        #     brown_cluster_file_path,
        #     use_full_paths=False,
        #     use_prefixes=True,
        #     prefixes=[4, 6, 10, 20],
        # ),
    ]

    crf_model = CRFsuiteEntityRecognizer(
        WindowedTokenFeatureExtractor(features, 2), BILOUEncoder()
    )

    crf_model.tagger.open(MODEL_DIR)
    for doc in test_docs:
        """
        tokens = []
        pos_tags = []
        for sent in doc.sents:
            for token in sent:
                tokens.append(str(token))
                pos_tags.append(token.tag_)

        #features = train.feature_extractor.extract(tokens, pos_tags)
        features =
        labels = tagger.tag(features)

        entities = train.decode_bilou(labels, doc.sents, doc)
        doc.ents = entities
        """

        doc = crf_model(doc)

        for ent in doc.ents:
            if ent.label_ not in entity_dict:
                entity_dict[ent.label_] = set([ent.text])
            else:
                entity_dict[ent.label_].add(ent.text)


    return entity_dict, test_docs




