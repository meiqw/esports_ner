import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE','esports_ner.settings')

import django
django.setup()

import json

from .train_crf import train
#from train import ingest_json_document, NLP

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "CRF", "data")

from crf_predict.models import Doc, Ent, TestDoc


def populate_test(doc, text, title, game):
    # populate Doc table
    d = TestDoc.objects.get_or_create(title=title, text=text, game=game)[0]
    d.save()


def populate_train(doc, text, title, game):
    # populate Doc table
    d = Doc.objects.get_or_create(title=title, text=text, game=game)[0]
    d.save()

    #populate Ent table
    for ent in doc.ents:
        e = Ent.objects.get_or_create(doc=d, label=ent.label_, text=ent.text,
                                      start=ent.start, end=ent.end)[0]
        e.save()



if __name__=="__main__":
    with open(os.path.join(DATA_DIR, "corpus_train.jsonl")) as json_file:
        id = 0
        for review in json_file:
            doc_json = json.loads(review)
            doc = train.ingest_json_document(doc_json, train.NLP)
            text = doc_json['text']
            title = text[0:30] + "..."
            game = "unk"
            populate_train(doc, text, title, game)
            print(id)
            id += 1

    with open(os.path.join(DATA_DIR, "corpus_dev.jsonl")) as json_file:
        id = 0
        for review in json_file:
            doc_json = json.loads(review)
            doc = train.ingest_json_document(doc_json, train.NLP)
            text = doc_json['text']
            title = text[0:30] + "..."
            game = "unk"
            populate_test(doc, text, title, game)
            print(id)
            id += 1




