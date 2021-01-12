from django.shortcuts import render
from django.http import HttpResponse
from .util import retrieve_test_docs_from_url, retrieve_test_docs_from_local, \
    add_href_to_entities, generate_sent_docs, entity_linking, display_json
from .predict import predict_ents
from spacy import displacy
from crf_predict.models import Doc, Ent, TestDoc
import random
from Entity_Linking import Dota2_Knowledge_Base
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR,  "Entity_Linking")

dota2_kb = Dota2_Knowledge_Base(
    player_file_name=os.path.join(DATA_DIR, "dota2_players.json"),
    team_file_name=os.path.join(DATA_DIR, "dota2_teams.json"),
    tournament_file_name=os.path.join(DATA_DIR, "dota2_tournaments.json")
)

# count the number of test documents
# n = Doc.objects.all().count()

colors = {"AVATAR" : "tomato",
          "GAME" : "cornflowerblue",
          "ORG" : "limegreen",
          "PLAYER" : "gold",
          "SPONS" : "orange",
          "TOURN" : "aqua"
          }
homepage_docs = {}

n = 5
pks = list(range(0, n))
random.shuffle(pks)

for pk in pks:
    doc = TestDoc.objects.all()[pk]
    homepage_docs[doc.title] = doc.text

test_docs = []
# Create your views here.
def index(request):

    if request.method == 'POST':
        print(request.POST.get('url'))
        test_docs = retrieve_test_docs_from_url(request.POST.get('url'))
        entity_dict, test_docs = predict_ents(test_docs)
        html = displacy.render(test_docs, style="ent", options={'colors': colors}, page=True)
        add_href_to_entities(html)
        return render(request, 'crf_predict/tag.html', context={})

    return render(request, 'crf_predict/homepage.html', context={'doc_dict':homepage_docs})

def linking(request, entity):

    ent_name, ent_label = entity.split("&amp;")

    # query the database
    matched_ents = Ent.objects.all().filter(text=ent_name, label=ent_label)
    # retrieve the sentences that contain this entity
    sents = generate_sent_docs(matched_ents)
    entity_links = entity_linking(ent_name, ent_label, dota2_kb)

    html = displacy.render(sents, style="ent", options={'colors': colors}, page=True)
    html = display_json(html, entity_links)
    return HttpResponse(html)

def tagging(request, doc_title):
    test_docs = retrieve_test_docs_from_local(doc_title, homepage_docs)
    entity_dict, test_docs = predict_ents(test_docs)
    html = displacy.render(test_docs, style="ent", options={'colors': colors}, page=True)
    add_href_to_entities(html)
    return render(request, 'crf_predict/tag.html', context={})
