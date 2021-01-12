import requests
from bs4 import BeautifulSoup
import spacy
from spacy.tokens import Span
import os
import json

NLP = spacy.load("en_core_web_sm", disable=["ner"])
CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CUR_DIR, "train_crf")

TEMPLATE_DIR = os.path.join(CUR_DIR, 'templates')
TEMPLATE_DIR = os.path.join(TEMPLATE_DIR, 'crf_predict')

def retrieve_test_docs_from_url(url):
    # Connect to the URL
    response = requests.get(url)
    # Parse HTML and save to BeautifulSoup object¶
    soup = BeautifulSoup(response.text, "html.parser")

    test_docs = []
    paragraphs = soup.findAll('p')
    for p in paragraphs:
        doc = NLP(p.text)
        #print(doc)
        test_docs.append(doc)
    return test_docs

def retrieve_test_docs_from_local(title, docs):
    test_docs = []
    text = docs[title]
    doc = NLP(text)
    test_docs.append(doc)
    return test_docs


def add_href_to_entities(html):
    # change the title
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.title
    title_tag.string = "Named Entity Tagging"

    for mark_tag in soup.find_all('mark'):
        ent_name = mark_tag.contents[0].strip()
        ent_label = mark_tag.contents[1].string
        url_tag = "{% url \'linking\' entity=" + \
                  "\'" + ent_name + "&" + ent_label + \
                  "\'" + "%}"
        new_tag = soup.new_tag("a", href=url_tag)
        new_tag.string = ent_name

        mark_tag.contents[0].replaceWith(new_tag)

    html = soup.prettify()
    html_file = open(os.path.join(TEMPLATE_DIR, "tag.html"), 'w')
    html_file.write(html)
    html_file.close()


def binary_search(token_idx, start):
    left = 0
    right = len(token_idx) - 1

    while (left < right):
        mid = (left + right) // 2
        mid_val = token_idx[mid]
        if mid_val == start:
            return mid
        elif mid_val > start:
            right = mid
        else:
            left = mid + 1
    if token_idx[left] < start:
        return left
    else:
        return left - 1


# each entity of its type is mapped to a document in the database
# find sentences each document
def generate_sent_docs(matched_ents):
    sents = []
    added = set()
    print(len(matched_ents))
    for ent in matched_ents:
        doc_obj = ent.doc
        print(doc_obj.title)
        print(doc_obj.text)
        if doc_obj.title not in added:
            doc = NLP(doc_obj.text)
            start = ent.start
            end = ent.end
            # retrieve the sentences from the doc that contains this entity
            start_idx = []
            for sent in doc.sents:
                start_idx.append(sent.start)

            sent_idx = binary_search(start_idx, start)
            sent_start = start_idx[sent_idx]

            sent = list(doc.sents)[sent_idx].as_doc()
            start -= sent_start
            end -= sent_start
            sent.ents = [Span(sent, start, end, ent.label)]
            sents.append(sent)
            added.add(doc_obj.title)
    print(len(sents))
    return sents


