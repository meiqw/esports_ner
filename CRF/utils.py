import re
import sys
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from typing import Sequence, Dict, List, NamedTuple, Tuple, Counter, Optional, Mapping
from decimal import ROUND_HALF_UP, Context
from collections import defaultdict

import regex
import spacy
from spacy.language import Language
from spacy.tokens import Token, Doc, Span

UPPERCASE_RE = regex.compile(r"[\p{Lu}\p{Lt}]")
LOWERCASE_RE = regex.compile(r"\p{Ll}")
DIGIT_RE = re.compile(r"\d")

PUNC_REPEAT_RE = regex.compile(r"\p{P}+")

class PRF1(NamedTuple):
    precision: float
    recall: float
    f1: float

def decode_bilou(labels: Sequence[str], tokens: Sequence[Token], doc: Doc) -> List[Span]:
    outside = True
    current_type = None
    start_index = 0
    res = []
    for label, token in zip(labels, tokens):
        if "-" in label:
            tag, entity_type = label.split("-", 2)
        else:
            tag, entity_type = "O", None
        if outside and tag != "O":
            current_type = entity_type
            outside = False
            start_index = token.i
        elif not outside:
            if tag == "O":
                outside = True
                res.append(
                    Span(doc=doc, start=start_index, end=token.i, label=current_type)
                )
                start_index = token.i
            elif tag == "B" or tag == "U" or entity_type != current_type:
                res.append(
                    Span(doc=doc, start=start_index, end=token.i, label=current_type)
                )
                current_type = entity_type
                start_index = token.i
    if not outside:
        outside = True
        res.append(
            Span(doc=doc, start=start_index, end=tokens[-1].i + 1, label=current_type)
        )
    return res

def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:
    doc = nlp(doc_json["text"])
    token_idx_list = [token.idx for token in doc]
    annotations = sorted(doc_json["labels"], key=lambda x: x[0])
    if len(annotations) == 0 and doc_json["annotation_approver"] is None:
        # raise ValueError
        pass
    first = True
    ents = []
    for annotation in annotations:
        character_start, character_end, label = annotation
        ent_start = bisect_right(token_idx_list, character_start) - 1
        ent_end = bisect_left(token_idx_list, character_end)
        if first:
            first = False
        else:
            ent_start = max(ent_start, ents[-1].end)
            # if ent_start >= ent_end:
            #     raise ValueError
        if ent_start < ent_end:
            ents.append(Span(doc, ent_start, ent_end, label))
    doc.ents = ents
    return doc

def span_prf1_type_map(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:
    res_dict = {}
    correct_count, test_count, ref_count = (
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    )
    correct_count_all, test_count_all, ref_count_all = 0, 0, 0  # For untyped calculation
    for ref_doc, test_doc in zip(reference_docs, test_docs):
        ref_ents = sorted(ref_doc.ents, key=lambda ent: ent.start)
        test_ents = sorted(test_doc.ents, key=lambda ent: ent.start)
        ref_i, test_i = 0, 0
        while ref_i < len(ref_ents) and test_i < len(test_ents):
            cur_ref, cur_test = ref_ents[ref_i], test_ents[test_i]
            cur_ref_label, cur_test_label = cur_ref.label_, cur_test.label_
            if type_map is not None:
                if cur_ref_label in type_map:
                    cur_ref_label = type_map[cur_ref_label]
                if cur_test_label in type_map:
                    cur_test_label = type_map[cur_test_label]
            if cur_ref.start < cur_test.start:
                ref_count[cur_ref_label] += 1
                ref_i += 1
            elif cur_ref.start == cur_test.start:
                if (
                    cur_ref_label == cur_test_label
                    and cur_ref.end == cur_test.end
                    and cur_ref_label == cur_test_label
                ):
                    correct_count[cur_test_label] += 1
                ref_count[cur_ref_label] += 1
                test_count[cur_test_label] += 1
                ref_i += 1
                test_i += 1
            else:
                test_count[cur_test_label] += 1
                test_i += 1
        while ref_i < len(ref_ents):
            cur_label = ref_ents[ref_i].label_
            if type_map is not None and cur_label in type_map:
                cur_label = type_map[cur_label]
            ref_count[cur_label] += 1
            ref_i += 1
        while test_i < len(test_ents):
            cur_label = test_ents[test_i].label_
            if type_map is not None and cur_label in type_map:
                cur_label = type_map[cur_label]
            test_count[cur_label] += 1
            test_i += 1

    correct_count_all = sum(correct_count.values())
    for label in set(test_count.keys()).union(ref_count.keys()):
        precision = (
            0.0
            if test_count[label] == 0
            else correct_count[label] / test_count[label]
        )
        recall = (
            0.0 if ref_count[label] == 0 else correct_count[label] / ref_count[label]
        )
        f1 = (
            0.0
            if precision + recall == 0.0
            else 2 * precision * recall / (precision + recall)
        )
        res_dict[label] = PRF1(precision, recall, f1)
    # all
    test_count_all = sum(test_count.values())
    ref_count_all = sum(ref_count.values())
    precision_all = 0.0 if test_count_all == 0 else correct_count_all / test_count_all
    recall_all = 0.0 if ref_count_all == 0 else correct_count_all / ref_count_all
    f1_all = (
        0.0
        if precision_all + recall_all == 0.0
        else 2 * precision_all * recall_all / (precision_all + recall_all)
    )
    res_dict[""] = PRF1(precision_all, recall_all, f1_all)

    return res_dict

def ent_mention(doc: Doc, count: Dict):
    ents = sorted(doc.ents, key=lambda ent: ent.start)
    for ent in ents:
        label = ent.label_
        text = doc[ent.start : ent.end]
        yield label, text


def evaluate_and_print(ref_doc: Sequence[Doc], test_doc: Sequence[Doc], type_map: Optional[Mapping[str, str]] = None):
    print("Type\tPrec\tRec\tF1", file=sys.stderr)
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    scores = span_prf1_type_map(ref_doc, test_doc, type_map)
    for ent_type, score in sorted(scores.items()):
        if ent_type == "":
            ent_type = "ALL"

        fields = [ent_type] + [
            str(rounder.create_decimal_from_float(num * 100)) for num in score
        ]
        print("\t".join(fields), file=sys.stderr)