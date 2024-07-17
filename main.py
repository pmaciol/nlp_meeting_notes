#!/usr/bin/env python
# coding: utf-8

# In[1]
import pathlib

import spacy
import pathlib
import networkx as nx
import owlready2
from owlready2 import Ontology, ThingClass, Thing, EntityClass, ObjectPropertyClass
from spacy.tokens import Span, Doc
from collections import ChainMap
from datetime import datetime

from typing import Generator, List, Mapping

OntologyIndividualSuperclass = Thing
OntologyClassSuperclass = EntityClass
OntologyClass = ThingClass


# In[2]
def flatten_comprehension(list_of_lists):
    return [item for row in list_of_lists for item in row]


def add_to_dictionary(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]


def remove_keys_from_dict(dictionary, keys_to_remove):
    return {
        key: value for key, value in dictionary.items() if key not in keys_to_remove
    }


def find_item(dictionary, item):
    for key, value in dictionary.items():
        if item in value:
            return key


# Create a pipe that converts lemmas to lower case:
@spacy.language.Language.component("lower_case_lemmas")
def lower_case_lemmas(doc):
    for token in doc:
        token.lemma_ = token.lemma_.lower()
    return doc


# Create a custom sentencizer that sets sentence boundaries at newline characters:
@spacy.language.Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    for i, token in enumerate(doc[:-2]):
        doc[i + 1].is_sent_start = token.text == "\n"
    return doc


def prepare_nlp(spacy_baseline: str, patterns) -> spacy.Language:
    nlp = spacy.load(spacy_baseline)
    nlp.remove_pipe("ner")  # Removing unnecessary ner with basic categories
    nlp.add_pipe("custom_sentencizer", before="parser")
    nlp.add_pipe("lower_case_lemmas", after="tagger")
    ruler = nlp.add_pipe(
        "entity_ruler", config={"overwrite_ents": True, "validate": False}
    )
    ruler.add_patterns(patterns)
    return nlp


# Ontology functions:
def load_ontology(filename: pathlib.Path) -> Ontology:
    return owlready2.get_ontology(filename).load()


def get_labeled_items(
    collection: Generator,
) -> List[dict[str, OntologyIndividualSuperclass | OntologyClass]]:
    return [dict([(label, ind) for label in ind.isDefinedBy]) for ind in collection]


def get_labels_from_ontology(onto: Ontology) -> ChainMap:
    labels_from_individuals = get_labeled_items(onto.individuals())
    labels_from_classes = get_labeled_items(onto.classes())
    return ChainMap(*(labels_from_individuals + labels_from_classes))


def patterns_from_ontology(flattened_list: ChainMap):
    patterns = []
    for [pattern, item] in flattened_list.items():
        patters_splitted = pattern.split()
        pat_dict = [{"LEMMA": p_item} for p_item in patters_splitted]
        patterns.append({"label": item.name, "pattern": pat_dict})
    return patterns


#########################################################################
def prepare_document(language: spacy.Language, file_path: pathlib.Path) -> Doc:
    return language(pathlib.Path(file_path).read_text(encoding="utf-8"))


def get_individuals_of_class(
    ontoClass: OntologyClass,
) -> List[OntologyIndividualSuperclass]:
    return [x for x in onto.individuals() if x.__class__ == ontoClass]


def get_product_names(product_individuals) -> List[str]:
    return [x.name for x in product_individuals]


def get_class_named(onto: Ontology, name: str) -> OntologyClass:
    return [x for x in onto.classes() if x.name == name][0]


def get_class_descendants(root: OntologyClass) -> List[OntologyClass]:
    return [x for x in root.descendants()]


def get_obj_propertied(onto: Ontology, name: str) -> ObjectPropertyClass:
    return [x for x in onto.object_properties() if x.name == name][0]


def all_stop_ents(
    entities: tuple[Span],
    dict_all_items: Mapping[str, EntityClass | Thing],
    stop_individuals: List[Thing],
) -> List[Span]:
    return [ent for ent in entities if dict_all_items[ent.lemma_] in stop_individuals]


def get_stop_ents(
    entities: tuple[Span],
    dict_all_items: Mapping[str, EntityClass | Thing],
    stop_individuals: List[Thing],
) -> List[Span]:
    stop_ents = all_stop_ents(entities, dict_all_items, stop_individuals)
    filtered_stop_ents = [stop_ents[0]]
    for se in stop_ents[1::]:
        if dict_all_items[se.lemma_] != dict_all_items[filtered_stop_ents[-1].lemma_]:
            filtered_stop_ents.append(se)
    return filtered_stop_ents


def pair_stop_entities(filtered_stop_ents: List[Span]) -> List[tuple[Span, Span]]:
    paired_stop_ents = [
        (
            filtered_stop_ents[i],
            filtered_stop_ents[i + 1] if i < len(filtered_stop_ents) - 1 else None,
        )
        for i in range(len(filtered_stop_ents))
    ]
    return paired_stop_ents


def get_cases(doc: Doc, paired_stop_ents: List[tuple[Span, Span]]) -> dict[Span, Span]:
    doc_split_for_cases = {}
    for case_token in paired_stop_ents:
        current_cell_label, next_cell_label = case_token
        case_text_start = doc[current_cell_label.sent.start]
        case_text_end = (
            doc[next_cell_label.sent.start] if next_cell_label != None else doc[-1]
        )
        doc_split_for_cases[current_cell_label] = Span(
            doc, case_text_start.i, case_text_end.i
        )
    return doc_split_for_cases


def extract_ontology_items(
    case_text: Span, onto: Ontology, case_item: OntologyClass
) -> tuple[
    dict[str, Span],
    dict[OntologyClass, list[OntologyClass]],
    dict[OntologyClass, list[Span]],
    dict[OntologyClass, list[Span]],
]:
    devices = get_class_descendants(onto.Device)
    devices_by_name = {item.name: item for item in devices}
    product_names = get_product_names(get_individuals_of_class(onto.Product))
    problems = get_class_descendants(onto.Occurrance)

    problems_in_case = {}
    mentioned_devices_classes = {}
    products_in_case = {}
    products_in_case_labels = {}
    for token in case_text.ents:
        if token.label_ in [x.name for x in problems]:
            add_to_dictionary(problems_in_case, token.label_, token)
        if token.label_ in [x.name for x in devices]:
            add_to_dictionary(
                mentioned_devices_classes, devices_by_name[token.label_], token
            )
        if token.label_ in product_names:
            product = [x for x in onto.individuals() if x.name == token.label_]
            add_to_dictionary(products_in_case, product[0], case_item)
            add_to_dictionary(products_in_case_labels, product[0], token)
    return (
        problems_in_case,
        mentioned_devices_classes,
        products_in_case,
        products_in_case_labels,
    )


def get_individuals_by_class(mentioned_devices_classes, all_devices_classes):
    individuals_by_class = {}
    for device_class in mentioned_devices_classes.keys():
        if device_class not in all_devices_classes:
            device_individual = device_class()
            current_cell_individual.IsEquippedWith.append(device_individual)
            device_individual.IsEquipmentIn = current_cell_individual
            print("adding ", device_individual)
        else:
            device_individual = next(
                dev_ind
                for dev_ind in current_cell_individual.IsEquippedWith
                if device_class in dev_ind.is_a
            )
            print("using ", device_individual)
        individuals_by_class[device_class] = device_individual
    print("INDIVIDUALS_BY_CLASS", individuals_by_class)
    return individuals_by_class


def get_all_parents(mentioned_devices_classes):
    all_parents = set()
    for dev_class in mentioned_devices_classes.keys():
        tmp = dev_class.ancestors()
        tmp.remove(dev_class)
        all_parents.update(tmp)
    return all_parents


#######################################################################################
def make_graph(cases, case):
    edges = []
    for token in cases[case]:
        for child in token.children:
            edges.append(
                (
                    "{0}-{1}".format(token.lower_, token.i),
                    "{0}-{1}".format(child.lower_, child.i),
                )
            )
    return nx.Graph(edges)


def make_distance_matrix(leaf_devices, graph, breakdowns):
    matrix = {}
    for l in leaf_devices.values():
        for s_d in l:
            matrix[s_d] = {}
            for s_b in breakdowns:
                s_d_name = "{0}-{1}".format(s_d[0].lower_, s_d[0].i)
                s_b_name = "{0}-{1}".format(s_b[0].lower_, s_b[0].i)

                local_dist = 100
                try:
                    local_dist = nx.shortest_path_length(
                        graph, source=s_d_name, target=s_b_name
                    )
                except:
                    local_dist = 100 + abs(s_d[0].i - s_b[0].i)
                matrix[s_d][s_b] = abs(local_dist)
    return matrix


# same as above?
def make_distance_matrix_defects(products_in_case_labels, graph, defects):
    matrix = {}
    for l in products_in_case_labels.values():
        for s_d in l:
            matrix[s_d] = {}
            for s_b in defects:
                s_d_name = "{0}-{1}".format(s_d[0].lower_, s_d[0].i)
                s_b_name = "{0}-{1}".format(s_b[0].lower_, s_b[0].i)

                local_dist = 100
                try:
                    local_dist = nx.shortest_path_length(
                        graph, source=s_d_name, target=s_b_name
                    )
                except:
                    local_dist = 100 + abs(s_d[0].i - s_b[0].i)
                matrix[s_d][s_b] = abs(local_dist)
    return matrix


def add_pboblems_to_devices(
    matrix, leaf_devices, case_item, individuals_by_class, problems_by_name
):
    dev_min = ""
    bre_min = ""
    dist = 0
    if (len(matrix.keys()) > 0) and (len(matrix[next(iter(matrix))].keys()) > 0):
        while dist < 1000:
            dist = 1000
            for s_d in matrix.keys():
                for s_b in matrix[s_d].keys():
                    if matrix[s_d][s_b] < dist:
                        dist = matrix[s_d][s_b]
                        dev_min = s_d
                        bre_min = s_b
            for s_d_internal in matrix[dev_min]:
                matrix[dev_min][s_d_internal] = 1000
            for s_d_other in matrix.keys():
                if str(s_d_other) == str(dev_min):
                    for s_d_internal in matrix[dev_min]:
                        matrix[s_d_other][s_d_internal] = 1000
            for s_b_internal in matrix:
                matrix[s_b_internal][bre_min] = 1000
            if dist < 1000:
                item_found = find_item(leaf_devices, dev_min)
                problem = problems_by_name[bre_min.ents[0].label_]()
                problem.isDiscussed.append(case_item)
                problem.occuredFor.append(individuals_by_class[item_found])
                print(
                    "occurance added ",
                    problem,
                    "happend to",
                    individuals_by_class[item_found],
                )


def add_pboblems_to_devices_another(
    matrix, products_in_case_labels, case_item, problems_by_name
):
    dev_min = ""
    bre_min = ""
    dist = 0
    if (len(matrix.keys()) > 0) and (len(matrix[next(iter(matrix))].keys()) > 0):
        while dist < 1000:
            dist = 1000
            for s_d in matrix.keys():
                for s_b in matrix[s_d].keys():
                    if matrix[s_d][s_b] < dist:
                        dist = matrix[s_d][s_b]
                        dev_min = s_d
                        bre_min = s_b
            for s_d_internal in matrix[dev_min]:
                matrix[dev_min][s_d_internal] = 1000
            for s_d_other in matrix.keys():
                if str(s_d_other) == str(dev_min):
                    for s_d_internal in matrix[dev_min]:
                        matrix[s_d_other][s_d_internal] = 1000
            for s_b_internal in matrix:
                matrix[s_b_internal][bre_min] = 1000
            if dist < 1000:
                item_found = find_item(products_in_case_labels, dev_min)
                problem = problems_by_name[bre_min.ents[0].label_]()
                problem.isDiscussed.append(case_item)
                print(
                    "PROBLEM ",
                    problem,
                    " DEFECT ",
                    products_in_case_labels[item_found],
                    " ITEM ",
                    item_found,
                )
                problem.occuredFor.append(item_found)
                # print('PROBLEM !!!!!!!!!!!!!!!!!!!!!', problem, item_found)


def get_cell_individual_by_cellname(onto: Ontology, cell_name: Span):
    # print("cell_name", cell_name)
    for x in onto.individuals():
        if cell_name.lemma_ in x.isDefinedBy:
            return x
    return None


def get_devices_in_cell(
    current_cell_individual: OntologyIndividualSuperclass,
) -> set[OntologyClass]:
    return set(
        flatten_comprehension(
            [x.is_a for x in list(current_cell_individual.IsEquippedWith)]
        )
    )


def get_leaf_devices(mentioned_devices_classes):
    """'Leaf device' is a device mentioned in the cese text, that is not a parent of any other device mentioned in the case text."""
    all_parents = get_all_parents(mentioned_devices_classes)
    return remove_keys_from_dict(mentioned_devices_classes, all_parents)


#######################################################################################


onto = load_ontology("sample.rdf")
labels = get_labels_from_ontology(onto)
patterns = patterns_from_ontology(labels)
nlp = prepare_nlp("pl_core_news_lg", patterns)
doc = prepare_document(nlp, "sample.txt")

dict_all_items = dict(list(labels.items()))
stop_individuals = get_individuals_of_class(onto.Cells)
stop_ents = get_stop_ents(doc.ents, dict_all_items, stop_individuals)
paired_stop_ents = pair_stop_entities(stop_ents)
doc_split_for_cases = get_cases(doc, paired_stop_ents)
problems = get_class_descendants(onto.Occurrance)
Meeting = get_class_named(onto, "Meeting")
meeting_individual = Meeting("meeting_" + datetime.now().strftime("%m_%d_%Y"))
problems_by_name = {item.name: item for item in problems}

for case_cell_name in doc_split_for_cases:
    # Assuming that case name is a single token
    assert len(case_cell_name) == 1
    new_case_individual = onto.Case()
    current_cell_individual = get_cell_individual_by_cellname(onto, case_cell_name)
    current_cell_individual.isDiscussed.append(new_case_individual)
    new_case_individual.refersTo.append(current_cell_individual)
    new_case_individual.isDiscussedDuring.append(meeting_individual)

    all_cell_devices_classes = get_devices_in_cell(current_cell_individual)
    (
        problems_in_case,
        mentioned_devices_classes,
        products_in_case,
        products_in_case_labels,
    ) = extract_ontology_items(
        doc_split_for_cases[case_cell_name],
        onto,
        new_case_individual,
    )

    for product in products_in_case:
        product.isDiscussed.append(products_in_case[product][0])

    
    individuals_by_class = get_individuals_by_class(
        mentioned_devices_classes, all_cell_devices_classes
    )
    leaf_devices = get_leaf_devices(mentioned_devices_classes)
    breakdowns = (
        problems_in_case["Breakdown"] if "Breakdown" in problems_in_case.keys() else []
    )
    graph = make_graph(doc_split_for_cases, case_cell_name)
    matrix = make_distance_matrix(leaf_devices, graph, breakdowns)
    add_pboblems_to_devices(
        matrix,
        leaf_devices,
        new_case_individual,
        individuals_by_class,
        problems_by_name,
    )

    defects = (
        problems_in_case["Disqualifying_defect"]
        if "Disqualifying_defect" in problems_in_case.keys()
        else []
    )
    matrix = make_distance_matrix_defects(products_in_case_labels, graph, defects)
    add_pboblems_to_devices_another(
        matrix, products_in_case_labels, new_case_individual, problems_by_name
    )


filenameUpdate = "result.rdf"
onto.save(filenameUpdate)

# %%
