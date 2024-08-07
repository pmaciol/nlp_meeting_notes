# Copyright (C) 2024 Piotr Macioł
# 
# nlp_meeting_notes is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2.1 of the License, or
# (at your option) any later version.
# 
# nlp_meeting_notes is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with nlp_meeting_notes. If not, see <https://www.gnu.org/licenses/>.




from spacy.tokens import Span, Doc
from typing import List, Mapping
from ontology import (
    OntologyClassSuperclass,
    OntologyIndividualSuperclass,
    OntologyClass,
    Ontology,
    get_class_descendants,
    get_product_names,
    get_individuals_of_class,
    get_all_parents,
)
from utils import add_to_dictionary, set_column_to_1000, set_row_to_1000, find_item


def all_stop_ents(
    entities: tuple[Span],
    dict_all_items: Mapping[
        str, OntologyIndividualSuperclass | OntologyClassSuperclass
    ],
    stop_individuals: List[OntologyIndividualSuperclass],
) -> List[Span]:
    return [ent for ent in entities if dict_all_items[ent.lemma_] in stop_individuals]


def get_stop_ents(
    entities: tuple[Span],
    labels: Mapping[str, OntologyClassSuperclass | OntologyIndividualSuperclass],
    stop_individuals: List[OntologyIndividualSuperclass],
) -> List[Span]:
    dict_all_items = dict(list(labels))
    stop_ents = all_stop_ents(entities, dict_all_items, stop_individuals)
    filtered_stop_ents = [stop_ents[0]]
    for se in stop_ents[1::]:
        if dict_all_items[se.lemma_] != dict_all_items[filtered_stop_ents[-1].lemma_]:
            filtered_stop_ents.append(se)
    return filtered_stop_ents


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


def get_cell_individual_by_cellname(onto: Ontology, cell_name: Span):
    for x in onto.individuals():
        if cell_name.lemma_ in x.isDefinedBy:
            return x
    return None


def extract_ontology_items(
    case_text: Span, onto: Ontology, case_item: OntologyClass
) -> tuple[
    dict[str, Span],
    dict[OntologyClass, list[Span]],
    dict[OntologyIndividualSuperclass, list[OntologyIndividualSuperclass]],
    dict[OntologyIndividualSuperclass, list[Span]],
]:
    devices = get_class_descendants(onto.Device)
    devices_by_name = {item.name: item for item in devices}
    product_names = get_product_names(
        get_individuals_of_class(onto.individuals(), onto.Product)
    )
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


def remove_keys_from_dict(
    dictionary: Mapping[OntologyClass, list[Span]],
    keys_to_remove: set[OntologyClass],
) -> dict[OntologyClass, list[Span]]:
    return {
        key: value for key, value in dictionary.items() if key not in keys_to_remove
    }


def get_leaf_devices(
    mentioned_devices_classes: Mapping[OntologyClass, list[Span]]
) -> dict[OntologyClass, list[Span]]:
    """'Leaf device' is a device mentioned in the cese text, that is not a parent of any other device mentioned in the case text."""
    all_parents = get_all_parents(mentioned_devices_classes.keys())
    return remove_keys_from_dict(mentioned_devices_classes, all_parents)


def add_problems_to_devices(
    devices_breakdowns_distances_matrix: Mapping[Span, Mapping[Span, int]],
    leaf_devices: Mapping[OntologyClass, List[Span]],
    case_item: OntologyIndividualSuperclass,
    individuals_by_class: Mapping[OntologyClass, OntologyIndividualSuperclass],
    problems,
):
    problems_by_name = {item.name: item for item in problems}
    dev_min = ""
    problem_min = ""
    dist = 0
    if (len(devices_breakdowns_distances_matrix.keys()) > 0) and (
        len(
            devices_breakdowns_distances_matrix[
                next(iter(devices_breakdowns_distances_matrix))
            ].keys()
        )
        > 0
    ):
        while dist < 1000:
            dist = 1000
            for device in devices_breakdowns_distances_matrix.keys():
                for problems in devices_breakdowns_distances_matrix[device].keys():
                    if devices_breakdowns_distances_matrix[device][problems] < dist:
                        dist = devices_breakdowns_distances_matrix[device][problems]
                        dev_min = device
                        problem_min = problems
            set_row_to_1000(devices_breakdowns_distances_matrix, dev_min)
            for device_internal in devices_breakdowns_distances_matrix.keys():
                if str(device_internal) == str(dev_min):
                    for problem_in_dev_min_row in devices_breakdowns_distances_matrix[
                        dev_min
                    ]:
                        devices_breakdowns_distances_matrix[device_internal][
                            problem_in_dev_min_row
                        ] = 1000
            set_column_to_1000(devices_breakdowns_distances_matrix, problem_min)
            if dist < 1000:
                item_found = find_item(leaf_devices, dev_min)
                problem = problems_by_name[problem_min.ents[0].label_]()
                problem.isDiscussed.append(case_item)
                problem.occuredFor.append(individuals_by_class[item_found])


def add_problems_to_devices_another(
    products_defects_distance_matrix: Mapping[Span, Mapping[Span, int]],
    products_in_case_labels: Mapping[OntologyIndividualSuperclass, List[Span]],
    case_item: OntologyIndividualSuperclass,
    problems,
):
    problems_by_name = {item.name: item for item in problems}
    dev_min = ""
    bre_min = ""
    dist = 0
    if (len(products_defects_distance_matrix.keys()) > 0) and (
        len(
            products_defects_distance_matrix[
                next(iter(products_defects_distance_matrix))
            ].keys()
        )
        > 0
    ):
        while dist < 1000:
            dist = 1000
            for s_d in products_defects_distance_matrix.keys():
                for s_b in products_defects_distance_matrix[s_d].keys():
                    if products_defects_distance_matrix[s_d][s_b] < dist:
                        dist = products_defects_distance_matrix[s_d][s_b]
                        dev_min = s_d
                        bre_min = s_b
            set_row_to_1000(products_defects_distance_matrix, dev_min)
            for s_d_other in products_defects_distance_matrix.keys():
                if str(s_d_other) == str(dev_min):
                    for s_d_internal in products_defects_distance_matrix[dev_min]:
                        products_defects_distance_matrix[s_d_other][s_d_internal] = 1000
            set_column_to_1000(products_defects_distance_matrix, bre_min)

            if dist < 1000:
                item_found = find_item(products_in_case_labels, dev_min)
                problem = problems_by_name[bre_min.ents[0].label_]()
                problem.isDiscussed.append(case_item)
                problem.occuredFor.append(item_found)


def apply_function_products(products_in_case_labels):
    def apply_on_products(dev_min, problem_min, problems, case_item):
        item_found = find_item(products_in_case_labels, dev_min)
        problem = {item.name: item for item in problems}[problem_min.ents[0].label_]()
        problem.isDiscussed.append(case_item)
        problem.occuredFor.append(item_found)
    return apply_on_products


def apply_function_devices(leaf_devices, individuals_by_class):
    def apply_on_devices(dev_min, problem_min, problems, case_item):
        item_found = find_item(leaf_devices, dev_min)
        problem = {item.name: item for item in problems}[problem_min.ents[0].label_]()
        problem.isDiscussed.append(case_item)
        problem.occuredFor.append(individuals_by_class[item_found])

    return apply_on_devices


def match_problems(
    distances_matrix: Mapping[Span, Mapping[Span, int]],
    case_item: OntologyIndividualSuperclass,
    problems,
    apply_function,
):
    dev_min = ""
    problem_min = ""
    dist = 0
    if (len(distances_matrix.keys()) > 0) and (
        len(distances_matrix[next(iter(distances_matrix))].keys()) > 0
    ):
        while dist < 1000:
            dist = 1000
            for row in distances_matrix.keys():
                for col in distances_matrix[row].keys():
                    if distances_matrix[row][col] < dist:
                        dist = distances_matrix[row][col]
                        dev_min = row
                        problem_min = col
            set_row_to_1000(distances_matrix, dev_min)
            for device_internal in distances_matrix.keys():
                if str(device_internal) == str(dev_min):
                    for problem_in_dev_min_row in distances_matrix[dev_min]:
                        distances_matrix[device_internal][problem_in_dev_min_row] = 1000
            set_column_to_1000(distances_matrix, problem_min)
            if dist < 1000:
                apply_function(dev_min, problem_min, problems, case_item)
