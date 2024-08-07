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


from datetime import datetime
from ontology import (
    load_ontology,
    get_labels_from_ontology,
    patterns_from_ontology,
    get_individuals_of_class,
    get_class_descendants,
    get_class_named,
    get_individuals_by_class,
)
from nlp import prepare_nlp, prepare_document, pair_stop_entities, get_named_problems
from nlp_ontology import (
    get_stop_ents,
    get_cases,
    get_cell_individual_by_cellname,
    extract_ontology_items,
    get_leaf_devices,
    match_problems,
    apply_function_devices,
    apply_function_products,
)
from graph import make_graph, make_distance_matrix

onto = load_ontology("sample.rdf")
labels = get_labels_from_ontology(onto)
patterns = patterns_from_ontology(labels)
nlp = prepare_nlp("pl_core_news_lg", patterns)
doc = prepare_document(nlp, "sample.txt")

stop_individuals = get_individuals_of_class(onto.individuals(), onto.Cells)
stop_ents = get_stop_ents(doc.ents, labels.items(), stop_individuals)
paired_stop_ents = pair_stop_entities(stop_ents)
doc_split_for_cases = get_cases(doc, paired_stop_ents)
problems = get_class_descendants(onto.Occurrance)
Meeting = get_class_named(onto, "Meeting")
meeting_individual = Meeting("meeting_" + datetime.now().strftime("%m_%d_%Y"))


for case_cell_name in doc_split_for_cases:
    # Assuming that case name is a single token
    assert len(case_cell_name) == 1
    new_case_individual = onto.Case()
    current_cell_individual = get_cell_individual_by_cellname(onto, case_cell_name)
    current_cell_individual.isDiscussed.append(new_case_individual)
    new_case_individual.refersTo.append(current_cell_individual)
    new_case_individual.isDiscussedDuring.append(meeting_individual)
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
        current_cell_individual, mentioned_devices_classes.keys()
    )
    leaf_devices = get_leaf_devices(mentioned_devices_classes)
    breakdowns = get_named_problems(problems_in_case, "Breakdown")
    items_graph = make_graph(doc_split_for_cases[case_cell_name])
    devices_breakdowns_distances_matrix = make_distance_matrix(
        leaf_devices.values(), items_graph, breakdowns
    )
    match_problems(
        devices_breakdowns_distances_matrix,
        new_case_individual,
        problems,
        apply_function_devices(leaf_devices, individuals_by_class),
    )

    defects = get_named_problems(problems_in_case, "Disqualifying_defect")
    products_defects_distance_matrix = make_distance_matrix(
        products_in_case_labels.values(), items_graph, defects
    )
    match_problems(
        products_defects_distance_matrix,
        new_case_individual,
        problems,
        apply_function_products(products_in_case_labels),
    )

filenameUpdate = "result.rdf"
onto.save(filenameUpdate)

# %%
