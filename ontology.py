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


import pathlib
from owlready2 import (
    Ontology,
    ThingClass,
    Thing,
    EntityClass,
    get_ontology,
)
from typing import Generator, List, Sequence
from collections import ChainMap
from utils import flatten_comprehension

OntologyIndividualSuperclass = Thing
OntologyClassSuperclass = EntityClass
OntologyClass = ThingClass


def load_ontology(filename: pathlib.Path) -> Ontology:
    return get_ontology(filename).load()


def _get_labeled_items(
    collection: Generator,
) -> List[dict[str, OntologyIndividualSuperclass | OntologyClass]]:
    return [dict([(label, ind) for label in ind.isDefinedBy]) for ind in collection]


def get_labels_from_ontology(onto: Ontology) -> ChainMap:
    labels_from_individuals = _get_labeled_items(onto.individuals())
    labels_from_classes = _get_labeled_items(onto.classes())
    return ChainMap(*(labels_from_individuals + labels_from_classes))


def patterns_from_ontology(flattened_list: ChainMap):
    patterns = []
    for [pattern, item] in flattened_list.items():
        patters_splitted = pattern.split()
        pat_dict = [{"LEMMA": p_item} for p_item in patters_splitted]
        patterns.append({"label": item.name, "pattern": pat_dict})
    return patterns


def get_individuals_of_class(
    onto_individuals: Generator[OntologyIndividualSuperclass, None, None],
    ontoClass: OntologyClass,
) -> List[OntologyIndividualSuperclass]:
    return [x for x in onto_individuals if x.__class__ == ontoClass]


def get_class_descendants(root: OntologyClass) -> List[OntologyClass]:
    return [x for x in root.descendants()]


def get_class_named(onto: Ontology, name: str) -> OntologyClass:
    return [x for x in onto.classes() if x.name == name][0]


def get_product_names(
    product_individuals: List[OntologyIndividualSuperclass],
) -> List[str]:
    return [x.name for x in product_individuals]


def _get_devices_in_cell(
    current_cell_individual: OntologyIndividualSuperclass,
) -> set[OntologyClass]:
    return set(
        flatten_comprehension(
            [x.is_a for x in list(current_cell_individual.IsEquippedWith)]
        )
    )


def get_individuals_by_class(
    current_cell_individual: OntologyIndividualSuperclass,
    mentioned_devices_classes: Sequence[OntologyClass],
) -> dict[OntologyClass, OntologyIndividualSuperclass]:
    all_devices_classes = _get_devices_in_cell(current_cell_individual)
    individuals_by_class = {}
    for device_class in mentioned_devices_classes:
        if device_class not in all_devices_classes:
            device_individual = device_class()
            current_cell_individual.IsEquippedWith.append(device_individual)
            device_individual.IsEquipmentIn = current_cell_individual
        else:
            device_individual = next(
                dev_ind
                for dev_ind in current_cell_individual.IsEquippedWith
                if device_class in dev_ind.is_a
            )
        individuals_by_class[device_class] = device_individual
    return individuals_by_class


def get_all_parents(
    mentioned_devices_classes: Sequence[OntologyClass],
) -> set[OntologyClass]:
    all_parents = set()
    for dev_class in mentioned_devices_classes:
        tmp = dev_class.ancestors()
        tmp.remove(dev_class)
        all_parents.update(tmp)
    return all_parents
