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


from typing import Sequence
import networkx as nx
from spacy.tokens import Span


def make_graph(case_text: Span) -> nx.Graph:
    edges = []
    for token in case_text:
        for child in token.children:
            edges.append(
                (
                    "{0}-{1}".format(token.lower_, token.i),
                    "{0}-{1}".format(child.lower_, child.i),
                )
            )
    return nx.Graph(edges)


def make_distance_matrix(
    items: Sequence[Sequence[Span]], graph: nx.Graph, problems: list[Span]
) -> dict[Span, dict[Span, int]]:
    matrix = {}
    for item in items:
        for item_occurance_in_text in item:
            matrix[item_occurance_in_text] = {}
            for problem_occurance_in_text in problems:
                item_name = "{0}-{1}".format(
                    item_occurance_in_text[0].lower_, item_occurance_in_text[0].i
                )
                problem_name = "{0}-{1}".format(
                    problem_occurance_in_text[0].lower_, problem_occurance_in_text[0].i
                )

                local_distance = 100
                try:
                    local_distance = nx.shortest_path_length(
                        graph, source=item_name, target=problem_name
                    )
                except:
                    local_distance = 100 + abs(
                        item_occurance_in_text[0].i - problem_occurance_in_text[0].i
                    )
                matrix[item_occurance_in_text][problem_occurance_in_text] = abs(
                    local_distance
                )
    return matrix
