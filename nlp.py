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
from spacy.language import Language
from spacy import load
from spacy.tokens import Span, Doc
from typing import List

# Create a pipe that converts lemmas to lower case:
@Language.component("lower_case_lemmas")
def lower_case_lemmas(doc):
    for token in doc:
        token.lemma_ = token.lemma_.lower()
    return doc


# Create a custom sentencizer that sets sentence boundaries at newline characters:
@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    for i, token in enumerate(doc[:-2]):
        doc[i + 1].is_sent_start = token.text == "\n"
    return doc


def prepare_nlp(spacy_baseline: str, patterns) -> Language:
    nlp = load(spacy_baseline)
    nlp.remove_pipe("ner")  # Removing unnecessary ner with basic categories
    nlp.add_pipe("custom_sentencizer", before="parser")
    nlp.add_pipe("lower_case_lemmas", after="tagger")
    ruler = nlp.add_pipe(
        "entity_ruler", config={"overwrite_ents": True, "validate": False}
    )
    ruler.add_patterns(patterns)
    return nlp

def prepare_document(language: Language, file_path: pathlib.Path) -> Doc:
    return language(pathlib.Path(file_path).read_text(encoding="utf-8"))

def pair_stop_entities(filtered_stop_ents: List[Span]) -> List[tuple[Span, Span]]:
    paired_stop_ents = [
        (
            filtered_stop_ents[i],
            filtered_stop_ents[i + 1] if i < len(filtered_stop_ents) - 1 else None,
        )
        for i in range(len(filtered_stop_ents))
    ]
    return paired_stop_ents

def get_named_problems(
    problems_in_case: dict[str, list[Span]], problem_name: str
) -> List[Span]:
    return (
        problems_in_case[problem_name]
        if problem_name in problems_in_case.keys()
        else []
    )
