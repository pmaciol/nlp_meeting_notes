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


from typing import Mapping, Any

def flatten_comprehension(list_of_lists):
    return [item for row in list_of_lists for item in row]

def add_to_dictionary(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]

def set_column_to_1000(matrix: Mapping[Any, Mapping], column_key):
    for row_key in matrix:
        matrix[row_key][column_key] = 1000


def set_row_to_1000(matrix: Mapping, row_key):
    for column_key in matrix[row_key]:
        matrix[row_key][column_key] = 1000

def find_item(dictionary: Mapping, item: Any) -> Any:
    for key, value in dictionary.items():
        if item in value:
            return key