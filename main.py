#!/usr/bin/env python
# coding: utf-8

# In[1]
import pathlib
from owlready2 import *
import spacy
import pathlib
import networkx as nx
from spacy.symbols import ORTH
from spacy import displacy
from spacy.tokens import Span
from collections import ChainMap
from datetime import datetime

# In[2]
def flatten_comprehension(list_of_lists):
    return [item for row in list_of_lists for item in row]

def print_names(label, iterable):
    print(label, [x.name for x in iterable])

def add_to_dictionary(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]

def remove_keys_from_dict(dictionary, keys_to_remove):
    return {key: value for key, value in dictionary.items() if key not in keys_to_remove}

def print_matrix(title, matrix):
    print(title)
    for row in matrix.keys():
        print (row,":", matrix[row])

def find_item(dictionary, item):
    for key, value in dictionary.items():
        if item in value:
            return key

# In[3]
# Create a pipe that converts lemmas to lower case:
from spacy.language import Language
from functools import reduce

@Language.component('lower_case_lemmas')
def lower_case_lemmas(doc):
    for token in doc :
        token.lemma_ = token.lemma_.lower()
    return doc

@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    for i, token in enumerate(doc[:-2]):
        # Define sentence start if pipe + titlecase token
        if token.text == "\n":
            doc[i + 1].is_sent_start = True
        else:
            # Explicitly set sentence start to False otherwise, to tell
            # the parser to leave those tokens alone
            doc[i + 1].is_sent_start = False
    return doc

nlp = spacy.load("pl_core_news_lg")
nlp.remove_pipe('ner') # Removing unnecessary ner with basic categories
nlp.add_pipe("custom_sentencizer", before="parser")


filename = "sample.rdf"
onto = get_ontology(filename).load()
from_individuals = [dict([(label, ind) for label in ind.isDefinedBy]) for ind in onto.individuals()]
from_classes = [dict([(label, ind) for label in ind.isDefinedBy]) for ind in onto.classes()]
flattened_list = ChainMap(*(from_individuals + from_classes))

patterns = []
for [pattern, item] in flattened_list.items():
    patters_splitted = pattern.split()
    # print (patters_splitted)
    # key = "LEMMA"
    # pat_dict = []
    # for p_item in patters_splitted:
    #      pat_dict.append({key: p_item})
    #      key = "LOWER"
    pat_dict = [{"LEMMA": p_item} for p_item in patters_splitted]
    print (pat_dict)
    patterns.append({"label" : item.name, "pattern": pat_dict })

print(*patterns,sep='\n')


# In[8]:


nlp.add_pipe("lower_case_lemmas", after="tagger")
ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True, "validate": False})
ruler.add_patterns(patterns)


# In[15]:

file_name = "sample.txt"

doc = nlp(pathlib.Path(file_name).read_text(encoding="utf-8"))

stop_individuals = [x for x in onto.individuals() if x.__class__.name == 'Cells']
print_names("Stop individuals: ", stop_individuals)
product_individuals = [x for x in onto.individuals() if x.__class__.name == 'Product']
print_names("Product individuals: ", product_individuals)
product_names = [x.name for x in product_individuals]
print ('product_names', product_names)
product_individuals = [x for x in onto.individuals() if x.__class__.name == 'Product']

# defects_individuals = [x for x in onto.individuals() if x.__class__.name == 'Disqualifying_defect']
# print_names("Disqualifying defects: ", defects_individuals)
# defects_names = [x.name for x in defects_individuals]
# print ('defects_names', defects_names)
# defect_individuals = [x for x in onto.individuals() if x.__class__.name == 'Disqualifying_defect']

labeled_classes = [x for x in onto.classes() if len(x.isDefinedBy) > 0]
labeled_individuals = [x for x in onto.individuals() if len(x.isDefinedBy) > 0]
print_names('labeled_classes', labeled_classes)
print_names('labeled_individuals', labeled_individuals)

Occurance = [x for x in onto.classes() if x.name == 'Occurrance'][0]
problems = [x for x in Occurance.descendants()]
print_names('Problems', problems)

Device = [x for x in onto.classes() if x.name == 'Device'][0]
devices = [x for x in Device.descendants()]

# In[16]
dict_all_items = dict(list(flattened_list.items()))
print('dict_all_items', dict_all_items)
stop_ents = [ent for ent in doc.ents if dict_all_items[ent.lemma_] in stop_individuals]

filtered_stop_ents = [stop_ents[0]]
for se in stop_ents[1::]:
    if (dict_all_items[se.lemma_] != dict_all_items[filtered_stop_ents[-1].lemma_]):
        filtered_stop_ents.append(se)

paired_stop_ents = [(filtered_stop_ents[i], filtered_stop_ents[i+1] if i < len(filtered_stop_ents)-1 else None) for i in range(len(filtered_stop_ents))]

cases = {}
Concern = [x for x in onto.classes() if x.name == 'Concern'][0]
Case = [x for x in onto.classes() if x.name == 'Case'][0]
is_discussed = [x for x in onto.object_properties() if x.name == 'isDiscussed'][0]

for case_token in paired_stop_ents:
    current_cell = case_token[0]
    current_item = dict_all_items[current_cell.lemma_]
    case_start = doc[current_cell.sent.start]
    case_end = doc[case_token[1].sent.start] if case_token[1] != None else doc[-1]
    cases[case_token[0]] = Span(doc, case_start.i, case_end.i)    

print('discussed cells', cases.keys())

# In[18]
problems_by_name = {item.name:item for item in problems}
devices_by_name = {item.name:item for item in devices}

Meeting = [x for x in onto.classes() if x.name == 'Meeting'][0]
meeting_individual = Meeting("meeting_" + datetime.now().strftime("%m_%d_%Y"))

for case in cases:    
    case_item = Case()    
    cell_name = case
    for x in onto.individuals():
        if cell_name.lemma_ in x.isDefinedBy:
            current_cell_ind = x
    current_cell_ind.isDiscussed.append(case_item)
    case_item.refersTo.append(current_cell_ind)
    case_item.isDiscussedDuring.append(meeting_individual)
    print('\nCASE LEMMA',case.lemma_, list(current_cell_ind.IsEquippedWith))
    all_devices_classes = set(flatten_comprehension([x.is_a for x in list(current_cell_ind.IsEquippedWith)]))
    print('all_devices_classes', all_devices_classes)

    problems_in_case = {}
    mentioned_devices_classes = {}
    products_in_case = {}
    products_in_case_labels = {}
    defects_in_case = {}
    for tok in cases[case].ents:
        if (tok.label_ in [x.name for x in problems]):
            add_to_dictionary(problems_in_case, tok.label_, tok)                
        if (tok.label_ in [x.name for x in devices]):
            add_to_dictionary(mentioned_devices_classes,devices_by_name[tok.label_], tok)
        if (tok.label_ in product_names):
            product = [x for x in onto.individuals() if x.name == tok.label_]
            add_to_dictionary(products_in_case, product[0], case_item)                
            add_to_dictionary(products_in_case_labels,product[0], tok)

    print('PROBLEMS', problems_in_case)
    print('DEVICES', mentioned_devices_classes)
    print('PRDUCTS_LABELS', products_in_case_labels)

    for prod in products_in_case:
        prod.isDiscussed.append(products_in_case[prod][0])
    #     concern_item = Concern()
    #     concern_item.isDiscussed.append(products_in_case[prod][0])
    #     print('AAAAAAAAAAA', prod)
    #     concern_item.isProduct.append(prod)    
        print('For cell ', case, 'discussed product: ', prod.isDiscussed)
        print('Full text: ', cases[case], '\n')


    individuals_by_class = {}
    for device_class in mentioned_devices_classes.keys():
        if device_class not in all_devices_classes:
            device_individual = device_class()
            current_cell_ind.IsEquippedWith.append(device_individual)
            device_individual.IsEquipmentIn = current_cell_ind
            print('adding ', device_individual)
        else:            
            device_individual = next(dev_ind for dev_ind in current_cell_ind.IsEquippedWith if device_class in dev_ind.is_a)
            print('using ', device_individual)
        individuals_by_class[device_class] = device_individual
    print('INDIVIDUALS_BY_CLASS', individuals_by_class)

    all_parents = set()    
    for dev_class in mentioned_devices_classes.keys():
        tmp = dev_class.ancestors()
        tmp.remove(dev_class)
        all_parents.update(tmp)
    
    leaf_devices = remove_keys_from_dict(mentioned_devices_classes, all_parents)
    print('LEAF DEVICES', leaf_devices)

    breakdowns = problems_in_case['Breakdown'] if 'Breakdown' in problems_in_case.keys() else []
    print('BREAKDOWNS', breakdowns)
    defects = problems_in_case['Disqualifying_defect'] if 'Disqualifying_defect' in problems_in_case.keys() else []
    print('DEFECTS', defects)

    edges = []
    for token in cases[case]:
        for child in token.children:
            edges.append(('{0}-{1}'.format(token.lower_,token.i),'{0}-{1}'.format(child.lower_,child.i)))
    graph = nx.Graph(edges)
    
    matrix = {}
    for l in leaf_devices.values():
        for s_d in l:
            matrix[s_d] = {}
            for s_b in breakdowns:
                s_d_name = '{0}-{1}'.format(s_d[0].lower_,s_d[0].i)
                s_b_name = '{0}-{1}'.format(s_b[0].lower_,s_b[0].i)

                local_dist = 100
                try:
                    local_dist = nx.shortest_path_length(graph, source=s_d_name, target=s_b_name)
                except:
                    local_dist = 100 + abs(s_d[0].i - s_b[0].i)
                matrix[s_d][s_b] = abs(local_dist)
    
    dev_min = ""
    bre_min = ""
    dist = 0    
    if(len(matrix.keys()) > 0) and (len(matrix[next(iter(matrix))].keys()) > 0):
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
                if (str(s_d_other) == str(dev_min)):
                    for s_d_internal in matrix[dev_min]:
                        matrix[s_d_other][s_d_internal] = 1000
            for s_b_internal in matrix:
                 matrix[s_b_internal][bre_min] = 1000
            if dist < 1000:
                item_found = find_item(leaf_devices, dev_min)
                problem = problems_by_name[bre_min.ents[0].label_]()
                problem.isDiscussed.append(case_item)
                problem.occuredFor.append(individuals_by_class[item_found])
                print('occurance added ', problem, 'happend to', individuals_by_class[item_found])

    matrix = {}
    for l in products_in_case_labels.values():
        for s_d in l:
            matrix[s_d] = {}
            for s_b in defects:
                s_d_name = '{0}-{1}'.format(s_d[0].lower_,s_d[0].i)
                s_b_name = '{0}-{1}'.format(s_b[0].lower_,s_b[0].i)

                local_dist = 100
                try:
                    local_dist = nx.shortest_path_length(graph, source=s_d_name, target=s_b_name)
                except:
                    local_dist = 100 + abs(s_d[0].i - s_b[0].i)
                matrix[s_d][s_b] = abs(local_dist)
    
    dev_min = ""
    bre_min = ""
    dist = 0    
    if(len(matrix.keys()) > 0) and (len(matrix[next(iter(matrix))].keys()) > 0):
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
                if (str(s_d_other) == str(dev_min)):
                    for s_d_internal in matrix[dev_min]:
                        matrix[s_d_other][s_d_internal] = 1000
            for s_b_internal in matrix:
                 matrix[s_b_internal][bre_min] = 1000
            if dist < 1000:
                item_found = find_item(products_in_case_labels, dev_min)
                problem = problems_by_name[bre_min.ents[0].label_]()
                problem.isDiscussed.append(case_item)                
                print('PROBLEM ', problem, ' DEFECT ', products_in_case_labels[item_found], ' ITEM ', item_found)
                problem.occuredFor.append(item_found)
                # print('PROBLEM !!!!!!!!!!!!!!!!!!!!!', problem, item_found)


filenameUpdate = "result.rdf"
onto.save(filenameUpdate)        

# %%
