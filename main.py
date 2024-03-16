#!/usr/bin/env python
# coding: utf-8

# In[1]
import pathlib
from owlready2 import *
import spacy
import pathlib
from spacy.symbols import ORTH
from spacy import displacy
from spacy.tokens import Span
from collections import ChainMap

# In[2]
def flatten_comprehension(list_of_lists):
    return [item for row in list_of_lists for item in row]

def print_names(label, iterable):
    print(label, [x.name for x in iterable])

def is_token_allowed(token):
    return bool(
        token
        and str(token).strip()
        and not token.is_stop
        and not token.is_punct)

def preprocess_token(token):
    return token.lemma_.strip().lower()

def findRelated(fr):
        isTrue=False
        currentIndividual=[]
        for x in problems_individuals:
            if (fr == x.name):
                currentIndividual=x
                print('isRef: ', x, case, fr, x.isRelatedTo)
                q=x.isRelatedTo
#                print('len:',len(x.isRelatedTo))
                for tok in cases[case].ents:
                    print('tok ', tok.label_, 'curind=', q)
                    for z in q:
                        y=z.is_a
                        print('z =', y[0].name, 'tok =', tok.label_)
                        if (z.is_a == tok.label_):
                            print('ci:', z.is_a)
#                        findRelated(tok.label_)
                            isTrue=True
        return(currentIndividual)


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


# patterns = [{"label" : item.name, "pattern": [{"LEMMA": pattern}] } for [pattern, item] in flattened_list.items()]


# {"label": "SHIFT", "pattern": [{"LEMMA": "drugi"}, {"LEMMA": "zmiana"}]},
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

for case in cases:    
    case_item = Case()
    cell_name = case
    for x in onto.individuals():
        if cell_name.lemma_ in x.isDefinedBy:
            current_cell_ind = x
    current_cell_ind.isDiscussed.append(case_item)
    print('CASE LEMMA',case.lemma_, list(current_cell_ind.IsEquippedWith))
    all_devices_classes = set(flatten_comprehension([x.is_a for x in list(current_cell_ind.IsEquippedWith)]))
    print('all_devices_classes', all_devices_classes)
    problems_in_case = []
    mentioned_devices_classes = set()
    for sent in cases[case].sents:
        problems_in_sentence = set()
        for tok in sent.ents:
            if (tok.label_ in [x.name for x in problems] and tok.label_ not in problems_in_sentence):
                problems_in_sentence.add(tok.label_)
                print('PROBLEM MATCH: ',tok.label_, "FRAZA: ", tok)
                print('SENTENCE: ',tok.sent)
                print('Problems in sentence: ', problems_in_sentence)
                #problem = pair(tok, problems_by_name[tok.label_]())
                # to będzie potrzebne do NXa
                problem = problems_by_name[tok.label_]()
                problem.isDiscussed.append(case_item)
                problems_in_case.append(problem)
                    
            if (tok.label_ in [x.name for x in devices]):
                mentioned_devices_classes.add(devices_by_name[tok.label_])
    
    individuals_by_class = {}
    for device_class in mentioned_devices_classes:
        if device_class not in all_devices_classes:
            device_individual = device_class()
            current_cell_ind.IsEquippedWith.append(device_individual)
            print('adding ', device_individual)
        else:            
            device_individual = next(dev_ind for dev_ind in current_cell_ind.IsEquippedWith if device_class in dev_ind.is_a)
            print('using ', device_individual)
        individuals_by_class[device_class] = device_individual

    all_parents = set()    
    for dev_class in mentioned_devices_classes:
        tmp = dev_class.ancestors()
        tmp.remove(dev_class)
        all_parents.update(tmp)
    
    leaf_devices = mentioned_devices_classes - all_parents
    Breakdown = [x for x in onto.classes() if x.name == 'Breakdown'][0]
    for dev_class in leaf_devices:
        for pro in problems_in_case:
            is_as = pro.is_a
            if (Breakdown in is_as):
                pro.occuredFor.append(individuals_by_class[dev_class])
                print('occurance added ', pro.is_a, 'happend to', individuals_by_class[dev_class])

filenameUpdate = "result.rdf"
onto.save(filenameUpdate)

# In[19]
import networkx as nx
# In[20]
# Load spacy's dependency tree into a networkx graph
for sent in doc.sents:
    edges = []
    for token in sent:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append(('{0}-{1}'.format(token.lower_,token.i),
                        '{0}-{1}'.format(child.lower_,child.i)))

    graph = nx.Graph(edges)
    print(graph.nodes)

    # https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.shortest_paths.html
    # print(nx.shortest_path_length(graph, source='robots-0', target='awesomeness-11'))
    # print(nx.shortest_path(graph, source='robots-0', target='awesomeness-11'))
    # print(nx.shortest_path(graph, source='robots-0', target='agency-15'))

# In[16]:


# from spacy import displacy
# html = displacy.render(doc, style="ent",jupyter=False)
# text_file = open("output.html", "w")
# text_file.write(html)
# text_file.close()


# In[ ]:




