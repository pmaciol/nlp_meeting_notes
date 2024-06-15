stop_individuals = all individuals of class 'Cells'

for each entity in all_entities_ordered
    if (entity.lemma in stop_individuals)
        stop_entities_pairs.append(previous_entity, entity)

for each start_stop_entity in stop_entities_pairs:
    start_entity = start_stop_entity.first
    stop_entity = start_stop_entity.second
    first_sentence = start_entity.sentence
    last_sentence = (stop_entity.sentence - 1)  //sentence before the sentence including stop_entity
    case_text = text starting with first_sentence, ending with last_sentence
    if (event in case_text)
        cases.append(case_text)


# Wyszukiwanie wszystkich produktów w przypadku
for entity in case.all_entities:
    if (entity.label in all_product_names):
        product = ontology.all_individuals[entity.label]
        products_in_case.append(product)    # dictionary, hence each product in case in incuded only once 
for product in products_in_case:
        product.isDiscussed.append(case)    # object dependency is added to the ontology

# Wyszukiwanie i tworzenie instancji dla wszystkich urządzeń przypadku
for entity in case.all_entities:
    if (entity.label in all_device_classes):
        if not (entity.label in case.discussed_cell.device)
            device_individual = make_new_individual(all_device_classes[entity.label])
        else 
            device_individual = case.discussed_cell.device[entity.label]
        device_individual.IsEquipmentIn(cell)
        all_devices_in_case.append(device_individual)


# Usuwanie urządzeń innych niż "liście"
for device in all_devices_in_case:
    superclasses_in_case.append(device.superclasses)

for device in all_devices_in_case:
    if not device.class in superclasses_in_case
        leaf_devices_in_case.append(device)

# Dopasowanie urządzeń do problemów
for device in leaf_devices_in_case:
    for problem in problems_in_case:
        if calculate_shortest_path(device,product):
            distance[device,product] = calculate_shortest_path(device,product)
        else:
            distance[device,product] = MAXIMUM_DISTANCE
do:
    device,problem = find_smallest_distance(distance)
    problem.OccuredFor(device)
    problem.IsDiscussed(case)
    for device_used in leaf_devices_in_case:
        distance[device_used,product] = MAXIMUM_DISTANCE
    for problem_used in problems_in_case:
        distance[device,product_used] = MAXIMUM_DISTANCE
until (distance[device_used,product] == MAXIMUM_DISTANCE)