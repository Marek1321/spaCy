import spacy
import json
from collections import Counter, defaultdict

# Load the SpaCy model
nlp = spacy.load('en_core_web_trf')
nlp.max_length = 2500000  # Increase max length to 2.5 million characters

# Add custom entity patterns
patterns = [
    {"label": "ORG", "pattern": "X-Force"},
    {"label": "PERSON", "pattern": [{"LOWER": "vladimir"}, {"LOWER": "putin"}]},
    {"label": "PERSON", "pattern": "Putin"},
    # Add more custom patterns here
]

# Create and add the EntityRuler to the pipeline
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(patterns)

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to extract meaningful relationships involving entities
def extract_meaningful_relationships(ent, sent):
    relationships = []
    excluded_deps = {'punct', 'case', 'space', 'det'}

    for token in sent:
        if token.head == ent.root or ent.root.head == token or token.head == ent:
            if token.dep_ not in excluded_deps and token.pos_ not in ('PUNCT', 'SPACE'):
                related_phrase = " ".join([child.text for child in token.subtree])
                relationships.append({
                    'related_word': related_phrase,
                    'related_to': token.head.text if token.head != ent.root else ent.root.text
                })
        elif ent in token.subtree:
            path_to_entity = " ".join([child.text for child in token.subtree if
                                       child.dep_ not in excluded_deps and child.pos_ not in (
                                           'PUNCT', 'SPACE') and ent in child.subtree])
            relationships.append({
                'related_word': path_to_entity,
                'related_to': token.head.text if token.head != ent.root else ent.root.text
            })
    return relationships

# Function to normalize entity names
def normalize_entity_name(ent):
    entity_aliases = {
        "putin": "vladimir putin",
        # Add more aliases as needed
    }
    normalized_name = entity_aliases.get(ent.lower(), ent.lower())
    return normalized_name

# Function to extract entities and their contexts, including relationships
def extract_entities_with_relationships(doc, article_title, article_link):
    entities_info = {}
    for ent in doc.ents:
        sentence = ent.sent
        sentence_text = sentence.text
        relationships = extract_meaningful_relationships(ent, sentence)
        entity_text = normalize_entity_name(ent.text)

        if entity_text not in entities_info:
            entities_info[entity_text] = {
                'label': ent.label_,
                'original_text': ent.text,
                'mentioned_in': [{'article': article_title, 'link': article_link, 'text': [sentence_text],
                                  'relationships': relationships}]
            }
        else:
            article_contexts = [context for context in entities_info[entity_text]['mentioned_in'] if
                                context['article'] == article_title and context['link'] == article_link]
            if article_contexts:
                article_contexts[0]['text'].append(sentence_text)
                article_contexts[0]['relationships'].extend(relationships)
            else:
                entities_info[entity_text]['mentioned_in'].append(
                    {'article': article_title, 'link': article_link, 'text': [sentence_text],
                     'relationships': relationships})

    return entities_info

# Function to process the entire content of each webpage
def process_text(data):
    aggregated_entities_info = {}
    entity_count = Counter()
    entity_type_count = Counter()

    raw_data_size = 0

    for article in data:
        article_title = article.get('title', 'Unknown Title')
        article_link = article.get('link', 'Unknown Link')
        article_text = " ".join([content_piece['text'] for content_piece in article['article_content']])

        raw_data_size += len(article_text)

        doc = nlp(article_text)  # Process each article as a separate document

        entities_info = extract_entities_with_relationships(doc, article_title, article_link)

        for entity, info in entities_info.items():
            entity_count[entity] += 1
            entity_type_count[info['label']] += 1
            if entity not in aggregated_entities_info:
                aggregated_entities_info[entity] = info
            else:
                for context in info['mentioned_in']:
                    existing_contexts = [c for c in aggregated_entities_info[entity]['mentioned_in'] if
                                         c['article'] == context['article'] and c['link'] == context['link']]
                    if existing_contexts:
                        existing_contexts[0]['text'].extend(context['text'])
                        existing_contexts[0]['relationships'].extend(context['relationships'])
                    else:
                        aggregated_entities_info[entity]['mentioned_in'].append(context)

    entities_by_type = defaultdict(list)
    for entity, info in aggregated_entities_info.items():
        entity_type = info['label']
        entities_by_type[entity_type].append({
            'named_entity': info['original_text'],
            'mentioned_in': info['mentioned_in']
        })

    top_10_entities = entity_count.most_common(10)
    top_10_entities_dict = {aggregated_entities_info[entity]['original_text']: count for entity, count in
                            top_10_entities}

    excluded_types = {'CARDINAL', 'ORDINAL', 'DATE'}
    filtered_entities = {entity: count for entity, count in entity_count.items()
                         if aggregated_entities_info[entity]['label'] not in excluded_types}
    top_10_filtered_entities = Counter(filtered_entities).most_common(10)
    top_10_filtered_entities_dict = {aggregated_entities_info[entity]['original_text']: count
                                     for entity, count in top_10_filtered_entities}

    product_entities = {entity: count for entity, count in entity_count.items()
                        if aggregated_entities_info[entity]['label'] == 'PRODUCT'}
    top_10_product_entities = Counter(product_entities).most_common(10)
    top_10_product_entities_dict = {aggregated_entities_info[entity]['original_text']: count
                                    for entity, count in top_10_product_entities}

    total_entities = sum(entity_type_count.values())
    entity_distribution = {etype: (count / total_entities) * 100 for etype, count in entity_type_count.items()}

    top_5_entity_types = Counter(entity_distribution).most_common(5)

    top_entities_by_type = {}
    for entity_type, percentage in top_5_entity_types:
        entities_of_type = {entity: count for entity, count in entity_count.items()
                            if aggregated_entities_info[entity]['label'] == entity_type}
        top_entities = Counter(entities_of_type).most_common(5)
        top_entities_by_type[entity_type] = {
            'percentage': percentage,
            'top_entities': [{aggregated_entities_info[entity]['original_text']: count} for entity, count in top_entities]
        }

    return {
        'entities_by_type': entities_by_type,
        'top_10_entities': top_10_entities_dict,
        'top_10_filtered_entities': top_10_filtered_entities_dict,
        'top_10_product_entities': top_10_product_entities_dict,
        'entity_distribution': entity_distribution,
        'top_entities_by_type': top_entities_by_type,
        'raw_data_size': raw_data_size
    }

# Function to save processed data into a JSON file
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# Main execution
if __name__ == '__main__':
    input_file_path = '/Users/marekstrba/Documents/skola/bakalarka/Pycharm/scrapy/scrapySpider/malwCrawl/crawledWebsites.json'
    output_file_path = 'processed_data.json'  # Saving in the current working directory

    data = load_json(input_file_path)
    processed_data = process_text(data)
    save_json(processed_data, output_file_path)
