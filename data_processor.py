import numpy as np 

def process_tag_genome(data_dir_name, num_movies, num_tags, cut_off_tags=None):
    if cut_off_tags is None:
        cut_off_tags = num_tags
    tags_filename = data_dir_name + '/' + 'tags.dat'
    tag_triples = []
    with open(tags_filename, 'r') as infile:
        for i in range(num_tags):
            line = infile.readline()
            token = line.split()
            name = ''
            for i in range(1, len(token) - 1):
                name += token[i] + ' '
            tag_triples.append((int(token[-1]), int(token[0]), name))
    tag_triples.sort(reverse=True)
    tag_triples = tag_triples[0:cut_off_tags]
    top_tag_ids = set()
    for p in tag_triples:
        top_tag_ids.add(p[1])
    
    # features_placeholder = tf.placeholder(features.dtype, features.shape)
    # labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    # dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    # iterator = dataset.make_initializable_iterator()
    # dataset.features = features
    # dataset.labels = labels

    data = np.zeros(num_movies * cut_off_tags)
    tag_relevance_filename = data_dir_name + '/' + 'tag_relevance.dat'
    with open(tag_relevance_filename, 'r') as infile:
        count = 0
        for i in range(num_movies * num_tags):
            line = infile.readline()
            token = line.split()
            if int(token[1]) in top_tag_ids:
                data[count] = float(token[2])
                count += 1
    features = np.zeros((num_movies, cut_off_tags))
    labels = np.zeros((num_movies, 1), dtype=int)
    
    for i in range(num_movies):
        labels[i] = i
        features[i] = data[i * cut_off_tags : (i+1) * cut_off_tags]
    
    dataset = {'features': features, 'labels': labels, 'tags': tag_triples}
    return dataset
