import datetime
import os
import re
from sentence_transformers import SentenceTransformer

from path_util import resource_path


# https://www.sbert.net/index.html
class SentenceBert:
    # model_name = 'all-mpnet-base-v2'
    model_name = 'all-MiniLM-L6-v2'
    model_file_path = resource_path('data/embeddings/sentence_bert')

    embedding_sentence_dict = {}

    def __init__(self):
        start = datetime.datetime.now()
        if os.path.exists(self.model_file_path):
            print('loading sentence BERT model from disk...')
            self.model = SentenceTransformer(self.model_file_path)
        else:
            print('downloading sentence BERT model...')
            self.model = SentenceTransformer(self.model_name)
            os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
            self.model.save(self.model_file_path)
        print(f'done loading model in {datetime.datetime.now() - start}')

    
    def get_embedding(self, sentence):
        # convert to lowercase and keep only alpha-numerical characters and spaces
        sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())
        # get embedding from the model
        embedding = self.model.encode([sentence])[0]
        # append embedding to dictionary
        self.embedding_sentence_dict[embedding.tobytes()] = sentence
        # return embedding
        return embedding

    
    def get_sentence(self, embedding):
        return self.embedding_sentence_dict.get(embedding.tobytes())
