from lib.sentence2vec import Sentence2Vec


model = Sentence2Vec('./data/word2vec.model')

print(model.get_vector('All right'))

print(model.similarity('All right', 'Time pressure'))

print(model.similarity('Right, cool', 'It was steady, yes'))

print(model.similarity('All right', 'Right, cool'))
