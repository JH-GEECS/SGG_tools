import gensim.downloader

print(list(gensim.downloader.info()['models'].keys()))
glove_vectors = gensim.downloader.load('glove-twitter-25')
test = 1
glove_vectors.most_similar('twitter')
glove_vectors.distance('sky','trucks')
