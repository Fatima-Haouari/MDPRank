import pandas as pd
import re
from scipy import spatial
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence

def get_BERT_embedding(text):
    # init embedding
    embedding = TransformerDocumentEmbeddings('aubmindlab/bert-base-arabertv02')
    # create a sentence
    sentence=Sentence(text)
    # embed the sentence
    embedding.embed(sentence)
    #return the embedding as a vector in a list
    #return sentence.embedding.to("cpu").detach().numpy().tolist()
    #return the embedding as a string
    return ' '.join(str(e) for e in sentence.embedding.to("cpu").detach().numpy().tolist())


