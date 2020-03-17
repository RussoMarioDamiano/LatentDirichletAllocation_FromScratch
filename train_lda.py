import pandas as pd
from lda import LDA

df = pd.read_pickle("df.pkl")

punctuation = set("""!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~""")
instances = [[lemma for lemmatized_sentence in lemmatized_speech for lemma in lemmatized_sentence if lemma not in punctuation] 
                    for lemmatized_speech in df.lemmas]

K = 50
beta = 0.01
epochs = 10000

lda = LDA(num_topics=K, corpus=instances, alpha=50/K, beta=beta, epochs=epochs, no_below=9, no_above=0.7)

pd.to_pickle(lda, "lda.pkl")
