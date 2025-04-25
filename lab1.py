import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt_tab')



from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


doc1="i am vignesh from palani best back player in Ball badminton"
doc2="I am vignesh did my schooling at madurai"

t1=word_tokenize(doc1.lower())
t2=word_tokenize(doc2.lower())

print("token of 1 :",t1)
print("token of 2 :",t2)

def generate_wordcloud(text):
    wordcloud=WordCloud(width=800,height=400,background_color='white').generate(text)
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()
generate_wordcloud(doc1)
generate_wordcloud(doc2)
    