from sklearn import preprocessing
import spacy
import numpy as np
from spacy import displacy
from afinn import Afinn
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# Self defined function to output a string based on afinity score
def sentiment_score(score):
    sentiment = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
    return sentiment

# Strings to evaluate
s1_long = " U.S. intelligence agencies concluded in January 2017 that Russia mounted a far-ranging influence campaign aimed at helping Trump beat Clinton. And the bipartisan Senate Intelligence Committee, after three years of investigation, affirmed those conclusions, saying intelligence officials had specific information that Russia preferred Trump and that Russian President Vladimir Putin had “approved and directed aspects” of the Kremlin’s influence campaign."
s1 = "Donald Trump sleeps."

# Opening the trump file and assigning it to a string
f = open("APonTrump", "r")
text = f.read()

# loading spacy
nlp = spacy.load("en_core_web_sm")

# creating afinn score object
afinn = Afinn()

# Creating arrays for usage
t1 = []
t2 = []
t1display = []
t2display = []

# Initializton of spacy object with desired text
# doc = nlp(s1_long)
doc = nlp(text)

# Token and sentence splitting
print("Sentence Analysis: ")
for sentence in doc.sents:
    score = afinn.score(str(sentence))
    print(sentence[0:4], "...", " , Sentiment score :", score,
          " , Sentiment Analysis:", sentiment_score(score))
    for token in sentence:
        text = token.text
        isNE = token.ent_type if token.ent_type == 0 else 1
        typeNE = token.ent_type_ if token.ent_type else "None"
        typeNEint = token.ent_type
        governor = token.head
        listofDependants = [t.text for t in token.children]
        sa_text = afinn.score(text)
        sa_sentence = score
        t1display.append([text, isNE, typeNE, governor, listofDependants, sentiment_score(
            sa_text), sentiment_score(sa_sentence)])
        t1.append([text, isNE, typeNEint, str(governor), sa_text, sa_sentence])
        if isNE:
            t2display.append([text, isNE, governor, listofDependants, sentiment_score(
                sa_text), sentiment_score(sa_sentence)])
            # Adding to t2 for clustering
            t2.append([text,isNE, str(governor), sa_text, sa_sentence])

# Uncomment this snippet to print tokens and named entities
print("Tokens(T1) :")
for tokenArr in t1display:
              print("Text: ", tokenArr[0], " Is NE?: ", tokenArr[1], " Named entity type: ", tokenArr[2], " Governor: ", tokenArr[3], " ListOfDependants: ", tokenArr[4], " Sentitment Analysis of Text: ", tokenArr[5], "Sentiment Analysis of Sentence :", tokenArr[6])

print("Named Entities(T2):")
for neArr in t2display:
                  print("NE Text: ", neArr[0], " Is Ne?: ", neArr[1], " Governor: ", neArr[2], " List Of Dependants:", neArr[3]," Sentitment Analysis of Text: ", neArr[4], "Sentiment Analysis of Sentence :", neArr[5])


dataset1 = np.array(t1, dtype=object)
dataset2 = np.array(t2, dtype=object)

# Converting Non Number Collumns into numbers 
le = preprocessing.LabelEncoder()
dataset1[:, 0] = le.fit_transform(dataset1[:, 0])
dataset1[:, 3] = le.fit_transform(dataset1[:, 3])
dataset2[:, 0] = le.fit_transform(dataset2[:, 0])
dataset2[:, 2] = le.fit_transform(dataset2[:, 2])


np.set_printoptions(threshold=np.inf)

# When uncommented, this snippet Writes T1 and T2 to T1.txt and T2.txt
# with open('T1.txt', 'w') as f1:
#     f1.write(str(dataset1))
# with open('T2.txt', 'w') as f2:
#     f2.write(str(dataset2))


# 3 means clustering for T1
kmeans = KMeans(n_clusters=3)
kmeans.fit(dataset1)
print("Printing: 3-Kmeans for T1")
print(kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black')
plt.show()

# Writes kmeans output of t1 to file
# with open('T1_kmeans.txt', 'w') as f1:
#     f1.write(str(kmeans.labels_))

# 2 means clustering for T2
kmeans = KMeans(n_clusters=2)
kmeans.fit(dataset2)
print("Printing: 2-Kmeans for T2")
print(kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black')
plt.show()

# Writes kmeans output of t2 to file
# with open('T2_kmeans.txt', 'w') as f2:
#     f2.write(str(kmeans.labels_))

# Displays dependency graph on local server
sentences = list(doc.sents)
displacy.serve(sentences, style="dep")
