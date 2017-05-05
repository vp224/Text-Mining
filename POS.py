import nltk
from pyspark import SparkContext,SparkConf
import re
import operator
from matplotlib import pyplot as plt
from matplotlib import colors


class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class POSTagger(object):
    def __init__(self):
        pass
        
    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos
file = open("dhoni.txt","r")
text=""
for line in file:
    text=text+line.decode('utf-8').strip("\n")
# text = "What can I say about this place. The staff of the restaurant is nice and the eggplant is not bad. Apart from that, very uninspired food, lack of atmosphere and too expensive. I am a staunch vegetarian and was sorely dissapointed with the veggie options on the menu. Will be the last time I visit, I recommend others to avoid."

splitter = Splitter()
postagger = POSTagger()

splitted_sentences = splitter.split(text)

# print splitted_sentences

pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

# print pos_tagged_sentences
final = ""
for a in pos_tagged_sentences:
    for b in a:
        final = final + b[0]+'-'+b[2][0] +" "
print "str obtained"
file = open("dhoniPOS.txt","w")  
file.write(final.encode('ascii','ignore'))
print "written to file"
pattern = re.compile("^([a-z]|[A-Z]|[0-9]|(-))+$")


sc = SparkContext()
# sc.setLogLevel("ERROR")
a = sc.textFile("dhoniPOS.txt")
b = a.flatMap(lambda x: x.split(" "))
d = b.map(lambda x: (x.lower(),1)).reduceByKey(lambda a,b: a+b)
g = d.collect()
f = {}
for w in g:
    if w[0]!= "" and pattern.match(w[0]):
        f[w[0]] = w[1]
sorted_x = sorted(f.items(), key=operator.itemgetter(1),reverse = True)
for w in sorted_x:
    print w

fig = plt.figure()
ax = fig.add_subplot(111)

X=list()
Y=list()
xt=list()
yt=list()

for i in range(0,15):
    t = sorted_x[i]
    X.append(i)
    xt.append(t[0])
    yt.append(t[0])
    Y.append(t[1])
    i=i+1

plt.xticks(X,xt)
plt.plot(X,Y)
i=0
for xy in zip(X,Y):                                       # <--
    ax.annotate(yt[i], xy=xy, textcoords='data') # <--
    i=i+1
sc.stop()
plt.grid()
plt.show()

