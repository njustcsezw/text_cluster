import nltk
import re
import time
import xml.dom.minidom
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from string import digits
from sklearn.cluster import KMeans, Birch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


# 去除文本里面的符号
def pre_process(data):
    text_new = []
    for text in data:
        remove_digits = str.maketrans('', '', digits)
        res = text.translate(remove_digits)
        result = res.replace('_', '')
        result = result.replace('*', '')
        result = result.replace('/', '')
        result = result.replace('\\', '')
        result = result.replace('{', '')
        result = result.replace('}', '')
        result = result.replace('-', '')
        result = result.replace('(', '')
        result = result.replace(')', '')
        text_new.append(result)
    return text_new


# 载入nltk的英文停用词作为停用词变量
stopwords = nltk.corpus.stopwords.words('english')

# 载入nltk的SnowballStemmer作为词干化变量
stemmer = nltk.SnowballStemmer("english")


# 分句加分词，分词后检查单词正确性并还原单词原型
def tokenize_and_stem(text):
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = ''
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if token not in stopwords:
                filtered_tokens += ' ' + stemmer.stem(token)
    return filtered_tokens


# 只分句分词，并都转换为小写
def tokenize_only(text):
    # 首先分句，接着分词，而标点也会作为词例存在
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # 过滤所有不含字母的词例（例如：数字、纯标点）
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

# 开始读文本
dom = xml.dom.minidom.parse('data\\nysk.xml')
root = dom.documentElement
xml_title = root.getElementsByTagName('title')
text_read = []
for item in xml_title:
    text_read.append(item.firstChild.data)

# print(text_read[0])

# 开始清理单词
text_clean = pre_process(text_read)
text_stemmed = []

for i in text_clean:
    words_stemmed = tokenize_and_stem(i)
    text_stemmed.append(words_stemmed)

# print(text_stemmed)
# print(len(text_stemmed))

tfidf_vectorizer = TfidfVectorizer(min_df=2)
tfidf_matrix = tfidf_vectorizer.fit_transform(text_stemmed)

pca = PCA(n_components=2)  # 进行PCA降维
newdata = pca.fit_transform(tfidf_matrix.toarray())


# 进行kmeans聚类
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
start = time.time()
result = km.fit_predict(newdata)
end = time.time()
print("k_means运行的时间是：", end-start)
plt.scatter(newdata[:, 0], newdata[:, 1], c=result)
plt.show()


# 进行dbscan聚类
start = time.time()
db = DBSCAN(eps=0.03, min_samples=30).fit_predict(newdata)
end = time.time()
print("DBscan运行的时间是：", end-start)
plt.scatter(newdata[:, 0], newdata[:, 1], c=db)
plt.show()


# 进行birch聚类
start = time.time()
result_birch = Birch(n_clusters=5).fit_predict(newdata)
end = time.time()
print("birch运行的时间是：", end-start)
plt.scatter(newdata[:, 0], newdata[:, 1], c=result_birch)
plt.show()
