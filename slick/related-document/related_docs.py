class related_docs():

    def __init__(self, docs = [], offset = 20):
        self.docs = []
        self.clean_docs = []
        self.labels = []
        self.size = 0
        self.best_k = 0
        self.nn = 3
        self.capacity = 0
        self.offset = offset
        self.vectorizer = TfidfVectorizer()
        self.knn = KNeighborsClassifier(n_neighbors=self.nn)
        self.nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
        self.stop = stopwords.words('english')


    def text_cleaner(self, doc):
        doc = re.sub("[,.']", "", doc)
        doc = [token for token in doc.split(' ') if len(re.sub("[a-zA-Z]", "", token)) <= 0]
        doc = self.nlp(" ".join(doc))
        taglist = ['RB','RBR', 'RBS','JJR','JJ','JJS','NN','NNS','VB','VBG','VBP','VBN']
        poslist = ['ADJ','ADV','NOUN','VERB']
        doc = [token.lemma_.lower() for token in doc if token.tag_ in taglist and token.pos_ in poslist]
        doc = [token for token in doc if not token in self.stop and len(token)>2]
        return " ".join(doc)

    def clean_vectorize(self):
        self.clean_docs = []
        for doc in self.docs:
            self.clean_docs.append(self.text_cleaner(doc))
        
        self.doc_vect = self.vectorizer.fit_transform(self.clean_docs)

    def add(self, _docs):
        # clustering on all docs 
        if (len(_docs) + self.size) > self.capacity:
            self.docs.extend(_docs)
            self.clean_vectorize()
            self.size = len(self.docs)
            self.capacity = self.size + self.offset

            scores = []
            K = range(2, max(3, self.size//4)+1)
            for k in K:
                km = KMeans(n_clusters=k, max_iter=200, n_init=10)
                km = km.fit(self.doc_vect)
                labels = km.labels_
                score = silhouette_score(self.doc_vect, labels)
                scores.append((k, score))
            
            scores.sort(key = lambda x : x[1], reverse = True)
            self.best_k = scores[0][0]

            km = KMeans(n_clusters=self.best_k, max_iter=200, n_init=10)
            km = km.fit(self.doc_vect)
            self.labels = list(km.labels_)

            id_label = list(zip(range(1, self.size+1), self.labels))
            id_label.sort(key = lambda x : x[1])
            print(id_label)

            self.nn = int(self.size**0.5)
            self.nn = self.nn if self.nn%2 else self.nn%2-1
            self.knn = KNeighborsClassifier(n_neighbors=self.nn)
            self.knn.fit(self.doc_vect, self.labels)

        # knn on new docs
        else:
            _clean_docs = []
            for doc in _docs:
                _clean_docs.append(self.text_cleaner(doc))
            _doc_vect = self.vectorizer.transform(_clean_docs)
            _labels = self.knn.predict(_doc_vect)
            self.docs.extend(_docs)
            self.clean_docs.extend(_clean_docs)
            self.labels.extend(_labels)
            self.size += len(_docs)
            print(_labels)

        
            

    def show_word_cloud(self):
        result={'cluster':self.labels,'data':self.clean_docs}
        result=pd.DataFrame(result)
        for k in range(0, self.best_k):
            s=result[result.cluster==k]
            text=s['data'].str.cat(sep=' ')
            text=text.lower()
            text=' '.join([word for word in text.split()])
            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
            print('Cluster: {}'.format(k))
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.show()

    def remove(self, pos):
        self.docs.pop(pos)
        self.clean_docs.pop(pos)
        self.labels.pop(pos)

    