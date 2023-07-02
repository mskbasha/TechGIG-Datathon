class classify:
    def __init__(self,label):
        self.mlb = MultiLabelBinarizer()  # For keywords column
        self.clf = DecisionTreeClassifier(random_state=0) # used basic decision tree
        self.encoders = [ ce.CountEncoder(cols=['BIDREQUESTIP']),
                          ce.CountEncoder(cols=['USERPLATFORMUID']),
                          ce.CountEncoder(cols=['USERAGENT']),
                          ce.CountEncoder(cols=['USERZIPCODE']),
                          ce.CountEncoder(cols=['URL']) ,
                          ce.CountEncoder(cols=['USERCITY'])]
        self.label = label
    def fit(self,df,count=1,total=10):
        train_keys = self.mlb.fit_transform(df.KEYWORDS.apply(lambda x:set(x.lower().split('|')))) # convert to one hot encoding
        train_keys = pd.DataFrame(train_keys,columns = self.mlb.classes_)
        train_y = df.TAXONOMY.apply(lambda x:x==self.label)
        d = pd.Series(dtype=int)
        pb = progressbar(1)
        # used Cramers V correlation to findout most correlated features with target
        for ind,i in enumerate(train_keys.columns):
            pb.print(ind,len(train_keys.columns),f'{self.label} Calculating correlations {round(count/total,2)}')
            d[i] = self.cramers_v(train_keys[i],train_y)
        d = d[d.isna()==0]
        d = d[d>0]
        self.imp_cols = list(d.keys())

        self.clf.fit(train_keys,train_y)

        for encoder in self.encoders:
            df = encoder.fit_transform(df)
        train_data = pd.concat([df[['BIDREQUESTIP',
                                    'USERCITY',
                                    'USERPLATFORMUID',
                                    'USERAGENT',
                                    'USERZIPCODE',
                                    'URL']].reset_index().drop('index',axis=1),
                                train_keys[self.imp_cols]],axis=1)
        self.clf.fit(train_data,train_y)


    def predict(self,df,count=0):
        test_keys = self.mlb.transform(df.KEYWORDS.apply(lambda x:set(x.lower().split('|'))))
        test_keys = pd.DataFrame(test_keys,columns = self.mlb.classes_)
        for encoder in self.encoders:
            df = encoder.transform(df)
        test_data = pd.concat([df[['BIDREQUESTIP',
                                    'USERCITY',
                                    'USERPLATFORMUID',
                                    'USERAGENT',
                                    'USERZIPCODE',
                                    'URL']].reset_index().drop('index',axis=1),
                                test_keys[self.imp_cols]],axis=1)
        return self.clf.predict(test_data)
        
    def cramers_v(self,x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        if r-1==0 or n-1==0:
            return 0
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        corr_val = (phi2corr / min((kcorr - 1), (rcorr - 1)))
        return np.sqrt(corr_val)
class Taxonomy:
    def __init__(self,df,no_cols = 10):
        self.Top_cols = df.TAXONOMY.value_counts().keys()[:no_cols]
        self.df = df
        self.classifiers = [classify(label) for label in self.Top_cols]
        self.no_cols = no_cols
    def fit(self):
        for ind,classifier in enumerate(self.classifiers):
            classifier.fit(self.df,count=ind,total=self.no_cols)
    def predict(self,df):
        tax = pd.DataFrame(0,index = df.index,columns = ['Taxonomy'])
        index = tax.index
        for classifier in self.classifiers:
            preds = pd.Series(classifier.predict(df.loc[index]),index = index)
            tax.loc[preds[preds==1].index] = classifier.label
            index = preds[preds==0].index
            if len(preds[preds==0])==0:
                return tax
        return tax