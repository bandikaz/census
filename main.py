import pickle,os

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc,classification_report,confusion_matrix,zero_one_score
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import pandas as pd

from SimpleReportWriter import SimpleReportWriter
from BasicStats import BasicStats
from WeightWatcher import LinearModelAnalyzer
from FeatureTransformations import Bucketizer,CrossFeaturesMaker



class CensusLearner:
    
    def __init__(self):
        
        # hardcoded parameters :)
        self.training_data_fn = './data/census_income_learn.csv'
        self.testing_data_fn = './data/census_income_test.csv'
        self.column_infos_fn = './data/census_income_columns.txt'
        self.model_dir = './model/'
        self.report_dir = './reports/'
        self.reporting = True
        self.vectorizer = None
        self.model = None
    
    # train a logistic regression model
    def trainModel(self):
        
        print 'Load dataset...'
        trainset = pd.read_csv(self.training_data_fn,header=None,names=self._getColumns())
        
        if self.reporting:
            print 'Analyze RAW data...'
            BasicStats(trainset,self.report_dir+'/raw/','target').analyze()
        
        print 'Transform data...'
        preprocessed_trainset = self._transformDataset(trainset)
        
        if self.reporting:
            print 'Analyze transformed data...'
            BasicStats(preprocessed_trainset,self.report_dir+'/processed/','target').analyze()
        
        print 'Vectorize data...'
        train_as_dicts = [dict(r.iteritems()) for _, r in preprocessed_trainset.drop('target',axis=1).iterrows()]
        
        self.vectorizer = DictVectorizer()
        vectorized_trainset = self.vectorizer.fit_transform(train_as_dicts)
        train_labels = preprocessed_trainset['target'].astype(float).values
                
        print 'Train a model...'
        self.model = SGDClassifier(penalty='elasticnet',loss='log',alpha=0.00001)
        self.model.fit(vectorized_trainset,train_labels)
        predictions = self.model.predict_proba(vectorized_trainset)
        
        if self.reporting:
            
            self._analyzeLinearModel()
            report = SimpleReportWriter(self.report_dir+'/trainreport/')
            report.title('Evaluation on the trainset')
            self._evaluatePredictions(report,predictions,train_labels)
            
    # apply the model on test data
    def testModel(self):
        
        print 'Load testset...'
        
        testset = pd.read_csv(self.testing_data_fn,header=None,names=self._getColumns())
        
        print 'Transform data...'
        preprocessed_testset = self._transformDataset(testset)

        print 'Vectorize data...'
        test_as_dicts = [dict(r.iteritems()) for _, r in preprocessed_testset.drop('target',axis=1).iterrows()]
        
        vectorized_testset = self.vectorizer.transform(test_as_dicts)
        test_labels = preprocessed_testset['target'].astype(float).values
                
        print 'Apply the model...'
        predictions = self.model.predict_proba(vectorized_testset)
        
        if self.reporting:
            report = SimpleReportWriter(self.report_dir+'/testreport/')
            report.title('Evaluation on the testset')
            self._evaluatePredictions(report,predictions,test_labels)
            
    # report various scores to assess the model's quality
    def _evaluatePredictions(self,report,predictions,groundtruth):
        predicted_labels =  1*(predictions>.5)
        fpr, tpr, thresholds = roc_curve(groundtruth, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.clf()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        
        report.title('Summary',level=2)
        accuracy = zero_one_score(groundtruth,predicted_labels)
        report.text('Global accuracy = %.1f%%'%(100.0*accuracy))
        
        report.title('ROC Curve',level=2)
        report.plot()
        
        plt.close()
        
        report.title('Short report',level=2)
        report.pre(classification_report(groundtruth,predicted_labels))
        
        report.title('Confusion matrix',level=2)
        report.table(['Ground truth','0','1'])
        
        confusion = confusion_matrix(groundtruth,predicted_labels)
        
        for k in xrange(2):
            report.row([k,confusion[k][0],confusion[k][1]])
            
        report.close()
        
    # Load model from file
    def loadModel(self):
        print 'Load model...'
        self.vectorizer = self._pickeFromFile(self.model_dir+'/mapping.dat')
        self.model = self._pickeFromFile(self.model_dir+'/model.dat')
    
    # SAve to file
    def saveModel(self):
        print 'Save model...'
        try:
            os.makedirs(self.model_dir)
        except:
            pass # bad practice in general but here we don't really care 
        self._pickeToFile(self.vectorizer,self.model_dir+'/mapping.dat')
        self._pickeToFile(self.model,self.model_dir+'/model.dat')
        
        
    def _analyzeLinearModel(self):
        print 'Analyze selected variables...'
        LinearModelAnalyzer().analyze(self.model,self.vectorizer,self.report_dir+'/variables/')
    
    # Load the column names 
    def _getColumns(self):
        
        column_names = list()
        for line in open(self.column_infos_fn,'r'):
            line = line.strip()
            if line:
                column_names.append(line)
                
        return column_names
    
    
    # two little helpers
    def _pickeToFile(self,obj,fn):
        fp = open(fn,'wb')
        pickle.dump(obj,fp)
        fp.close()
    
    def _pickeFromFile(self,fn):
        fp = open(fn,'rb')
        obj = pickle.load(fp)
        fp.close()
        return obj


        
    # Transform the dataset in order to generate features for prediction
    def _transformDataset(self,dataset):
        
        def stringify(v):
            return '(nominal) %s'%v
        
        # "Cast" from continuous to nominal (those variables are not really continuous...even if they're numbers)
        dataset['veterans benefits'] = dataset['veterans benefits'].map(stringify)
        dataset['year'] = dataset['year'].map(stringify)
        dataset['num persons worked for employer']  = dataset['num persons worked for employer'].map(stringify)
        dataset['own business or self employed']  = dataset['own business or self employed'].map(stringify)
        
        dataset['occupation code']  = dataset['occupation code'].map(stringify)
        dataset['industry code']  = dataset['industry code'].map(stringify)
        
        
        
        # Bucketization of all remaining continuous variables (age,etc)
        dataset = Bucketizer.linear(dataset,'age',0,100,8)
        dataset = Bucketizer.linear(dataset,'wage per hour',0,500,5)
        dataset = Bucketizer.linear(dataset,'capital gains',0,500,5)
        dataset = Bucketizer.linear(dataset,'capital losses',0,500,5)
        dataset = Bucketizer.linear(dataset,'divdends from stocks',0,800,8)
        dataset = Bucketizer.linear(dataset,'instance weight',0,4700,16)
        dataset = Bucketizer.linear(dataset,'weeks worked in year',0,54,8)
        
        # Create some cross features to overcome linear models limitations (no interaction between variables)
        # I've not tried to find the best cross features, the goal is to show that it can be done easily here...
        dataset = CrossFeaturesMaker.cross(dataset,'industry code','occupation code')
        
        
        # Rename the target variable (for convenience only)
        def rename_target(n):
            if n==' - 50000.':
                return '0'
            else:
                return '1'
                
        dataset['target'] = dataset['target'].map(rename_target)
        
        return dataset
    
    

l = CensusLearner()
l.trainModel()
l.testModel()

