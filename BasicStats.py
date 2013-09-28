import matplotlib.pyplot as plt
from SimpleReportWriter import SimpleReportWriter
import pandas as pd

# This class generates a report summarizing basic informations about each variable
# (continuous/nominal, values/histogram & relation to a special 'target' variable)

# In this case (census dataset), 'target' is the variable we want to predict

class BasicStats:
    
    def __init__(self,dataset,report_dir,target_var):
        plt.clf()
        self.dataset = dataset
        self.report = SimpleReportWriter(report_dir)
        self.target = target_var
        
    def _summarizeContinuousVariable(self,column,target):
        
        self.report.row(['Variable type','Continuous'])
        description = self.dataset[column].describe()
        for k,v in description.iteritems():
            self.report.row([k,v])
        
        self.dataset[column].hist(bins=20)
        self.report.title('Histogram of values',level=2)
        self.report.plot()
        plt.close()
        
    def _summarizeCategoricalVariable(self,column,target):
        
        
        agg = self.dataset[column].value_counts()
        agg.plot(kind='bar')
        

        self.report.row(['Variable type','Categorical'])
        self.report.row(['Nb. of modalities','%d'%len(agg)])
        
        # could be done only one time , but who cares.. (it's not that slow ! :)
        target_values = self.dataset[target].value_counts()
        
        table_cols = ['Value','Total count']
   
        for k,v in target_values.iteritems():
            table_cols += ['%s=%s (in %%)'%(target,k),'%s=%s (count)'%(target,k)]
        
        self.report.title('Modalities & relation to %s'%target,level=2)
        self.report.table(table_cols)
        
        grouped = self.dataset.groupby(column)
        hist = dict()
        
        for name,group in grouped:

            table_row = [name, len(group)]
            
            for idx, (k,v) in enumerate(target_values.iteritems()):
                target_col = group[target]
                nb = len(target_col[target_col==k])
                percentage = 100.0*nb/len(group)
                table_row += ['%.1f%%'%percentage,'%d'%nb]
                if idx==0:
                    hist[name]=percentage
  
            self.report.row(table_row)
            
        self.report.title('Distribution & relation to \'%s\''%target,level=2)
        try :   
            pd.Series(hist).plot(secondary_y=True,style='g',figsize=(6,3))
            
            self.report.text('Bars = distribution of the values<br>Line = proportion of [%s=="%s"]'%(target,target_values.index[0]))
            self.report.plot()
        except:
            self.report.text('Histogram generation failed')
        plt.close()
    

    def analyze(self):
        
        for k,c in enumerate(self.dataset.columns):
            self.report.title('Column %d : %s'%(k+1,c))
            
            self.report.title('Overview',level=2)
            self.report.table(['Field name','Value'])
            self.report.row(['Variable name',c])
            
            #self.report.row(['Missing values',self.dataset[self.dataset[c]].count()])
            
            if self.dataset.dtypes[c].char=='O':
                self._summarizeCategoricalVariable(c,self.target)
            else:
                self._summarizeContinuousVariable(c,self.target)
            
        self.report.close()
 