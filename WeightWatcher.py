from SimpleReportWriter import SimpleReportWriter
import numpy as np

# Generate a report showing the coefficients associated with each modalities
class LinearModelAnalyzer:
    
    def __init__(self):
        pass
    
    def analyze(self,model,vectorizer,report_dir):
        
        report = SimpleReportWriter(report_dir)
        report.title('Modalities selected by elastic net')
        report.text('Note : The "INTERCEPT" reflects the prior information')
        report.table(['Variable','Value','Coef','Absolute coef'])
        report.row(['INTERCEPT', '', '%f'%model.intercept_, '%f'%abs(model.intercept_)])
        
        names = vectorizer.get_feature_names()
        couples = zip(names,[float(v) for v in np.nditer(model.coef_)])
        
        accumulator = dict()
        
        for key,coef in sorted(couples,key=lambda x:abs(x[1]),reverse=True):
            variable_name,variable_value = key.split('=')
            accumulator.setdefault(variable_name,[]).append(coef)
            report.row([variable_name,variable_value,coef,abs(coef)])
        
        report.title('Usefulness of each variable (norm of the associated weight vector)')
        report.text('This is a completely non-scientific indicator.  ')
        report.table(['Variable','L1 norm','L1 norm / nb values','L2 norm','L2 norm / nb values','Linf norm'])
        
        for key,vals in accumulator.iteritems():
            report.row([key,np.linalg.norm(vals,1),np.linalg.norm(vals,1)/len(vals),np.linalg.norm(vals,2),np.linalg.norm(vals,2)/len(vals),np.linalg.norm(vals,np.inf)])
            
        
        report.close()


