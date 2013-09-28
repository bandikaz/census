import math
 
 
# Transform a continuous variable into a nominal one
# by replacing each value by the bucket containing it
# The linear bucketizer equally splits the range (minval;maxval)
# into 'nbuckets' parts
class Bucketizer:
    
    # very simple linear bucketizer
    # we should also try a logarithmic scale...
    @staticmethod
    def linear(trainset,colname,minval,maxval,nbuckets):
        
        buckets = []
        for k in xrange(nbuckets):
            bkmin = minval+1.0*k*(maxval-minval)/(nbuckets)
            bkmax = minval+1.0*(k+1)*(maxval-minval)/(nbuckets)
            buckets.append('Bucket %3d (%.1f - %.1f)'%(k+1,bkmin,bkmax))
        
        def helper(v):
            tmp = nbuckets*(float(v)-minval)/(maxval-minval)
            bucket = int(math.floor(max(0.0,min(nbuckets-1,tmp))))
            return buckets[bucket]
            
        trainset['%s bktz'%colname] = trainset[colname].map(helper)
        trainset = trainset.drop(colname,axis=1)
        return trainset

# Create cross features
# vectorized cross-feature = kronecker product of two vectorized categorical variables 
class CrossFeaturesMaker:
    
    @staticmethod
    def cross(dataset,col1,col2):
        
        def helper(row):
            return '%s^%s'%(row[col1],row[col2])
            
        dataset['%s^%s'%(col1,col2)] = dataset.apply(helper,axis=1)
        
        return dataset