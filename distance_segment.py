class DistanceEventSegmentation(object):

    def __init__(self):
        """[summary]
        """
        super().__init__()


    def fit(X, distance='euclidean', window=None):
        
        distance_b = [euclidean(timecourses[i], timecourses[i+1]) for i in range(timecourses.shape[0] - 1)]

