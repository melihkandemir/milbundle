from melpy.core.Prediction import Prediction

class ClassPrediction(Prediction):
    
    predictions=0;
    probabilities=0;
    
    def __init__(self,predictions,probabilities):

       Prediction.__init__(self)
       
       self.predictions=predictions
       self.probabilities=probabilities