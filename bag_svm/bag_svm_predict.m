function [bpred,bagprob]=bag_svm_predict(testdata,model)

 
   for bb = 1:length(testdata)
     Xbag = testdata(bb).instance;
      Nb=size(Xbag,1);
     
      
      [ypred, ~, probs] = svmpredict(ones(Nb,1), Xbag, model.svmmodel);
      
      if model.possign==-1
          probs=-probs;
      end
       
      bagprob(bb)=max(probs);
      bpred(bb)=bagprob(bb)>0;

  
 
   end
   
   bagprob=bagprob';
   bpred=bpred';
  
end
