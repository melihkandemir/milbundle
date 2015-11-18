function [bpred,bagprob]=misvm_predict(testdata,model)
  
    addpath ../libsvm/
 
   for bb = 1:length(testdata)
     Xbag = testdata(bb).instance;
      Nb=size(Xbag,1);
     
      
      [ypred, ~, probs] = svmpredict(ones(Nb,1), Xbag, model);
       

      bpred(bb)=max(ypred>0);
      bagprob(bb)=max(probs);


  
 
   end
  
  bagprob=bagprob';
   bpred=bpred';
   
end
