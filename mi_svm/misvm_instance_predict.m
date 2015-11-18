function [ypred,probs]=misvm_instance_predict(testdata,model)

    [Xts,~,~,ytest_inst,ytest_bag]=struct_to_concat(testdata);  

    [ypred, ~, probs] = svmpredict(ones(size(Xts,1),1), Xts, model);
        
end
