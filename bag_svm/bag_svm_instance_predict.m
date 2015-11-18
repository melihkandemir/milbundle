function [ypred,probs]=bag_svm_instance_predict(testdata,model)

    [Xts,~,~,ytest_inst,ytest_bag]=struct_to_concat(testdata);  

    [ypred, ~, probs] = svmpredict(ones(size(Xts,1),1), Xts, model.svmmodel);
        
end
