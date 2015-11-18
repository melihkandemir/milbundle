function finalmodel=bag_svm_train(traindata,opt)

        addpath '/home/mkandemi/Dropbox/code/libsvm'
   
        model.ytr=zeros(length(traindata)*1,1);
        for bb = 1:length(traindata)
            model.ytr(bb)=traindata(bb).label;
        end
        model.ytr(model.ytr==0)=-1;
        model.best_thr=0.5;
        
        Xall=[];
        yall=[];
        indices=[];
        for bb = 1:length(traindata)
            Nb=size(traindata(bb).instance,1);
            indices=[indices; bb*ones(Nb,1)];
            yall=[yall; traindata(bb).label*ones(Nb,1)];
            Xall=[Xall; traindata(bb).instance];
        end
        probs=yall;      
        
        [N,D]=size(Xall);
        
        chosen=zeros(N,1);
        prev_chosen=zeros(N,1);


        hyp0.cov=[log(sqrt(D)) 1];
        hyp0.mean=mean(mean(Xall));

       % finalmodel.hyp=hyp0;

       % meanfunc = @meanConst; 
       % covfunc = @covSEiso; 
       % likfunc = @likErf;

        % initialize by randomly selectign samples1
       
        model.hyp=hyp0;
        probs=zeros(N,1);
        
        bestLoss=1;
        
        Xp=initialize_important_points(traindata);           

        for nn=1:30

          % fprintf('.')----------------------------
           if mod(nn,10)==0
          %    fprintf(' %d\n',uint16(nn));
           end                      
           
          if opt.kerneltype == 1
             svmmodel=svmtrain(model.ytr,Xp,['-q -c ' num2str(opt.C) ' -t 0']);              
          else
             svmmodel=svmtrain(model.ytr,Xp,['-q -c ' num2str(opt.C) ' -g  ' num2str(1/opt.sigma)]);              
          end
          
          [ypred, ~, probs] = svmpredict(ones(N,1), Xall, svmmodel);               
          
          bagprob=calcbagprob(indices,probs);

          curLoss=mean(abs(bagprob-(model.ytr>0)));

          if curLoss<bestLoss
              finalmodel=model;
              bestLoss=curLoss;
              fprintf('Loss: %.4f\n',curLoss);
          end

          for bb = 1:length(traindata)

               prob_bb =  probs(indices==bb);
              [probmax,maxidx]=sort(prob_bb,'descend');

               Xp(bb,:)=traindata(bb).instance(maxidx(1),:); 

               chosen_bb=zeros(length(prob_bb),1);
               chosen_bb(maxidx(1))=1;

               chosen(indices==bb)=chosen_bb;

          end

          %disp(mean(chosen==prev_chosen))
            if mean(chosen==prev_chosen)==1                
                break;
            else
                prev_chosen=chosen;
            end

        end

        finalmodel.svmmodel=svmmodel;
        finalmodel.possign=mean(sign(probs(ypred==1)));
        finalmodel.negsign=mean(sign(probs(ypred==-1)));

end

function Xp=initialize_important_points(traindata)
    D=size(traindata(1).instance,2);
    Xp=zeros(length(traindata),D);

    for bb = 1:length(traindata)
       Xp(bb,:)=mean(traindata(bb).instance);      
    end
end

function bagprob=calcbagprob(indices,probs)
    B=length(unique(indices));
    bagprob=zeros(B,1);
    
    for bb=1:B
        bagprob(bb)=max(probs(indices==bb));
    end
end
