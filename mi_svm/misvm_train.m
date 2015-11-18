function finalmodel=misvm_train(traindata,opt)

   % addpath(genpath('../gpml'))
     
    addpath /home/mkandemi/Dropbox/code/libsvm/
    warning off
   
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
    ytrue=yall;
    yall(yall==0)=-1;    
    ypred=yall;
    
    mean(yall>0)
    
    [N,D]=size(Xall);
           
    
    lossList=ones(30,1);
  
    for tt=1:100
       
        fprintf('.')
        if mod(tt,10)==0
            fprintf(' %d\n',uint8(tt));
        end                             
              
        svmmodel=svmtrain(yall,Xall,['-q -c ' num2str(opt.C) ' -g     ' num2str(1/opt.sigma)]);
        [ypred, acc, probs] = svmpredict(yall, double(Xall), svmmodel);
        yall=ypred;  	
        
        
        
        for bb=1:length(traindata)
            if max(yall(indices==bb))==-1 && traindata(bb).label==1
                ybb=ypred(indices==bb);
                [~,idx]=max(ybb);
                ybb(idx(1))=1;
                yall(indices==bb)=ybb;
                %fprintf('here\n')
            end
        end       
       
        lossList(tt)=mean(ypred(ytrue==1)==1);       
       
       % current score
       fprintf('Iter %d Fval: %f\n',tt,lossList(tt));       
       
       if tt>1 && abs(lossList(tt-1)-lossList(tt))<0.000001
           break;
      
       end       
                   
       
    end
    
     
    finalmodel=svmmodel;
    

end
