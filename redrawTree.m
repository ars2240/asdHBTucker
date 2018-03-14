function [samples,tree,r] = redrawTree(dims,samples,L,tree,r,gam,xStarts)

    xEnds = [xStarts(2:dims(1))-1;size(samples,1)];
    for i=1:dims(1)
       for j=1:2
           curRes=1; %set current restaurant as root
           col=5+(j-1)*L(1); %starting column

           %all samples, excluding the one being resampled
           ex=samples;
           ex(xStarts(i):xEnds(i),:)=[];
           s=samples(xStarts(i):xEnds(i),:);

           new=0; %set boolean for new table to false

           %get number of samples in that restaurant
           ir=[xStarts(1:(i-1));
               (xStarts((i+1):dims(1))-xEnds(i)-1+xStarts(i))];

           for k=2:L(j)
               %get label of new table
               newRes=find(~ismember(1:max(r{j}+1),r{j}),1);

               if ~isempty(tree{j}{curRes})
                   %add new restaurant to list
                   rList=[tree{j}{curRes} newRes];
                   [rList, order]=sort(rList);

                   %compute CRP part of pdf
                   pdf=histc(ex(ir,col+k)',rList);

                   %get counts
                   cts1=accumarray(ex(:,[1+j col+k]),1,[dims(1+j) max(rList)]);
                   cts1=cts1(:,rList);
                   cts2=accumarray(s(:,[1+j col+k]),1,[dims(1+j) max(rList)]);
                   cts2=cts2(:,rList);
                   cts2=cts1+cts2;

                   %compute contribution to pdf
                   [~,l]=max(order);
                   pdf(l)=gam(j);
                   pdf=log(pdf); %take long to prevent overflow
                   pdf=pdf+gammaln(sum(cts1,1)+1);
                   pdf=pdf-sum(gammaln(cts1+1/dims(1+j)),1);
                   pdf=pdf+sum(gammaln(cts2+1/dims(1+j)),1);
                   pdf=pdf-gammaln(sum(cts2,1)+1);
                   pdf=exp(pdf);

                   %pick new table
                   nextRes=rList(multi(pdf));

                   if k~=L(j)
                       sub=ex(ir,col+k)==nextRes;
                       irEnd=[ir(2:size(ir))-1;size(ex,1)];
                       x=1:size(ir);
                       x=x(sub);
                       pos=mems(x,ir,irEnd);
                       ex=ex(pos,:);
                       [~,ir,~]=unique(ex(:,1));
                   end
                   
                   if nextRes==newRes
                       new=1;
                   end
               else
                   new=1; %set boolean for new table to false
                   nextRes=newRes;
               end

               samples(xStarts(i):xEnds(i),col+k)=nextRes; %sit at table

               %if new table
               if new==1
                   r{j}=[r{j} newRes]; %add to restaurant list

                   %add new table to tree
                   tree{j}{curRes}=[tree{j}{curRes} newRes];
                   tree{j}{newRes}=[]; %add new table to tree
               end

               curRes=nextRes; %cycle restaurants

           end

           %handle abandoned tables
           rList=reshape(samples(xStarts,(col+1):(col+L(j))),[],1);
           in=ismember(r{j},rList);
           r{j}=r{j}(in);
           r{j}=sort(r{j});
           for k=find(~in)
              tree{j}{k}=[];
           end
           for k=1:max(r{j})
              tree{j}{k}=tree{j}{k}(ismember(tree{j}{k},r{j}));
           end
       end
    end
end

function y = mems(x,ir,irEnd)
    y=ir(x):irEnd(x);
end