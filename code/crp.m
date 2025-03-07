function [table, p] = crp(r,gam)
    %draws a new restaurant from the Chinese Restaurant Process (CRP)
    %r = current state of restaurant tables
    %gam = hyperparameter
    
    d=gam+sum(r); %denominator of probability
    pdf=[r/d gam/d]; %probability of each table
    table=find(mnrnd(1,pdf,1)); %draw table
    p=pdf(table); %probability
end
