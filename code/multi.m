function x = multi(pdf)
    %generates single value from multinomial pdf
    %pdf= vector of probabilities
    %x= sample
    
    pdf=pdf/sum(pdf); %normalize
    cdf=cumsum(pdf); %calculate cdf
    p=rand; %uniform random variable
    x=find(cdf>p,1);
end
