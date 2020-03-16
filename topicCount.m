options=init_options();
options.gam = .5;
options.L = 4;

t_g = zeros(1,10);
t_p = zeros(1,10);
t_t = zeros(1,10);
for i=1:10
    load(['data/cancerHBTuckerCV_L', int2str(options.L), '_gam', ...
            num2str(options.gam), '_', int2str(i),'_trees.mat'])
    t_g(i)=size(phi,2);
    t_p(i)=size(phi,3);
    t_t(i)=t_g(i)*t_p(i);
end

sprintf('%f',mean(t_g))
sprintf('%f',mean(t_p))
sprintf('%f',mean(t_t))