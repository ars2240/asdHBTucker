function [c, nu] = coh(X, t, options)
    if isfield(options,'coh')
        eps = options.coh.eps;
        x = double(X > 0); % convert to whether or not a word occurs
        dw = x.' * x + eps; % word co-occurrence
        p = log(dw); % log probabilities
        if ~contains(lower(options.coh.measure), "mass")
            p = p - log(size(X, 1) * (1 + eps));
        end
        d = diag(p); % diagonal is occurrence of words
        if contains(lower(options.coh.measure), "mass")
            s = log(dw + 1) - d';
        else
            s = p - d - d.'; % compute UCI/PMI score
        end
        
        topics = options.coh.topics;
        if isempty(topics)
            nt = size(t, 2); % number of topics
            topics = 1:nt;
        else
            nt = length(topics);
        end
        
        c = zeros(1,nt); % initialize topic coherence vector
        ui = []; % list of unique words
        for i = 1:nt
            [~,idx] = sort(t(:, topics(i)), 'descend'); % get top n words in topic
            n = min(options.coh.n,length(idx));
            idx = idx(1:n);
            c(i) = sum(tril(s(idx, idx)), 'all'); % get topic coherence
            ui = [ui; idx];
        end
        nu = length(unique(ui))/length(ui); % % of top words that are unique
        
        if options.coh.mean
            c = mean(c);
        end
    else
        error('Error. \nMissing coherence options.');
    end
end
