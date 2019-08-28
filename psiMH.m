function [bestDiff, bord] = psiMH(oPsi, psi, iters)

    l=size(oPsi,2);
    if size(psi,2) ~= l
        error("Dimensions don't correspond.");
    end
    
    ord = randperm(l);
    bord = ord;
    curDiff = norm(oPsi - psi(:,ord));
    bestDiff = curDiff;

    for i=1:iters
        flips = randperm(l, 2);
        flops = fliplr(flips);
        nord = ord;
        nord(flips) = nord(flops);
        diff = norm(oPsi - psi(:,nord));
        if diff<curDiff
            curDiff = diff;
            ord = nord;
            if curDiff<bestDiff
                bestDiff = curDiff;
                bord = ord;
            end
        else
            alpha = rand;
            if alpha<curDiff/diff
                curDiff = diff;
                ord = nord;
            end
        end
    end
end