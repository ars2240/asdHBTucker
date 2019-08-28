function [bestDiff, bord] = tuckerMH(oTuck, tuck, iters)

    l=size(oTuck,1);
    if ~isequal(size(tuck),size(oTuck))
        error("Dimensions don't correspond.");
    end
    
    ord = randperm(l);
    bord = ord;
    curDiff = 0;
    for j=1:l
        oTuckS = tensor(ttensor(oTuck.core(j,:,:),{oTuck.U{2},oTuck.U{3}}));
        tuckS = tensor(ttensor(tuck.core(ord(j),:,:),{tuck.U{2},tuck.U{3}}));
        curDiff = curDiff + norm(oTuckS - tuckS)^2;
    end
    bestDiff = curDiff;

    for i=1:iters
        flips = randperm(l, 2);
        flops = fliplr(flips);
        nord = ord;
        nord(flips) = nord(flops);
        diff = curDiff;
        for j=flips
            oTuckS = tensor(ttensor(oTuck.core(j,:,:),{oTuck.U{2},oTuck.U{3}}));
            tuckS = tensor(ttensor(tuck.core(ord(j),:,:),{tuck.U{2},tuck.U{3}}));
            diff = diff - norm(oTuckS - tuckS)^2;
        end
        for j=flips
            oTuckS = tensor(ttensor(oTuck.core(j,:,:),{oTuck.U{2},oTuck.U{3}}));
            tuckS = tensor(ttensor(tuck.core(nord(j),:,:),{tuck.U{2},tuck.U{3}}));
            diff = diff + norm(oTuckS - tuckS)^2;
        end
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
    curDiff = sqrt(curDiff);
end