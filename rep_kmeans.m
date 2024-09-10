function [Acc, Nmi] = rep_kmeans(F, class, lab, repk)
    F = diag(1./sqrt(sum(F.^2,2)+eps))*F;
    for i = 1:repk
        idx = kmeans(F, class);
        [acc(i),~,~] = AccMeasure(lab',idx');
        nmii(i) = nmi(lab', idx');
    end
    Acc = mean(acc)/100;
    Nmi = mean(nmii);
end