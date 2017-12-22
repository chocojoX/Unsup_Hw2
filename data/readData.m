d = dir;
for i = 1:length(d)
    if ( (d(i).isdir == 1) && ~strcmp(d(i).name,'.') && ~strcmp(d(i).name,'..') )
        filepath = d(i).name;
        eval(['cd ' filepath]);
        f = dir;
        foundValidData = false;
        for j = 1:length(f)
            if (~isempty(strfind(f(j).name,'_truth.mat')) )
                ind = j;
                foundValidData = true;
                break
            end
        end
        eval(['load ' f(ind).name]);
        cd ..

        if (foundValidData)
            n = max(s);
            N = size(x,2);
            F = size(x,3);
            D = 2*F;
            X = reshape(permute(x(1:2,:,:),[1 3 2]),D,N);
        end
    end
end

