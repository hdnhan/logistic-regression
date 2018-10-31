function dt = featureScaling(data)
	dt = [];
	nfeature = size(data, 2);

	for n = 1:nfeature
	    x = data(:, n);
	    %xmax = max(x); 
	    %xmin = min(x);
	    %dt = [dt, (x - xmin) / (xmax - xmin)];
	    xmean = mean(x);
        %dt = [dt, (x-xmean) / (xmax - xmin)];
	    xstd = std(x);
	    dt = [dt, (x-xmean) / xstd];   
	end
end