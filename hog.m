for i=1:10000
    t = x_test(i,1:3072);
    t1 = reshape(t,32,32,3);
    t2 = rgb2gray(t1);
    w = fspecial('log',[3 3],0.5);
    fil_img = imfilter(t2,w,'replicate');
    t3 = reshape(fil_img, 1, 1024);
    t4 = extractHOGFeatures(t2);
    x_hog_test(i,:)= t4;
    x_log_test(i,:) = t3;
end
    
for i=1:10000
    t = x_cv(i,1:3072);
    t1 = reshape(t,32,32,3);
    t2 = rgb2gray(t1);
    w = fspecial('log',[3 3],0.5);
    fil_img = imfilter(t2,w,'replicate');
    t3 = reshape(fil_img, 1, 1024);
    t4 = extractHOGFeatures(t2);
    x_hog_cv(i,:)= t4;
    x_log_cv(i,:) = t3;
end
    
for i=1:40000
    t = x_train(i,1:3072);
    t1 = reshape(t,32,32,3);
    t2 = rgb2gray(t1);
    w = fspecial('log',[3 3],0.5);
    fil_img = imfilter(t2,w,'replicate');
    t3 = reshape(fil_img, 1, 1024);
    t4 = extractHOGFeatures(t2);
    x_hog_train(i,:)= t4;
    x_log_train(i,:) = t3;
end
    