% step 0: read the images
image_dims = [112, 92];
num_images = 40;
test_images = cell(40, 3);
train_images = zeros(prod(image_dims), num_images);

input_dir = 'att_faces';

for i=1:40
    sub_dir = strcat('s', num2str(i));
    images = cell(10);
    for j=1:10
        filename = fullfile(input_dir, sub_dir, strcat(num2str(j), '.pgm'));
        images{j} = imread(filename); 
    end
   
    images = images(randperm(10));
    
    img = zeros(image_dims);
    
    for j=1:7
        img = img + double(images{j});
    end
    
    img = img / 7;
    train_images(:, i) = img(:);
    
    for j=8:10
        test_images{i,j-7}=images{j};
    end
end

% steps 1: find the mean image and the mean-shifted input images
mean_face = mean(train_images, 2);
shifted_images = train_images - repmat(mean_face, 1, num_images);
 
% steps 2: calculate the ordered eigenvectors and eigenvalues
[full_evectors, score, evalues] = pca(train_images');


% step 3: show eigenfaces
% figure;
% for i = 1:num_eigenfaces
%    subplot(5, ceil(num_eigenfaces/5), i);
%    evector = MinMaxTransform(reshape(evectors(:,i), image_dims));
%    imshow(evector);
%end
 
result = zeros(39, 2);
% step 4: only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
for num_eigenfaces = 1:39
    evectors = full_evectors(:, 1:num_eigenfaces);

    features = evectors' * shifted_images;

    cnt = 0;

    for i=1:40
        for j=1:3
            input_image = double(test_images{i,j});
            % calculate the similarity of the input to each training image
            feature_vec = evectors' * (input_image(:) - mean_face);
            similarity_score = arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)), 1:num_images);

            % find the image with the highest similarity
            [match_score, match_ix] = max(similarity_score);

            if match_ix == i
                cnt = cnt + 1;
            else
                % display the result
                %figure, imshow([uint8(input_image) uint8(reshape(train_images(:,match_ix), image_dims))]);
                %title(sprintf('test_image %d matches %d %d, score %f', i, match_i, match_ix, match_score));
            end
        end
    end
    result(num_eigenfaces, 1) = cnt;
    result(num_eigenfaces, 2) = cnt / 120;
end

figure, plot(result(:, 2));
xlabel('No. of eigenfaces'), ylabel('Correct rate');
xlim([1 39]), ylim([0 1]), grid on;


    
    
    