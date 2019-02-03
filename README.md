
# 数字图像处理 Final Project


## 1. 算法描述


使用特征脸算法进行实验。

以下是特征脸的实现过程：

1. 准备一个训练集的人脸图像。本实验选取剑桥大学ORL人脸数据库。一共40个不同的人，每人10张人脸图像，随机选取7张用作训练（取平均后作为一张脸），图像分辨率为112*92.
2. 将原始图像的每一行的像素串联在一起，产生一个具有112*92个元素的列向量，每个图像被视为一个向量。然后，使所有的训练集的图像（一共40张）存储在一个单一的矩阵T中，矩阵的每一列是一个图像。
3. 减去均值向量. 均值向量a要首先计算，并且T中的每一个图像都要减掉均值向量。
4. 计算协方差矩阵S的特征值和特征向量。每一个特征向量的维数与原始图像的一致，因此可以被看作是一个图像。因此这些向量被称作特征脸。
5. 选择主成分。一般选择最大的k个特征值，保留对应的特征向量。
6. 对每个训练图像的向量，投影到特征空间后得到一组坐标。对测试图像，也作同样的投影运算，得到坐标，与训练图像的坐标进行二范数最小匹配。


## 2. 源代码
完整代码见附件main.m，MinMaxTransform.m文件。
### step 0: 读取图像，对每个人，随机选择7张用作训练，3张用作测试

```matlab
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
```

### step 1：计算均值图像和mean-shifted图像
```matlab
mean_face = mean(train_images, 2);
shifted_images = train_images - repmat(mean_face, 1, num_images);
```
### step 2:  计算特征值和特征向量
```matlab
[full_evectors, score, evalues] = pca(train_images');
```


### step 3: 显示特征脸
```matlab
figure;
for i = 1:num_eigenfaces
    subplot(5, ceil(num_eigenfaces/5), i);
    evector = MinMaxTransform(reshape(evectors(:,i), image_dims));
    imshow(evector);
end
```
其中MinMaxTransform函数用于将特征向量变换到[0,255]值域上，定义如下：
```matlab
function [res] = MinMaxTransform(image)
[m,n] = size(image);
Min = min(min(image));
Max = max(max(image));
range = Max - Min; 
res = zeros(m,n);
for i=1:m
    for j=1:n
        res(i,j)=(image(i,j)-Min)*255/range;
    end
end
res = uint8(res);
end
```

### step 4: 保留主成分，即前num_eigenfaces个特征向量。然后测试。
```matlab
evectors = full_evectors(:, 1:num_eigenfaces);
features = evectors' * shifted_images;

cnt = 0;
for i=1:40
    for j=1:3
        input_image = double(test_images{i,j});
        feature_vec = evectors' * (input_image(:) - mean_face);
        
        similarity_score = arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)), 1:num_images);

        % find the image with the highest similarity
        [match_score, match_ix] = max(similarity_score);

        if match_ix == i
            cnt = cnt + 1;
    end
end
```

cnt即正确识别的图像数。


## 3. 实验结果
### 3.1 性能测试

特征维数取不同值时的准确率
测试图像：120张

特征维数|测试正确的图像数|正确率
:--:|:----:|:----:
1	|	17	|	14.17%
2	|	47	|	39.17%
3	|	66	|	55.00%
4	|	80	|	66.67%
5	|	87	|	72.50%
6	|	86	|	71.67%
7	|	93	|	77.50%
8	|	100	|	83.33%
9	|	102	|	85.00%
10	|	102	|	85.00%
11	|	106	|	88.33%
12	|	105	|	87.50%
13	|	105	|	87.50%
14	|	105	|	87.50%
15	|	107	|	89.17%
16	|	108	|	90.00%
17	|	108	|	90.00%
18	|	108	|	90.00%
19	|	108	|	90.00%
20	|	108	|	90.00%
21	|	108	|	90.00%
22	|	108	|	90.00%
23	|	109	|	90.83%
24	|	110	|	91.67%
25	|	110	|	91.67%
26	|	109	|	90.83%
27	|	109	|	90.83%
28	|	110	|	91.67%
29	|	110	|	91.67%
30	|	109	|	90.83%
31	|	109	|	90.83%
32	|	109	|	90.83%
33	|	109	|	90.83%
34	|	109	|	90.83%
35	|	109	|	90.83%
36	|	109	|	90.83%
37	|	109	|	90.83%
38	|	109	|	90.83%
39	|	109	|	90.83%

![image.png-49.5kB][1]

### 3.2 特征脸图像

如果有 N 个训练样本，则最多有 N − 1 个对应非零特征值的特征向量。本实验中有40个训练样本，所以特征向量有39个。

![image.png-704.6kB][2]



## Reference

1. Eigenfaces face recognition (MATLAB): https://blog.cordiner.net/2010/12/02/eigenfaces-face-recognition-matlab/

2. PCA 10, eigen-faces: https://www.youtube.com/watch?v=_lY74pXWlS8

3. 特征脸，维基百科：https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E8%84%B8

  [1]: http://static.zybuluo.com/dafenghh/9uy9pwki1anhg1yd7mctpaoz/image.png
  [2]: http://static.zybuluo.com/dafenghh/51yp226uwoq4b28fiw1wpqak/image.png