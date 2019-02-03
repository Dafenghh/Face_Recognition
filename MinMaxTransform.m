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

