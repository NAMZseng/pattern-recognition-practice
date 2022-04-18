clc;
tic; % 记录时间

% 绘制原始图像的三维像素散点图
original_img = imread("Lena.png");
size_2d = size(original_img);
height = size_2d(1);
width = size_2d(2);
origianl_img_R = original_img(:,:,1);
origianl_img_G = original_img(:,:,2);
origianl_img_B = original_img(:,:,3);
figure(1)
scatter3(origianl_img_R(:), origianl_img_G(:), origianl_img_B(:), 1); % (:)表示把二位矩阵拉平成一维矩阵
xlabel("R");
ylabel("G");
zlabel("B");
hold on;

% 使用Kmeans对原始图像进行聚类，并绘制聚类结果的三维像素散点图
pixels = double([origianl_img_R(:)  origianl_img_G(:) origianl_img_B(:)]);
[index,center] = kmeans(double(pixels),512, "MaxIter",200);

scatter3(center(:,1),center(:,2),center(:,3), 40, "filled", "red");
lgd=legend("Original","Centroids-512");
lgd.FontSize = 20;

% 还原压缩图像
compressed_vector = zeros(height*width,3);
for i=1:height*width
    compressed_vector(i,:) = round(center(index(i),:));
end
compressed_vector = round(compressed_vector);
compressed_img = uint8(reshape(compressed_vector,[height,width,3]));


% 显示原始图像与压缩图像
figure(2)
subplot(1,2,1), imshow(original_img);
t1=title('Original Image');
t1.FontSize=20;
subplot(1,2,2), imshow(compressed_img);
t2=title('Compressed-512 Image');
t2.FontSize=20;

% 显示原始图像与压缩图像各自的内存大小
fprintf('Original Image Memory Size = %d bytes\n',numel(original_img));
fprintf('Compressed Image Memory Size = %d bytes\n',numel(index) + numel(center));
fprintf('Compression ratio = %f\n', (numel(index) + numel(center))/numel(original_img));

toc;
