 %---------------------------------
% 使用广义感知器处理多分类问题
% 参考资料：《神经网络与深度学习》 邱锡鹏
% https://github.com/nndl/nndl.github.io/blob/master/old-chap/chap-%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B.pdf
%---------------------------------
clc;
tic;

train_images = cell2mat(struct2cell(load("train_images.mat")));
train_labels = cell2mat(struct2cell(load("train_labels.mat")));
test_images = cell2mat(struct2cell(load("test_images.mat")));
test_labels = cell2mat(struct2cell(load("test_labels.mat")));

% 将样本图片向量化，由（width，height，N）变为（width*height，N）
train_images_vec = reshape(train_images, [], length(train_images));
test_images_vec = reshape(test_images, [], length(test_images));

class = 1:10;
num_class = 10;
class_onehot = full(ind2vec(class, num_class));
% 将label表示为onehot向量，由（1，N）变为（num_class，N）
% 由于ind2vec函数输入不可有0,所以对标签统一加1，即0-9变成1-10
train_labels_onehot = full(ind2vec(train_labels+1, num_class));

epoch = 5;
% 感知机的训练与测试
weight = train(train_images_vec, train_labels_onehot, class_onehot, epoch);
test_predict_labels = test(weight,test_images_vec,class_onehot);

test_acc = length(find(test_labels==test_predict_labels)) / length(test_labels);
fprintf('Test Accuracy:%.2f\n',test_acc);

%绘制混淆矩阵
confusion_matrix(test_labels'+1,test_predict_labels'+1);
toc;
%% 训练广义感知器
function w = train(x,y,class_onehot,epoch)
% x:[D,N] , y:[c,N], class_onehot:[c,c]
% w:[D×c,1]
    d = length(x(:,1)); 
    num_class = length(class_onehot);
    w = zeros(d*num_class, 1); % 初始化权重向量
    N = length(x); %样本数
    for t = 1:epoch
        % 对训练样本进行随机排序
        rand_idx = randperm(N);
        x = x(:, rand_idx);
        y = y(:, rand_idx);
        for n = 1:N
            results = zeros(1,num_class);
            for i = 1:num_class
                % 求解特征函数φ(x,y) = vec(xy')
                phi = reshape(x(:,n) * class_onehot(:, i)', [], 1);
                results(i) = dot(w, phi);
            end
           % argmax
           [~, y_predict] = max(results);
           [~, y_gt] = max(y(:,n));
           if y_gt ~= y_predict
               phi_gt = reshape(x(:,n) * y(:,n)', [], 1);
               phi_pre = reshape(x(:,n) * class_onehot(:, y_predict)', [], 1);
               w = w + phi_gt - phi_pre;
           end
        end
    end    
end
%%
function y_predict_list = test(w,x,class_onehot)
    num_class = length(class_onehot);
    N = length(x); %样本数
    y_predict_list = zeros(1,N);
    for n = 1:N
        results = zeros(1,num_class);
        for i = 1:num_class
             phi = reshape(x(:,n) * class_onehot(:, i)', [], 1);
             results(i) = dot(w, phi);
        end
       [~, y_predict] = max(results);
       y_predict_list(:,n) = y_predict - 1; %1-10变为0-9
    end
end
