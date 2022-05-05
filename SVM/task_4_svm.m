%---------------------------
% 使用SVM解决二分类兵王问题
%---------------------------
clc;
tic;

%% 数据读取
fid = fopen('krkopt.data');
vec = zeros(6, 1);
xapp = []; % 用来存放棋子的位置
yapp = []; % 用来存放棋局结果
while ~feof(fid) %检测流上的文件结束符，如果文件结束，则返回非0值，否则返回0
    string = []; %用来存放每次读取的一行六个数据
    c = fread(fid, 1); %每次读取一个字节的数据
    while c ~= 10 %每行读取结束后，最后一个字节数据为10
        string = [string, c]; %将读取到的数据保存到string[]中
        c = fread(fid, 1); % 继续读取下一个字节的数据
    end
    if length(string) > 10
        vec(1) = string(1) - 96; %字符a的ascii码值为96，数字0ascii码值为48
        vec(2) = string(3) - 48;
        vec(3) = string(5) - 96;
        vec(4) = string(7) - 48;
        vec(5) = string(9) - 96;
        vec(6) = string(11) - 48;
        xapp = [xapp, vec]; %将转换后的数据存入xapp[]
        if string(13) == 100 %判断每行最后一个单词首字母是否为小写的d，是则标记为1，表示和棋，否则标记为-1
            yapp = [yapp, 1];
        else
            yapp = [yapp, -1];
        end
    end
end
fclose(fid);

%% 数据预处理
[N, M] = size(xapp);
p = randperm(M); %打乱数据样本
numberOfSamplesForTraining = 5000;
xTraining = [];
yTraining = [];
for i = 1:numberOfSamplesForTraining
    xTraining = [xTraining, xapp(:, p(i))];
    yTraining = [yTraining, yapp(p(i))];
end
xTraining = xTraining';
yTraining = yTraining';

xTesting = [];
yTesting = [];
for i = numberOfSamplesForTraining + 1:M
    xTesting = [xTesting, xapp(:, p(i))];
    yTesting = [yTesting, yapp(p(i))];
end
xTesting = xTesting';
yTesting = yTesting';


%标准化，每个样本都减去均值再除以方差，使得数据的均值为0，方差为1
[numVec, numDim] = size(xTraining);
avgX = mean(xTraining);
stdX = std(xTraining);
for i = 1:numVec
    xTraining(i, :) = (xTraining(i, :) - avgX) ./ stdX;
end
[numVec, numDim] = size(xTesting);
avgX = mean(xTesting);
stdX = std(xTesting);
for i = 1:numVec
    xTesting(i, :) = (xTesting(i, :) - avgX) ./ stdX;
end

%% 模型训练
%首先需要对C和Gamma两个参数的取值进行初步搜索，c的取值范围是：2^-5--2^15,gamma的取值范围：2^-15--2^3,该范围是基于人工的经验；
%对数据进行交叉验证，初步找出识别率最高的c与gamma的组合
CScale = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15];
gammaScale = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3];
C = 2.^CScale;
gamma = 2.^gammaScale;
maxRecognitionRate = 0;
for i = 1:length(C)
    for j = 1:length(gamma)
        cmd = ['-t 2 -c ', num2str(C(i)), ' -g ', num2str(gamma(j)), ' -v 5']; %设置支持向量机的参数
        recognitionRate = svmtrain(yTraining, xTraining, cmd);
        if recognitionRate > maxRecognitionRate
            maxRecognitionRate = recognitionRate;
            maxCIndex = i;
            maxGammaIndex = j;
        end
    end
end

%进一步缩小搜索范围，再次进行交叉验证，找出识别率最高的更精确的c与gamma的组合
n = 10;
minCScale = 0.5 * (CScale(max(1, maxCIndex-1)) + CScale(maxCIndex));
maxCScale = 0.5 * (CScale(min(length(CScale), maxCIndex+1)) + CScale(maxCIndex));
newCScale = [minCScale:(maxCScale - minCScale) / n:maxCScale];

minGammaScale = 0.5 * (gammaScale(max(1, maxGammaIndex-1)) + gammaScale(maxGammaIndex));
maxGammaScale = 0.5 * (gammaScale(min(length(gammaScale), maxGammaIndex+1)) + gammaScale(maxGammaIndex));
newGammaScale = [minGammaScale:(maxGammaScale - minGammaScale) / n:maxGammaScale];
newC = 2.^newCScale;
newGamma = 2.^newGammaScale;
maxRecognitionRate = 0;
for i = 1:length(newC)
    for j = 1:length(newGamma)
        cmd = ['-t 2 -c ', num2str(newC(i)), ' -g ', num2str(newGamma(j)), ' -v 5']; %设置支持向量机的参数
        recognitionRate = svmtrain(yTraining, xTraining, cmd);
        if recognitionRate > maxRecognitionRate
            maxRecognitionRate = recognitionRate;
            maxC = newC(i);
            maxGamma = newGamma(j);
        end
    end
end

%使用最优的c与gamma的组合训练数据，保存结果
cmd = ['-t 2 -c ', num2str(maxC), ' -g ', num2str(maxGamma)];
model = svmtrain(yTraining, xTraining, cmd); %#ok<*SVMTRAIN> 
save model.mat model;
save xTesting.mat xTesting;
save yTesting.mat yTesting;


%% 加载训练结果并进行测试，保存结果
load model.mat;
[yPred, accuracy, decisionValues] = svmpredict(yTesting, xTesting, model);
save yPred.mat yPred;

toc;

%% 绘制混淆矩阵
mat = confusionmat(yTesting,yPred);
cm = confusionchart(mat,["-1", "1"]);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
