templates_root_path = '.\手写数字\templates\';
test_root_path = '.\手写数字\test\';

%%

% 预测test文件中每个图片的数值类别
for test_class = 0 : 9
    test_images_path = dir([test_root_path num2str(test_class) '\' '*.bmp']);
    test_images_num = size(test_images_path, 1);

    accurate_num = 0;

    for i = 1:test_images_num
        test_image = imread([test_images_path(i).folder '\' test_images_path(i).name]);
        % 将所有图片统一尺寸，同时数字识别不需要太高的分辨率，可以缩小尺寸从而减小计算复杂度
        test_image = imresize(test_image, [25, 25]);

        min_euler_distance = Inf;
        predict_class = -1;

        % 每张测试图片与template文件夹中所有的图片进行模板匹配
        for template_class = 0 : 9
            templates_path = dir([templates_root_path num2str(template_class) '\' '*.bmp']);
            templates_num = size(templates_path, 1);

            for j = 1 : templates_num
                template = imread([templates_path(j).folder '\' templates_path(j).name]);
                template = imresize(template, [25, 25]);

                % 计算测试图片与模板图片的欧拉距离
                diff = test_image - template;
                euler_distance = sqrt( sum( diff(:).*diff(:) ) );

                if(euler_distance < min_euler_distance)
                    min_euler_distance = euler_distance;
                    predict_class = template_class;
                end
            end
        end

       disp(['测试图片：', test_images_path(i).name, '的预测值为：', num2str(predict_class), '真实值为：', num2str(test_class)])
        if predict_class == test_class
            accurate_num = accurate_num + 1;
        end
    end

    disp(['数字',num2str(test_class),'：正确匹配数/总数: ',num2str(accurate_num),'/',num2str(test_images_num), ...
        '，准确率:', num2str(accurate_num./test_images_num)]);
end
