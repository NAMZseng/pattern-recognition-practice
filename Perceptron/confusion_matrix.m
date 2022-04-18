function confusion_matrix(y_gt,y_pre)

[mat,order] = confusionmat(y_gt,y_pre);
k=max(order); %k为分类的个数
 
 
imagesc(mat); %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
 
%#black and lower values are white)
textStrings = num2str(mat(:),'%d');       %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
 
 
%# Create x and y coordinates for the strings
[x,y] = meshgrid(1:k);  
hStrings=text(x(:),y(:),textStrings(:),'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors；
 
set(gca,'XTick',1:k,...                                    
        'XTickLabel',{'0','1','2','3','4','5','6','7','8','9'},...  %#   and tick labels
        'YTick',1:k,...
        'YTickLabel',{'0','1','2','3','4','5','6','7','8','9'},...
        'TickLength',[0 0]);

end