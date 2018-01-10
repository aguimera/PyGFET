close all
clear all


% Files = dir('/home/aguimera/UserGuimeraLocal/SGFETs/Experimentals/171031/TimePlot/*.mat');
Files=dir('C:\Users\eduard\Dropbox (GAB GBIO)\GAB GBIO Team Folder\Experimentals\171010\TimePlot\Mats');
VdsRow = 19;
VgsRow = 20;

for ii = 1:length(Files)
   if strcmp(Files(ii).name(end),'t')
    clear('out','outh')
    FileName = ['C:\Users\eduard\Dropbox (GAB GBIO)\GAB GBIO Team Folder\Experimentals\171010\TimePlot\Mats\' Files(ii).name];

    load(FileName);

    fi = fieldnames(ds);

    for i=1:length(fi)
        if strcmp(fi{i},'Properties')
            disp('f')   
        else
            outh{i}=fi{i};
            out(:,i) = ds.(fi{i});
        end
    end

    FileName
    size(out)
    out(:,2)=(out(:,2)-(out(:,VdsRow)-out(:,VgsRow)))/10e3;

    save([FileName  '.mat'],'out','outh')
   else
   end    
end