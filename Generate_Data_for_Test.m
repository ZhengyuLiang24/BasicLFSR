%% Initialization
clear all;
clc;

%% Parameters setting
angRes = 5;                 % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
factor = 4;                 % SR factor
downRatio = 1/factor;
src_data_path = './datasets/';
src_datasets = dir(src_data_path);
src_datasets(1:2) = [];
num_datasets = length(src_datasets); 


%% Test data generation
for index_dataset = 1 : num_datasets 
    idx_save = 0;
    name_dataset = src_datasets(index_dataset).name;
    src_sub_dataset = [src_data_path, name_dataset, '/test/'];
    scenes = dir(src_sub_dataset);
    scenes(1:2) = [];
    num_scene = length(scenes); 
    
    for index_scene = 1 : num_scene 
        % Load LF image
        idx_scene_save = 0;
        name_scene = scenes(index_scene).name;
        name_scene(end-3:end) = [];
        fprintf('Generating test data of Scene_%s in Dataset %s......\t\t', name_scene, src_datasets(index_dataset).name);
        data_path = [src_sub_dataset, name_scene];
        data = load(data_path);
        LF = data.LF;
        [U, V, H, W, ~] = size(LF);
        while mod(H, 4) ~= 0
            H = H - 1;
        end
        while mod(W, 4) ~= 0
            W = W - 1;
        end
        
        % Extract central angRes*angRes views
        LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), 1:H, 1:W, 1:3); % Extract central angRes*angRes views
        [U, V, H, W, ~] = size(LF);
    
        % Convert to YCbCr
        idx_save = idx_save + 1;
        idx_scene_save = idx_scene_save + 1;
        Hr_SAI_y = single(zeros(U * H, V * W));
        Lr_SAI_y = single(zeros(U * H * downRatio, V * W * downRatio));           
        Sr_SAI_cbcr = single(zeros(U * H, V * W, 2));
    
        for u = 1 : U
            for v = 1 : V
                x = (u-1)*H+1;
                y = (v-1)*W+1;
                
                temp_Hr_rgb = double(squeeze(LF(u, v, :, :, :)));
                temp_Hr_ycbcr = rgb2ycbcr(temp_Hr_rgb);
                tmp = ycbcr2rgb(temp_Hr_ycbcr);
                Hr_SAI_y(x:u*H, y:v*W) = single(temp_Hr_ycbcr(:,:,1));
                
                temp_Hr_y = squeeze(temp_Hr_ycbcr(:,:,1));
                temp_Lr_y = imresize(temp_Hr_y, downRatio);
                Lr_SAI_y((u-1)*H*downRatio+1 : u*H*downRatio, (v-1)*W*downRatio+1:v*W*downRatio) = single(temp_Lr_y);                  
                
                tmp_Hr_cbcr = temp_Hr_ycbcr(:,:,2:3);
                tmp_Lr_cbcr = imresize(tmp_Hr_cbcr, 1/factor);
                tmp_Sr_cbcr = imresize(tmp_Lr_cbcr, factor);
                Sr_SAI_cbcr(x:u*H, y:v*W, :) = tmp_Sr_cbcr;
            end
        end 
        
        SavePath = ['./data_for_test/SR_', num2str(angRes), 'x' , num2str(angRes), '_' ,num2str(factor), 'x/', name_dataset,'/' ];
        if exist(SavePath, 'dir')==0
            mkdir(SavePath);
        end

        SavePath_H5 = [SavePath, name_scene,'.h5'];
        
        h5create(SavePath_H5, '/Hr_SAI_y', size(Hr_SAI_y), 'Datatype', 'single');
        h5write(SavePath_H5, '/Hr_SAI_y', single(Hr_SAI_y), [1,1], size(Hr_SAI_y));
        
        h5create(SavePath_H5, '/Lr_SAI_y', size(Lr_SAI_y), 'Datatype', 'single');
        h5write(SavePath_H5, '/Lr_SAI_y', single(Lr_SAI_y), [1,1], size(Lr_SAI_y));
        
        h5create(SavePath_H5, '/Sr_SAI_cbcr', size(Sr_SAI_cbcr), 'Datatype', 'single');
        h5write(SavePath_H5, '/Sr_SAI_cbcr', single(Sr_SAI_cbcr), [1,1,1], size(Sr_SAI_cbcr));

        fprintf([num2str(idx_scene_save), ' test samples have been generated\n']);
    end
end

