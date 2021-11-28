%% Initialization
clear all;
clc;

%% Parameters setting
angRes = 5;                 % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
factor = 4;                 % SR factor
patchsize = factor*32;  	% Spatial resolution of each SAI patch
stride = patchsize/2;       % stride between two patches. Default: 32
downRatio = 1/factor;
src_data_path = './datasets/';
src_datasets = dir(src_data_path);
src_datasets(1:2) = [];
num_datasets = length(src_datasets); 


%% Training data generation
for index_dataset = 1 : num_datasets
    idx_save = 0;
    name_dataset = src_datasets(index_dataset).name;
    src_sub_dataset = [src_data_path, name_dataset, '/training/'];
    folders = dir(src_sub_dataset);
    folders(1:2) = [];
    num_scene = length(folders); 
    
    for index_scene = 1 : num_scene 
        % Load LF image
        idx_scene_save = 0;
        name_scene = folders(index_scene).name;
        name_scene(end-3:end) = [];
        fprintf('Generating training data of Scene_%s in Dataset %s......\t\t', name_scene, name_dataset);
        data_path = [src_sub_dataset, name_scene];
        data = load(data_path);
        LF = data.LF; 
        [U, V, ~, ~, ~] = size(LF);
         
        % Extract central angRes*angRes views
        LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, 1:3); 
        [U, V, H, W, ~] = size(LF);
                
        % Generate patches of size 32*32
        for h = 1 : stride : H - patchsize + 1
            for w = 1 : stride : W - patchsize + 1
                idx_save = idx_save + 1;
                idx_scene_save = idx_scene_save + 1;
                Hr_SAI_y = single(zeros(U * patchsize, V * patchsize));
                Lr_SAI_y = single(zeros(U * patchsize * downRatio, V * patchsize * downRatio));             

                for u = 1 : U
                    for v = 1 : V     
                        x = (u-1) * patchsize + 1;
                        y = (v-1) * patchsize + 1;
                        
                        % Convert to YCbCr
                        patch_Hr_rgb = double(squeeze(LF(u, v, h : h+patchsize-1, w : w+patchsize-1, :)));
                        patch_Hr_ycbcr = rgb2ycbcr(patch_Hr_rgb);
                        patch_Hr_y = squeeze(patch_Hr_ycbcr(:,:,1)); 
                                                
                        patchsize_Lr = patchsize / factor;
                        Hr_SAI_y(x:x+patchsize-1, y:y+patchsize-1) = single(patch_Hr_y);
                        patch_Sr_y = imresize(patch_Hr_y, downRatio);
                        Lr_SAI_y((u-1)*patchsize_Lr+1 : u*patchsize_Lr, (v-1)*patchsize_Lr+1:v*patchsize_Lr) = single(patch_Sr_y);
         
                    end
                end

                SavePath = ['./data_for_training/SR_', num2str(angRes), 'x' , num2str(angRes), '_' ,num2str(factor), 'x/', name_dataset,'/' ];
                if exist(SavePath, 'dir')==0
                    mkdir(SavePath);
                end

                SavePath_H5 = [SavePath, num2str(idx_save,'%06d'),'.h5'];
                
                h5create(SavePath_H5, '/Lr_SAI_y', size(Lr_SAI_y), 'Datatype', 'single');
                h5write(SavePath_H5, '/Lr_SAI_y', single(Lr_SAI_y), [1,1], size(Lr_SAI_y));
                
                h5create(SavePath_H5, '/Hr_SAI_y', size(Hr_SAI_y), 'Datatype', 'single');
                h5write(SavePath_H5, '/Hr_SAI_y', single(Hr_SAI_y), [1,1], size(Hr_SAI_y));
                
            end
        end
        fprintf([num2str(idx_scene_save), ' training samples have been generated\n']);
    end
end



