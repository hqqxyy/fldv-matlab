function Error = fldv_fc_test(conf, imdb, roidb, varargin)
% mAP = fldv_fc_test(conf, imdb, roidb, varargin)

% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb',                              @isstruct);
    ip.addRequired('roidb',                             @isstruct);
    ip.addRequired('stage_num',                         @isinteger);
    ip.addRequired('meanshape',                         @ismatrix);
    ip.addParamValue('imdb_init',       struct('gtshape', {}), @isstruct)
    ip.addParamValue('net_def_file',    '', 			@isstr);
    ip.addParamValue('net_file',        '', 			@isstr);
    ip.addParamValue('cache_name',      '', 			@isstr); 
    ip.addParamValue('ignore_cache',    false,          @islogical);
    
    ip.parse(conf, imdb, roidb, varargin{:});
    opts = ip.Results;
    

%%  set cache dir
    cache_dir = fullfile(pwd, 'output', 'fldv_fc_cachedir', opts.cache_name, imdb.name);
    mkdir_if_missing(cache_dir);

%%  init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['test_', timestamp, imdb.name,'.txt']);
    diary(log_file);
    
    num_videos = imdb.num_videos;
    
    try
      aboxes = cell(num_classes, 1);
      if opts.ignore_cache
          throw('');
      end
    catch    
%%      testing 
        % init caffe net
        caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
        caffe.init_log(caffe_log_file_base);
        caffe_net = caffe.Net(opts.net_def_file, 'test');
        caffe_net.copy_from(opts.net_file);

        % set random seed
        % prev_rng = seed_rand(conf.rng_seed);
        % caffe.set_random_seed(conf.rng_seed);

        % set gpu/cpu
        if conf.use_gpu
            caffe.set_mode_gpu();
        else
            caffe.set_mode_cpu();
        end             

        % determine the maximum number of rois in testing 
        % max_rois_num_in_gpu = check_gpu_memory(conf, caffe_net);

        disp('opts:');
        disp(opts);
        disp('conf:');
        disp(conf);
        
        count = 0;
        t_start = tic;
        rstshape = cell(num_videos, 1);
        gtshape = cell(num_videos, 1);
        for i = 1:num_videos
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_videos);
            num_frames = length(imdb.videos{i}.images);
            for j = 1:num_frames
                th = tic;
                im = imread(imdb.videos{i}.images{j});
                bbox = imdb.videos{i}.bboxes{j};
                dummy_gtshape = zeros(68, 2);
                [im_blob, gt_blob, init_blob, mean_blob] = get_blobs(im, bbox, dummy_gtshape, meanshape,...
                                                                             'use_resize',          false, ...
                                                                             'init_with_meanshape', false,...
                                                                             'init_imdb',           opts.imdb_init,...
                                                                             'image_means',         conf.image_means,...
                                                                             'bbox_shift_range',    conf.test_bbox_shift_range,...
                                                                             'batch_size',          conf.test_batch_size);
                
                im_blob = permute(im_blob, [2, 1, 3, 4]);
                im_blob = single(im_blob);
                
                gt_blob = gt_blob - 1;% c's index start from 0
                gt_blob = permute(gt_blob, [3, 2, 1]);
                
                init_blob = init_blob - 1;
                init_blob = permute(init_blob, [3, 2, 1]);
                
                mean_blob = mean_blob - 1;
                mean_blob = permute(mean_blob, [2, 1]);
                
                net_inputs = {im_blob, gt_blob, init_blob, mean_blob};
                
                caffe_net.reshape_as_input(net_inputs);
                output_blobs = caffe_net.forward(net_inputs);
                
                rstshape{i, 1}{j, 1} = output_blobs{2};
                gtshape{i, 1}{j, 1} = imdb.videos{i}.gtshapes{j};
                fprintf(' time: %.3fs\n', toc(th)); 
            end
        end

        fprintf('test all images in %f seconds.\n', toc(t_start));
        caffe.reset_all(); 
    end

    % ------------------------------------------------------------------------
    % Peform error evaluation
    % ------------------------------------------------------------------------
    rstError = cell(length{rstshape}, 1);
    meanError = [];
    for i = 1:length(rstshape)
        rstError{i, 1} = cell(length(rstshape{i, 1}), 1);
        for j = 1:length(rstshape{i, 1})
            tmp_gtshape = repmat(gtshape{i, 1}{j, 1}, 1, 1, conf.test_batch_size);
            rstError{i, 1}{j, 1} = compute_error(tmp_gtshape, rstshape{i, 1}{j, 1});
            meanError(end+1, 1) = mean(rstError{i, 1}{j, 1});
        end
    end
    
    fprintf('mean error : %f \n', mean(meanError));
    diary off;
end

function max_rois_num = check_gpu_memory(conf, caffe_net)
%%  try to determine the maximum number of rois

    max_rois_num = 0;
    for rois_num = 500:500:5000
        % generate pseudo testing data with max size
        im_blob = single(zeros(conf.max_size, conf.max_size, 3, 1));
        rois_blob = single(repmat([0; 0; 0; conf.max_size-1; conf.max_size-1], 1, rois_num));
        rois_blob = permute(rois_blob, [3, 4, 1, 2]);

        net_inputs = {im_blob, rois_blob};

        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);

        caffe_net.forward(net_inputs);
        gpuInfo = gpuDevice();

        max_rois_num = rois_num;
            
        if gpuInfo.FreeMemory < 2 * 10^9  % 2GB for safety
            break;
        end
    end

end


% ------------------------------------------------------------------------
function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, thresh)
% ------------------------------------------------------------------------
    % Keep top K
    X = cat(1, boxes{1:end_at});
    if isempty(X)
        return;
    end
    scores = sort(X(:,end), 'descend');
    thresh = scores(min(length(scores), top_k));
    for image_index = 1:end_at
        if ~isempty(boxes{image_index})
            bbox = boxes{image_index};
            keep = find(bbox(:,end) >= thresh);
            boxes{image_index} = bbox(keep,:);
            box_inds{image_index} = box_inds{image_index}(keep);
        end
    end
end