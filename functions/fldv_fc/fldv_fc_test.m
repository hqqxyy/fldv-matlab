function mAP = fldv_fc_test(conf, imdb, roidb, varargin)
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
        for i = 1:num_videos
            count = count + 1;
            fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_videos);
            num_frames = length(imdb.videos{i}.images);
            for j = 1:num_frames
                im = imread(imdb.videos{i}.images{j});
                box = imdb.videos{i}.bboxes{j};
                dummy_gtshape = zeros(68, 2);
                [im_blob, gtshape_blob, initshape_blob, meanshape_blob] = get_blobs(conf, imdb_train, im, box, dummy_gtshape, model.meanshape)
            end
            th = tic;
            im = imread(imdb.image_at(i));

            [boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im, d.boxes, max_rois_num_in_gpu);

            for j = 1:num_classes
                inds = find(~d.gt & scores(:, j) > thresh(j));
                if ~isempty(inds)
                    [~, ord] = sort(scores(inds, j), 'descend');
                    ord = ord(1:min(length(ord), max_per_image));
                    inds = inds(ord);
                    cls_boxes = boxes(inds, (1+(j-1)*4):((j)*4));
                    cls_scores = scores(inds, j);
                    aboxes{j}{i} = [aboxes{j}{i}; cat(2, single(cls_boxes), single(cls_scores))];
                    box_inds{j}{i} = [box_inds{j}{i}; inds];
                else
                    aboxes{j}{i} = [aboxes{j}{i}; zeros(0, 5, 'single')];
                    box_inds{j}{i} = box_inds{j}{i};
                end
            end

            fprintf(' time: %.3fs\n', toc(th));  

            if mod(count, 1000) == 0
                for j = 1:num_classes
                [aboxes{j}, box_inds{j}, thresh(j)] = ...
                    keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
                end
                disp(thresh);
            end    
        end

        for j = 1:num_classes
            [aboxes{j}, box_inds{j}, thresh(j)] = ...
                keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
        end
        disp(thresh);

        for i = 1:num_classes

            top_scores{i} = sort(top_scores{i}, 'descend');  
            if (length(top_scores{i}) > max_per_set)
                thresh(i) = top_scores{i}(max_per_set);
            end

            % go back through and prune out detections below the found threshold
            for j = 1:length(imdb.image_ids)
                if ~isempty(aboxes{i}{j})
                    I = find(aboxes{i}{j}(:,end) < thresh(i));
                    aboxes{i}{j}(I,:) = [];
                    box_inds{i}{j}(I,:) = [];
                end
            end

            save_file = fullfile(cache_dir, [imdb.classes{i} '_boxes_' imdb.name opts.suffix]);
            boxes = aboxes{i};
            inds = box_inds{i};
            save(save_file, 'boxes', 'inds');
            clear boxes inds;
        end
        fprintf('test all images in %f seconds.\n', toc(t_start));
        
        caffe.reset_all(); 
        rng(prev_rng);
    end

    % ------------------------------------------------------------------------
    % Peform AP evaluation
    % ------------------------------------------------------------------------

    if isequal(imdb.eval_func, @imdb_eval_voc)
        for model_ind = 1:num_classes
          cls = imdb.classes{model_ind};
          res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, opts.cache_name, opts.suffix);
        end
    else
    % ilsvrc
        res = imdb.eval_func(aboxes, imdb, opts.cache_name, opts.suffix);
    end

    if ~isempty(res)
        fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
        fprintf('Results:\n');
        aps = [res(:).ap]' * 100;
        disp(aps);
        disp(mean(aps));
        fprintf('~~~~~~~~~~~~~~~~~~~~\n');
        mAP = mean(aps);
    else
        mAP = nan;
    end
    
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