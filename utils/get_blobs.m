function [im_blob, gt_blob, init_blob, mean_blob] = get_blobs(img, bbox, gtshape, varargin)

%% inputs
    ip = inputParser;
    % ip.addRequired('conf',                                      @isstruct);
    ip.addRequired('img',                                       @ismatrix);
    ip.addRequired('bbox',                                      @ismatrix);
    ip.addRequired('gtshape',                                   @ismatrix);
    ip.addRequired('meanshape',                                 @ismatrix);
    
    ip.addParamValue('use_resize',          false,                      @islogical);     
    ip.addParamValue('scales',              480,                        @isscalar);
    ip.addParamValue('max_size',            640,                        @isscalar);
    ip.addParamValue('image_means',         [128.68, 116.779, 103.939], @ismatrix); %VGG RGB
        
    ip.addParamValue('init_with_meanshape', false,          @islogical);  
    ip.addParamValue('imdb_init',       struct('gtshape', {}), @isstruct);
  
    ip.addParamValue('bbox_shift_range',    0.2,            @isscalar);  
    ip.addParamValue('batch_size',          256,            @isscalar);  
    
    ip.addParamValue('ignore_initshape',    false,          @islogical);  
    ip.addParamValue('ignore_meanshape',    false,          @islogical);  
    
  
    ip.parse(conf, imdb, roidb, varargin{:});
    opts = ip.Results;
    
    % im blob
    if opts.use_resize
        scales = opts.scales;
        max_size = opts.max_size;
    else
        imsize = size(img);
        scales = imsize(1);
        max_size = imsize(2);
        assert(scales <= max_size, 'imsize(1) should be less than imsize(2)');
    end    
    [im_blob, im_scale] = prep_im_for_blob(img, opts.image_means, scales, max_size);
    
    im_height = size(im_blob, 1);
    im_width = size(im_blob, 2);
    % gt blob
    gt_blob = zeros(opts.batch_size, size(gtshape, 1), size(gtshape, 2), 'single');
    for i = 1:opts.batch_size
        gt_blob(i, :, :) = gtshape * im_scale;
    end
    
    %init blob
    if opts.ignore_initshape
        init_blob = [];
    else
        init_blob = zeros(opts.batch_size, size(gtshape, 1), size(gtshape, 2), 'single');
        bbox = bbox * im_scale;
        bbox_ltwh = getShiftedBbox(bbox, opts.bbox_shift_range, opts.batch_size, im_height, im_width);
        if opts.init_with_meanshape
            for i = 1:batch_size
                init_blob(i, :, :) = backshape(meanshape, bbox_ltwh(i, :)); 
            end
        else
            for i = 1:batch_size
                video_id = randi(opts.imdb_init.num_videos, 1);
                frame_id = randi(length(opts.imdb_init.videos{video_id}.gtshapes), 1);
                init_shape = opts.imdb_init.videos{video_id}.gtshapes{frame_id};
                norm_init_shape = normshape(init_shape);
                init_blob(i, :, :) = backshape(norm_init_shape, bbox_ltwh(i, :));
            end
        end                    
    end
    
    %mean blob
    if opts.ignore_meanshape
        mean_blob = [];
    else
        mean_blob = single(meanshape);
    end
end

function dst = normshape(src)

l = min(src(:, 2));
t = min(src(:, 1));
r = max(src(:, 2));
b = max(src(:, 1));

w = r - l + 1;
h = b - t + 1;

np = size(src, 1);
dst = (src - repmat([l, t], np, 1) ) ./ (repmat([h, w], np, 1));
end

function dst = backshape(src, bbox)
np = size(src, 1);
dst = src * repmat(bbox([4:3]), np, 1) + repmat(bbox([2 1]), np, 1);
end

function bbox_ltwh = getShiftedBbox(bbox, bbox_shift_range, batch_size, im_height, im_width)
bbox_ltwh = bbox;
bbox_ltwh(3:4) = bbox(3:4) - bbox(1:2) + 1;
bbox_ltwh = repmat(bbox_ltwh, batch_size, 1);
shift_factor = 2 * bbox_shift_range * (rand(batch_size, 2) - 0.5);
bbox_ltwh(:, 1:2) = bbox_ltwh(:, 1:2) + bbox_ltwh(:, 3:4) .* shift_factor;
bbox_ltrb = bbox_ltwh;
bbox_ltrb(:, 3:4) = bbox_ltwh(:, 1:2) + bbox_ltwh(:, 3:4) - 1;
bbox_ltrb(bbox_ltrb < 1) = 1;
bbox_r = bbox_ltrb(:, 3);
bbox_r(bbox_r > im_width) = im_width;
bbox_b = bbox_ltrb(:, 4);
bbox_b(bbox_b > im_height) = im_height;
bbox_ltrb(:, 3) = bbox_r;
bbox_ltrb(:, 4) = bbox_b;
bbox_ltwh = bbox_ltrb;
bbox_ltwh(:, 3:4) = bbox_ltrb(:, 3:4) - bbox_ltrb(:, 1:2) + 1;
end
