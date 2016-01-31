function conf = fldv_fc_config(varargin)

    ip = inputParser;
    
    %% training
    ip.addParamValue('use_gpu',             gpuDeviceCount > 0, ...            
                                                        @islogical);
    % use resize,                                                                                                
    ip.addParamValue('use_resize',          false,                      @islogical);                                                  
    % Image scales -- the short edge of input image                                                                                                
    ip.addParamValue('scales',              600,                        @isscalar);
    % Max pixel size of a scaled input image
    ip.addParamValue('max_size',            1000,                       @isscalar);
    % Images per batch, only supports ims_per_batch = 1 currently
    ip.addParamValue('ims_per_batch',       1,                          @isscalar);
    % Minibatch size of one image
    ip.addParamValue('batch_size',          256,                        @isscalar);
    % bbox_shift_range
    ip.addParamValue('bbox_shift_range',    0.2,                        @isscalar);    
    % init by meanshape?
    ip.addParamValue('use_meanshape',       false,                      @isscalar);    
    
    % mean image, in RGB order
    ip.addParamValue('image_means',         [128.68, 116.779, 103.939],	@ismatrix); %VGG RGB
    % Use horizontally-flipped images during training?
    ip.addParamValue('use_flipped',         false,                      @islogical);
    % Stride in input image pixels at ROI pooling level (network specific)
    % 16 is true for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
    ip.addParamValue('feat_stride',         2,                          @isscalar);
    % train proposal target only to labled ground-truths or also include
    % other proposal results (selective search, etc.)
    ip.addParamValue('target_only_gt',      true,                       @islogical);
    % random seed                    
    ip.addParamValue('rng_seed',            6,                          @isscalar);

    
    %% testing
    % use resize,                                                                                                
    ip.addParamValue('test_use_resize',     false,          @islogical);        
    ip.addParamValue('test_scales',         600,            @isscalar);
    ip.addParamValue('test_max_size',       1000,           @isscalar);
    % if we test with 256 initshapes 
    ip.addParamValue('test_batch_size',     256,            @isscalar);
    % test bbox_shift_range
    ip.addParamValue('test_bbox_shift_range',    0.05,      @isscalar);    
    % test init by meanshape?
    ip.addParamValue('test_use_meanshape',       false,     @isscalar);    
    % test use flip image?
    ip.addParamValue('test_use_flipped',     true,     @isscalar);    
 
    ip.parse(varargin{:});
    conf = ip.Results;
    
    assert(conf.ims_per_batch == 1, 'currently rpn only supports ims_per_batch == 1');
    
    % if image_means is a file, load it
    if ischar(conf.image_means)
        s = load(conf.image_means);
        s_fieldnames = fieldnames(s);
        assert(length(s_fieldnames) == 1);
        conf.image_means = s.(s_fieldnames{1});
    end
end