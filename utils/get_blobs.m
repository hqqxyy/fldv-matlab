function [im_blob, gt_blob, init_blob, mean_blob] = get_blobs(conf, img, bbox, gtshape, varargin)

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                                      @isstruct);
    ip.addRequired('img',                                       @ismatrix);
    ip.addRequired('bbox',                                      @ismatrix);
    ip.addRequired('gtshape',                                   @ismatrix);
    ip.addRequired('meanshape',                                 @ismatrix);
    
    ip.addParamValue('use_resize',          false,          @islogical);     
    ip.addParamValue('scales',              480,         	@isscalar);
    ip.addParamValue('max_size',            640,            @isscalar);
    
    ip.addParamValue('use_meanshape',       false,          @islogical);  
    ip.addParamValue('bbox_shift_range',    0.2,            @isscalar);  
    ip.addParamValue('batch_size',          256,            @isscalar);  
    
    ip.addParamValue('imdb_init',       struct('gtshape', {}), @isstruct);
    
    ip.parse(conf, imdb, roidb, varargin{:});
    opts = ip.Results;
    
    
    
    
    
    
    
    
    
    
    
    


end