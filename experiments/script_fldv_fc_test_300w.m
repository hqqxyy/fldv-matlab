function script_fldv_fc_test_300w()

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_FLDV_FC_1_STAGE_CONV2_2_VOC2007;
% cache base
cache_base_proposal         = 'fldv_fc_vgg16_1_stage_conv2_2';

% test data
dataset                     = [];
dataset                     = Dataset.test_300w(dataset, 'test_common');

%% Final test
fprintf('\n***************\nfinal test\n***************\n');
     
dataset.roidb_test       	= Fldv_Train.do_fldv_fc_test(conf_proposal, model.stage2_rpn, dataset.imdb_test);

% save final models, for outside tester
Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);
end
