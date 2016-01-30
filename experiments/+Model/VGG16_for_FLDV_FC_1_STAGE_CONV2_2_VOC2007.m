function model = VGG16_for_FLDV_FC_1_STAGE_CONV2_2_VOC2007(model)
% VGG 16layers (only finetuned from conv3_1)

model.mean_image                                = fullfile(pwd, 'models', 'pre_trained_models', 'vgg_16layers', 'mean_image');
model.pre_trained_net_file                      = fullfile(pwd, 'models', 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 2;


model.final_test.model_file        = fullfile(pwd, 'models', 'fldv_fc_prototxts', 'vgg_16layers_fc6', 'test.prototxt');
model.final_test.test_net_def_file        = fullfile(pwd, 'models', 'fldv_fc_prototxts', 'vgg_16layers_fc6', 'test.prototxt');

end