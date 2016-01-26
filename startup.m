function startup()
% startup()
% --------------------------------------------------------
% fldv matlab
% Copyright (c) 2015, QIQI
% --------------------------------------------------------

    curdir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(curdir, 'utils')));
    addpath(genpath(fullfile(curdir, 'functions')));
    addpath(genpath(fullfile(curdir, 'experiments')));
    addpath(genpath(fullfile(curdir, 'imdb')));

    mkdir_if_missing(fullfile(curdir, 'datasets'));

    caffe_path = fullfile(curdir, '..', 'fldv-caffe', 'matlab', 'caffe');
    if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab; See README.md');
    end
    addpath(genpath(caffe_path));

    mkdir_if_missing(fullfile(curdir, 'imdb', 'cache'));

    mkdir_if_missing(fullfile(curdir, 'output'));

    mkdir_if_missing(fullfile(curdir, 'models'));

    fprintf('fldv startup done\n');
end
