function dataset = test_300w(dataset, usage)

root_dir = '/home/qiqi/data/face_alignment/300w/';
switch usage
    case {'train'}
        dataset.imdb_train     = imdb_from_300w(root_dir, 'train', true) ;
        % dataset.roidb_train    = dataset.imdb_train.roidb_func(dataset.imdb_test);
    case {'test_common'}
        dataset.imdb_test     = imdb_from_300w(root_dir, 'test_common', false) ;
        % dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    case {'test_challenging'}
        dataset.imdb_test     = imdb_from_300w(root_dir, 'test_challenging', false) ;
        % dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    otherwise
        error('usage = "train", "test_common" or "test_challenging"');
end

end