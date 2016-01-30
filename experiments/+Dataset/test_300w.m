function dataset = test_300w(dataset, usage)

switch usage
    case {'train'}

    case {'test_common'}
        dataset.imdb_test     = imdb_from_300w('test') ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    case {'test_challenging'}
        dataset.imdb_test     = imdb_from_300w('test') ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    otherwise
        error('usage = "train", "test_common" or "test_challenging"');
end

end