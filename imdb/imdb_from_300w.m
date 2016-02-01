function imdb = imdb_from_300w(root_dir, image_set, flip)
% imdb = imdb_from_voc(root_dir, image_set, year)

% for convience, we name helen, lfpw... as videos

%imdb.name = 'voc_train_2007'
%imdb.videos.images = {'/path/to/image/1', '/path/to/image/2', ...}
%imdb.videos.bboxes = {[23 45 67 89], [12 46 79 80] ....} % left top right bottom
%imdb.videos.gtshapes = {[1 1; 2 2; ...], [1 1; 2 2; ...], ...} gtshapes
%
%imdb.num_videos
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

cache_file = ['./imdb/cache/imdb_300w_' image_set];
if flip
    cache_file = [cache_file, '_flip'];
end

try
  load(cache_file);
catch
  imdb.name = ['300w_' image_set];
  switch image_set
      case {'train'}
          video_list = loadTxtLines([root_dir, 'train/videolist.txt']);
      case {'test_common'}
          video_list = loadTxtLines([root_dir, 'test_common/videolist.txt']);
      case {'test_challenging'}
          video_list = loadTxtLines([root_dir, 'test_challenging/videolist.txt']);
      otherwise
          error('image_set = "train", "test_common" or "test_challenging"')
  end
  
  imdb.flip = flip;
  if flip
      flip_video_list = cell(size(video_list));
      for i = 1:length(video_list)
          flip_video_list{i} = [video_list{i}, '_flip'];
      end
      video_list = cat(1, video_list, flip_video_list);
  end
  
  videos = cell(length(video_list), 1);
  image_at = @(video_name, frame_id) sprintf([root_dir, image_set, '/', '%s/images/%.6d.jpg'], video_name, frame_id);
  bbox_at = @(video_name, frame_id) sprintf([root_dir, image_set, '/', '%s/boxes/%.6d.txt'], video_name, frame_id);
  gtshape_at = @(video_name, frame_id) sprintf([root_dir, image_set, '/', '%s/pts/%.6d.pts'], video_name, frame_id);
  for i = 1:length(video_list)
      imlist = dir([root_dir, image_set, '/', video_list{i}, '/images/*.jpg']);
      num_frames = length(imlist);
      images = cell(num_frames, 1);
      bboxes = cell(num_frames, 1);
      gtshapes = cell(num_frames, 1);
      for j = 1:num_frames
          images{j} = image_at(video_list{i}, j);
          bboxes{j} = loadbbox(bbox_at(video_list{i}, j));
          gtshapes{j} = loadshape(gtshape_at(video_list{i}, j));
      end
      videos{i}.images = images;
      videos{i}.bboxes = bboxes;
      videos{i}.gtshapes = gtshapes;
  end
  imdb.videos = videos;
  imdb.num_videos = length(video_list);

  imdb.eval_func = @imdb_eval_300w;
  imdb.roidb_func = @roidb_from_300w;
  
  for i = 1:length(imdb.videos)
      tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.videos));
      im = imread(imdb.videos{i}.images{1});
      imsize = size(im);
      imdb.sizes(i, :) = imsize(1:2);
  end
  
  fprintf('Saving imdb to cache...');
  save(cache_file, 'imdb', '-v7.3');
  fprintf('done\n');
end

function lines = loadTxtLines(path)
% function: load shape from pts file
fid = fopen(path, 'r');
lines = cell(0, 1);
line = fgetl(fid);
while ischar(line)
    lines{end + 1, 1} = line;
    line = fgetl(fid);
end
fclose(fid);


