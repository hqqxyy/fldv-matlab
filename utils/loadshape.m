function shape = loadshape(path)
% function: load shape from pts file
file = fopen(path, 'r');
shape = textscan(file, '%f %f', 'HeaderLines', 3, 'CollectOutput', 2);
fclose(file);
shape = shape{1};