function bbox = loadbbox(path)
% function: load shape from pts file
file = fopen(path, 'r');
bbox = textscan(file, '%f');
fclose(file);
bbox = bbox{1}';