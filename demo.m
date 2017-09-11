
clear
clc
close all


addpath src

image_name = 'images\240p.png';

image = imread(image_name);

% call gGMMSP
label = mx_gGMMSP(image, 10);

% show the results
bound = display_superpixels(label, image);

figure; imshow(image); title('original image')
figure; imshow(bound); title('superpixel boundaries');

% write to file
[~, image_stem, ~] = fileparts(image_name);
superpixel_bound_name = fullfile('result', [image_stem, '.png']);
fprintf('Save the boundries to %s.\n', superpixel_bound_name);
imwrite(bound, superpixel_bound_name);

