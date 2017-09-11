
clear
clc
close all


addpath src

image = imread('images\240p.png');

% call gGMMSP
label = mx_gGMMSP(image, 10);

% show the results
bound = display_superpixels(label, image);

figure; imshow(image); title('original image')
figure; imshow(bound); title('superpixel boundaries');



