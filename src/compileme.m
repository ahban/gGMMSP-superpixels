
% download the nvcc function from https://github.com/ahban/compilecuda
urlwrite('https://raw.githubusercontent.com/ahban/compilecuda/master/nvcc.m', 'nvcc.m');

% compile cuda files with nvcc 
if ispc
  nvcc -w -c -O2 -arch sm_30 -I. sp/gpu/extract_labels.cu
  nvcc -w -c -O2 -arch sm_30 -I. sp/gpu/initialize_theta.cu
  nvcc -w -c -O2 -arch sm_30 -I. sp/gpu/update_R.cu
  nvcc -w -c -O2 -arch sm_30 -I. sp/gpu/update_theta.cu 
  nvcc -w -c -O2 -arch sm_30 -I. su/cuda/colors.cu
else
  nvcc -w -c -O2 -std=c++11 -Xcompiler "-std=c++11" -arch sm_30 -I. sp/gpu/extract_labels.cu
  nvcc -w -c -O2 -std=c++11 -Xcompiler "-std=c++11" -arch sm_30 -I. sp/gpu/initialize_theta.cu
  nvcc -w -c -O2 -std=c++11 -Xcompiler "-std=c++11" -arch sm_30 -I. sp/gpu/update_R.cu
  nvcc -w -c -O2 -std=c++11 -Xcompiler "-std=c++11" -arch sm_30 -I. sp/gpu/update_theta.cu 
  nvcc -w -c -O2 -std=c++11 -Xcompiler "-std=c++11" -arch sm_30 -I. su/cuda/colors.cu
end



% find cuda root
% windows
if ispc
  nvcc_root = getenv('CUDA_PATH');
  lib_path = fullfile(nvcc_root, 'lib/x64');  
  obj_list = dir('*.obj');
else
  [~, nvcc_root] = system('which nvcc');
  nvcc_root = regexprep(nvcc_root, '/nvcc\S*\s*$', '/../');
  lib_path = fullfile(nvcc_root, 'lib64');
  obj_list = dir('*.o');
end

obj_list = struct2cell(obj_list);
obj_list = strjoin(obj_list(1, :));
inc_path = fullfile(nvcc_root, 'include');


mex_command = ['mex -O -largeArrayDims mx_gGMMSP.cpp ', obj_list, ' -lcudart -L', lib_path, ' -I', inc_path, ' -I.'];
fprintf('%s\n', mex_command);
eval(mex_command);


obj_list = strsplit(obj_list);

for i = 1:numel(obj_list)
  delete(obj_list{i});
end

