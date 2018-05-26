% all rights reserved
% author: Zhihua Ban
% contact: sawpara@126.com

function nvcc(varargin)

nvcc_customer = strjoin(varargin);



% host compiler
compilerForMex = mex.getCompilerConfigurations('C++','selected');
compiler_arch = strtrim(compilerForMex(1).Details.CommandLineShellArg);

if ispc && isempty(strfind(compiler_arch, '64'))
  error('we only suport 64-bit');
end

if ispc
  compiler_parent = fullfile(compilerForMex.Location, 'VC', 'bin', compiler_arch);
  host_compiler_opt = sprintf(' -ccbin "%s"', compiler_parent);
  host_compiler_flg = regexprep(compilerForMex.Details.CompilerFlags, '/[Ww][0-9]\s*', '');
else
  host_compiler_opt = ' ';
  host_compiler_flg = regexprep(compilerForMex.Details.CompilerFlags, '-std\S*\s*', '');
end  
host_compiler_opt = sprintf('%s -Xcompiler "%s"', host_compiler_opt, host_compiler_flg);
nvcc_command_line = ['nvcc', ' ', host_compiler_opt];


nvcc_command = sprintf('%s %s\n', nvcc_command_line, nvcc_customer);
status = system(nvcc_command);
if status < 0
  error('Error invoking nvcc');
end
  


