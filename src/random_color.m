function rdcolor = random_color(label)

label = label + 1;
ul = unique(label);

mm = max(ul);

rdc = uint8(255*rand(mm, 3));

C = unique(rdc, 'rows');

if (size(C, 1) ~= numel(ul))
  warning('not same a.......');
end

[H, W] = size(label);

rdcolor = uint8(zeros(H, W, 3));

for y = 1:H
  for x = 1:W
    lab = label(y,x);
    rdcolor(y,x,1) = rdc(lab,1);
    rdcolor(y,x,2) = rdc(lab,2);
    rdcolor(y,x,3) = rdc(lab,3);
  end
end

end