% for group in groups:
  bool HasExt${group.name} = false;
% for extension in group.extensions:
  bool HasExt${extension.name} = false;
% endfor
% endfor
