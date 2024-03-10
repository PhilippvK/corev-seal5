% for group in groups:
  bool hasExt${group.name}() const { return HasExt${group.name}; }
% for extension in group.extensions:
  bool hasExt${extension.name}() const { return HasExt${extension.name}; }
% endfor
% endfor
