% for ext in extensions:
    {"${ext.identifier}", RISCVExtensionVersion{${ext.version.major}, ${ext.version.minor}}},
% endfor
