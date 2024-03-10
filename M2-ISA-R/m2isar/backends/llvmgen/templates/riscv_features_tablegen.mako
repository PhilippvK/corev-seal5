% for group in groups:
// ---------------------
// ${group.name}
// ---------------------
% if len(group.desc) > 0:
// ${group.desc}
% endif

% for extension in group.extensions:
// ${extension.name}
% if len(group.desc) > 0:
// ${extension.desc}
% endif
<%
def camelcase(x):
    return x
%>
def FeatureExt${camelcase(extension.name)} : SubtargetFeature<"${extension.name.lower()}", "HasExt${camelcase(extension.name)}", "true", "'${extension.name.capitalize()}' (${extension.desc})">;

def HasExt${camelcase(extension.name)} : Predicate<"Subtarget->hasExt${camelcase(extension.name)}()">, AssemblerPredicate<(any_of FeatureExt${camelcase(extension.name)}), "'${extension.name.capitalize()}' (${extension.desc})">;

% endfor

// All ${group.name} Extensions
<%
temp = ", ".join([f"FeatureExt{x.name}" for x in group.extensions])
%>
def FeatureExt${camelcase(group.name)} : SubtargetFeature<"${group.name}", "HasExt${camelcase(group.name)}", "true", "'${group.name.capitalize()}' (${group.desc})", [${temp}]>;

def HasExt${camelcase(group.name)} : Predicate<"Subtarget->hasExt${camelcase(group.name)}()">, AssemblerPredicate<(any_of FeatureExt${camelcase(group.name)}), "'${group.name.capitalize()}' (${group.desc})">;
% endfor
