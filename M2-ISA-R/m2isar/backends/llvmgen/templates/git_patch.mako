diff --git a/${filename} b/${filename}
--- a/${filename}
+++ b/${filename}
% for patch in patches:
@@ -${patch[0]},${patch[1]} +${patch[0]},${patch[2]} @@ ${patch[3]}
${patch[4]}
% endfor
