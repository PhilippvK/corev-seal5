#!/bin/python3
# inject_extension.py
#
# Given LLVM(15-ish) source ornamented with site markers and the fragments
# of code generated from the CoreDSL2TableGen tool, inserts those fragments

import os
import argparse
import yaml
import subprocess
from email.utils import formatdate


# class injectionSite:
#     siteTag = ''
#     fullPath = ''
#     startMark = ''
#     endMark = ''
#     # line + len magic values to give 0,0 for new files
#     siteLine = -1
#     siteLen = 0
#     payload = ''
#
#     def __init__(self, siteTag, fullPath, payload):
#         self.siteTag = siteTag
#         self.fullPath = fullPath
#         self.payload = payload
#
#     def hasPatchSite(self):
#         return self.siteTag
#
#     def findSite(self):
#         self.startMark = self.siteTag + ' - INSERTION_START'
#         self.endMark = self.siteTag + ' - INSERTION_END'
#         with open(self.fullPath) as src:
#             lines = src.readlines()
#             for line in lines:
#                 if line.find(self.startMark) != -1:
#                     self.startMark = line
#                     self.siteLine = lines.index(line)
#                 elif line.find(self.endMark) != -1:
#                     self.endMark = line.rstrip('\n')
#                     self.siteLen = lines.index(line) - self.siteLine + 1
#                     break
#                 elif self.siteLine >= 0:
#                     # Accumulate all lines between the start mark and the end
#                     # as part of the mark, so the new lines get injected just
#                     # before the end mark
#                     self.startMark += line
#         return self.siteLen > 0
#
#
# class patchSet:
#     patches = {}
#
#     def add(self, filename, filepath, payload):
#         self.patches[filename] = injectionSite(filename, filepath, payload)
#
#     def findSites(self):
#         for patch in self.patches.values():
#             if patch.hasPatchSite() and not patch.findSite():
#                 raise ValueError('Start mark for injecting in {0} not found'.format(patch.fullPath))
#
#     def generatePatches(self):
#         ps = ''
#         # Build minimal patch header for 'git am'
#         userName = "bob" #subprocess.check_output(['git', 'config', 'user.name']).decode('utf-8').strip()
#         userEmail = "bob@bob.com" #subprocess.check_output(['git', 'config', 'user.email']).decode('utf-8').strip()
#         ps += 'From: {0} <{1}>\n'.format(userName, userEmail)
#         ps += 'Date: {0}\n'.format(formatdate())
#         ps += 'Subject: [PATCH] Instructions injected\n\n\n'
#
#         for patch in self.patches.values():
#             insPayload = '+' + patch.payload.replace('\n', '\n+')
#             if patch.hasPatchSite():
#                 # Updating existing file
#                 origFile = 'a/' + patch.fullPath
#                 newFile = 'b/' + patch.fullPath
#                 newStart = patch.siteLine + 1
#                 newLen = patch.payload.count('\n') + 1 + patch.siteLen
#                 # ensure all existing lines in the match prefixed by a space
#                 patch.startMark = patch.startMark.rstrip('\n').replace('\n', '\n ')
#                 insPayload = ' {0}\n{1}\n {2}'.format(patch.startMark, insPayload, patch.endMark)
#             else:
#                 # Adding new file
#                 origFile = '/dev/null'
#                 newFile = 'b/' + patch.fullPath
#                 newStart = 1
#                 newLen = patch.payload.count('\n') + 1
#             ps += '''--- {0}
# +++ {1}
# @@ -{2},{3} +{4},{5} @@
# {6}
# '''.format(     origFile,
#                 newFile,
#                 patch.siteLine + 1,
#                 patch.siteLen,
#                 newStart,
#                 newLen,
#                 insPayload )
#         return ps
#
# # The sole argument is a YAML file describing a list of extensions to process, of the format:
# #
# # - extension
# #   name: (short name for extension, used as identifier and for enabling target feature)
# #   description: (short human-readable description for display)
# #   instructions: (path to .td file specifying instructions)
# #   BuiltinsRISCV.def: (path to table fragment for Builtins entries)
# #   CGBuiltin.cpp: (path to code fragment for CGBuiltin switch)
# #   IntrinsicsRISCV.td: (path to table fragment for IntrinsicsRISCV entries)
# #
# # The extension name needn't be unique, e.g. two sets of extension instructions can share the
# # same name. Then the clang switch "-target-feature +x(extension name)" will enable both sets.
# # The name will prefixed with "x" and made lower-case as a control switch, e.g. "S4EMAC" becomes
# # "xs4emac" as an option name.
parser = argparse.ArgumentParser(description="Generate patch file for adding extension to LLVM")
parser.add_argument("index_file", help="list of code fragments to inject")
parser.add_argument("--llvm-dir", type=str, default=None, help="path to llvm repo (default: cwd)")
parser.add_argument("--out-file", type=str, default=None, help="write patchset to file (default: print to stdout)")
parser.add_argument("--author", type=str, default="bob", help="author name in patchset")
parser.add_argument("--mail", type=str, default="bob@bob.com", help="author mail in patchset")
parser.add_argument("--msg", type=str, nargs="+", default="[PATCH] Instructions injected", help="custom text for commit message")
parser.add_argument("--append", "-a", action="store_true", help="whether existing patches should be appended instead of beeing overwritten")
args = parser.parse_args()

base_dir = os.path.dirname(args.index_file)

with open(args.index_file) as file:
    index = yaml.safe_load(file)

global_artifacts = index["artifacts"]

def generate_patch_set(fragments):
    ps = ''
    # Build minimal patch header for 'git am'
    userName = args.author
    userEmail = args.mail
    ps += 'From: {0} <{1}>\n'.format(userName, userEmail)
    ps += 'Date: {0}\n'.format(formatdate())
    ps += 'Subject: {0}\n\n\n'.format(" ".join(args.msg))
    for fragment in fragments:
        ps += fragment
    return ps


def find_site(path, key):
    # print("find_site", path, key)
    # startMark = key + ' - INSERTION_START'
    # endMark = key + ' - INSERTION_END'
    if key:
        startMark_ = path.split("/")[-1] + ' - ' + key + ' - INSERTION_START'
        endMark_ = path.split("/")[-1] + ' - ' + key + ' - INSERTION_END'
    else:
        startMark_ = path.split("/")[-1] + ' - INSERTION_START'
        endMark_ = path.split("/")[-1] + ' - INSERTION_END'
    # print("start", startMark_)
    # print("end", endMark_)
    siteLine = -1
    siteLen = 0
    startMark = None
    fullpath = path if args.llvm_dir is None else os.path.join(args.llvm_dir, path)
    with open(fullpath, "r") as src:
        lines = src.readlines()
        for line in lines:
            if line.find(startMark_) != -1:
                startMark = line
                siteLine = lines.index(line)
            elif line.find(endMark_) != -1:
                endMark = line.rstrip('\n')
                siteLen = lines.index(line) - siteLine + 1
                break
            elif siteLine >= 0:
                # Accumulate all lines between the start mark and the end
                # as part of the mark, so the new lines get injected just
                # before the end mark
                startMark += line
    if startMark is None:
        # fallback
        if key:
            return find_site(path, None)
        else:
            assert False, "Marker not found!"
    return siteLine, siteLen, startMark, endMark


def generate_patch_fragment(artifact):
    assert "path" in artifact
    is_file = False
    is_patch = False
    if "key" in artifact:  # NamedPatch
        assert "src" in artifact
        is_patch = True
    elif "start" in artifact and "end" in artifact:
        raise NotImplementedError
    elif "line" in artifact:
        raise NotImplementedError
    else:  # File
        is_file = True
    with open(os.path.join(base_dir, artifact["src"] if is_patch else artifact["path"]), "r") as f:
        content_ = f.read()
    content = '+' + content_.replace('\n', '\n+')
    if is_patch:
        # Updating existing file
        origFile = 'a/' + artifact["path"]
        newFile = 'b/' + artifact["path"]
        siteLine, siteLen, startMark, endMark = find_site(artifact["path"], artifact["key"])
        newStart = siteLine + 1
        newLen = content_.count('\n') + 1 + siteLen
        # ensure all existing lines in the match prefixed by a space
        startMark = startMark.rstrip('\n').replace('\n', '\n ')
        content = ' {0}\n{1}\n {2}'.format(startMark, content, endMark)
    else:
        # Adding new file
        origFile = '/dev/null'
        newFile = 'b/' + artifact["path"]
        newStart = 1
        siteLen = 0
        siteLine = -1
        newLen = content_.count('\n') + 1
    return '''--- {0}
+++ {1}
@@ -{2},{3} +{4},{5} @@
{6}
'''.format(origFile,
             newFile,
             siteLine + 1,
             siteLen,
             newStart,
             newLen,
             content )

fragments = []

def process_artifacts(artifacts):
    ret = []
    for artifact in artifacts:
        fragment = generate_patch_fragment(artifact)
        ret.append(fragment)
    return ret


fragments.extend(process_artifacts(global_artifacts))


#
# patches = patchSet();
# patchFiles = [
#     ['BuiltinsRISCV.def', 'clang/include/clang/Basic/BuiltinsRISCV.def'],
#     ['CGBuiltin.cpp', 'clang/lib/CodeGen/CGBuiltin.cpp'],
#     ['IntrinsicsRISCV.td', 'llvm/include/llvm/IR/IntrinsicsRISCV.td']
# ]
# extnNames = {}
# RISCVInstrInfo = ''
for extn in index["extensions"]:
    # Accumulate the list of extension namese
    # extnNames[extn['name']] = extn['description']
    fragments.extend(process_artifacts(extn["artifacts"]))


patch_set = generate_patch_set(fragments)

if args.out_file:
    with open(args.out_file, "w") as f:
        f.write(patch_set)
else:
    print(patch_set)
    # Include the instuction files
    # instrFilename = os.path.basename(extn['instructions'])
    # RISCVInstrInfo += r'include "{0}"'.format(instrFilename)
#     filePath = os.path.join(os.path.dirname(args.index_file), extn['instructions'])
#     with open(filePath) as file:
#         payload = file.read().rstrip()
#     patches.add('', os.path.join('llvm/lib/Target/RISCV', instrFilename), payload)
#
#     # include the file fragments for the per-instruction patches
#     for fileTag in patchFiles:
#         if fileTag[0] in extn:
#           fileName = extn[fileTag[0]]
#           fileName = os.path.join(os.path.dirname(args.index_file), fileName)
#           with open(fileName) as file:
#               payload = file.read()
#           patches.add(fileTag[0], fileTag[1], payload)
#
# # Build the extension-name-dependent patches
# RISCVISAInfo = ''
# RISCV = ''
# RISCVSubTargetPrivate = ''
# RISCVSubTargetPublic = ''
# for extn in extnNames.items():
#     if extn[0][0].lower().startswith("x"):
#         extnOpt = extn[0].lower()
#     else:
#         extnOpt = "x" + extn[0].lower()
#     RISCVISAInfo += r'    {{"{0}", RISCVExtensionVersion{{0, 1}}}},'.format(extnOpt)
#     RISCV += r"""
# def FeatureExt{1}
#     : SubtargetFeature<"{0}", "HasExt{1}", "true",
#                        "'{0}'">;
# def HasExt{1} : Predicate<"Subtarget->hastExt{1}()">,
#                              AssemblerPredicate<(all_of FeatureExt{1}),
#                              "'{0}' ({2})">;
# """.format(extnOpt, extn[0], extn[1])
#     RISCVSubTargetPrivate += r'  bool HasExt{0} = false;'.format(extn[0])
#     RISCVSubTargetPublic += r'  bool hasExt{0}() const {{ return HasExt{0}; }}'.format(extn[0])
#
# # Apply the extension-name-dependent patches
# patches.add('RISCVISAInfo.cpp', 'llvm/lib/Support/RISCVISAInfo.cpp', RISCVISAInfo)
# patches.add('RISCVFeatures.td', 'llvm/lib/Target/RISCV/RISCVFeatures.td', RISCV)
# patches.add('RISCVInstrInfo.td', 'llvm/lib/Target/RISCV/RISCVInstrInfo.td', RISCVInstrInfo)
#
# # Find injection locations
# patches.findSites()
#
# # Generate patch file
# print(patches.generatePatches())
