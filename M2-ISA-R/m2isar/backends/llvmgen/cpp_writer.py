# SPDX-License-Identifier: Apache-2.0
#
# This file is part of the M2-ISA-R project: https://github.com/tum-ei-eda/M2-ISA-R
#
# Copyright (C) 2022
# Chair of Electrical Design Automation
# Technical University of Munich

"""Actual text output functions for functions and instructions."""

# import re
import logging

from mako.template import Template

from .templates import template_dir
from .types import ImmOperand

from .tablegen_writer import _create_groups

logger = logging.getLogger("isa_writer")


# def write_isa_info_patch(arch_versions, experimental=False):
def write_isa_info_patches(extensions):

	normal_extensions = [ext for ext in extensions if not ext.experimental]
	experimental_extensions = [ext for ext in extensions if ext.experimental]

	isa_info_template = Template(filename=str(template_dir / 'riscv_isa_info_cpp.mako'))
	ext_text = isa_info_template.render(extensions=normal_extensions)
	experimental_ext_text = ""
	if len(experimental_extensions) > 0:
		raise NotImplementedError
		isa_info_experiemental_template = Template(filename=str(template_dir / 'riscv_isa_info_experimental_cpp.mako'))
		experimental_ext_text = isa_info_experimental_template.render(extensions=experimental_extensions)
	return ext_text, experimental_ext_text
	# if experimental:
	# 	raise NotImplementedError
	# 	isa_info_template = Template(filename=str(template_dir / 'riscv_isa_info_experimental_cpp.mako'))
	# else:
	# 	isa_info_template = Template(filename=str(template_dir / 'riscv_isa_info_cpp.mako'))

	# out_str = ""

	# logger.info("writing patch for RISCVISAInfo.cpp")

	# patch_template = Template(filename=str(template_dir / 'git_patch.mako'))

# 	def get_patch_contents():
# 		content_text = isa_info_template.render(arch_versions=arch_versions)
# 		content_text = content_text.rstrip("\n")
# 		content_pre = """    {"zicbop", RISCVExtensionVersion{1, 0}},
#
#     {"svnapot", RISCVExtensionVersion{1, 0}},"""
# 		content_pre = "\n".join([f" {line}" for line in content_pre.split("\n")])
# 		content_main = "\n".join([f"+{line}" for line in content_text.split("\n")])
# 		content_post = """};
#
# static const RISCVSupportedExtension SupportedExperimentalExtensions[] = {"""
# 		content_post = "\n".join([f" {line}" for line in content_post.split("\n")])
# 		content = "\n".join([content_pre, content_main, content_post])
# 		label = ""
# 		start = 103
# 		length_before = len(content_pre.split("\n")) + len(content_post.split("\n"))
# 		length_after = length_before + len(content_main.split("\n"))
# 		return (start, length_before, length_after, label, content)

# 	patches = [get_patch_contents()]
# 	filename = "llvm/lib/Support/RISCVISAInfo.cpp"
# 	patch_text = patch_template.render(filename=filename, patches=patches)
#
# 	return patch_text


def write_subtarget_patch(features):
	groups = _create_groups(features)

	patch_template = Template(filename=str(template_dir / 'git_patch.mako'))

	def get_private_patch():
		content_template = Template(filename=str(template_dir / 'riscv_subtarget_private_cpp.mako'))
		content_text = content_template.render(groups=groups)
		content_text = content_text.rstrip("\n")
		content_pre = """  bool HasStdExtZmmul = false;
  bool HasStdExtZawrs = false;
  bool HasStdExtZtso = false;"""
		content_pre = "\n".join([f" {line}" for line in content_pre.split("\n")])
		content_main = "\n".join([f"+{line}" for line in content_text.split("\n")])
		content_post = """  bool HasRV32 = false;
  bool HasRV64 = false;
  bool IsRV32E = false;"""
		content_post = "\n".join([f" {line}" for line in content_post.split("\n")])
		content = "\n".join([content_pre, content_main, content_post])
		label = ""
		start = 89
		length_before = len(content_pre.split("\n")) + len(content_post.split("\n"))
		length_after = length_before + len(content_main.split("\n"))
		return (start, length_before, length_after, label, content)

	def get_public_patch():
		content_template = Template(filename=str(template_dir / 'riscv_subtarget_public_cpp.mako'))
		content_text = content_template.render(groups=groups)
		content_text = content_text.rstrip("\n")
		content_pre = """  bool hasStdExtZawrs() const { return HasStdExtZawrs; }
  bool hasStdExtZmmul() const { return HasStdExtZmmul; }
  bool hasStdExtZtso() const { return HasStdExtZtso; }"""
		content_pre = "\n".join([f" {line}" for line in content_pre.split("\n")])
		content_main = "\n".join([f"+{line}" for line in content_text.split("\n")])
		content_post = """  bool is64Bit() const { return HasRV64; }
  bool isRV32E() const { return IsRV32E; }
  bool enableLinkerRelax() const { return EnableLinkerRelax; }"""
		content_post = "\n".join([f" {line}" for line in content_post.split("\n")])
		content = "\n".join([content_pre, content_main, content_post])
		label = ""
		start = 187
		length_before = len(content_pre.split("\n")) + len(content_post.split("\n"))
		length_after = length_before + len(content_main.split("\n"))
		return (start, length_before, length_after, label, content)

	patches = [get_private_patch(), get_public_patch()]
	filename = "llvm/lib/Target/RISCV/RISCVSubtarget.h"
	patch_text = patch_template.render(filename=filename, patches=patches)

	return patch_text


def write_asm_parser_patch(field_types):
	lines = []
	for ty in field_types:
		assert isinstance(ty, ImmOperand)
		line = f"  bool is{ty.sign_letter_upper}Imm{ty.bits}() {{ return Is{ty.sign_letter_upper}Imm<{ty.bits}>(); }};"
		lines.append(line)
	return "\n".join(lines)
