# SPDX-License-Identifier: Apache-2.0
#
# This file is part of the M2-ISA-R project: https://github.com/tum-ei-eda/M2-ISA-R
#
# Copyright (C) 2022
# Chair of Electrical Design Automation
# Technical University of Munich

"""Actual text output functions for functions and instructions."""

import re
import logging

from mako.template import Template

from ...metamodel import arch, behav

from .templates import template_dir

logger = logging.getLogger("tablegen_writer")

class Operand:
	def __init__(self, name, lower, upper):
		self.name = name
		# self._name = name
		self.lower = lower
		self.upper = upper

	@property
	def length(self):
		return (self.upper - self.lower + 1) if self.upper >= self.lower else (self.lower - self.upper + 1)

	# @property
	# def name(self):
	# 	ty, name_ = self._name.split(":")
	# 	if self.lower == 1:
	# 		ty += "_lsb0"
	# 	ret = f"{ty}:{name_}"
	# 	return ret

	def __repr__(self):
		return f"Operand({self.name}, {self.lower}, {self.upper})"

class EncodingField:
	def __init__(self, name, start, length, value=None, extra=None):
		self.name = name
		self.start = start
		self.length = length

		self.value = value
		self.const = value is not None
		self.extra = extra if extra is not None else value

	def __repr__(self):
		return f"EncodingField({self.name}, {self.start}, {self.length}, {self.value}, {self.const})"


def process_encoding(enc):
	# print("get_encoding", enc)
	operands = {}
	for e in reversed(enc):
		if isinstance(e, arch.BitField):
			name = e.name
			if name in operands:
				op = operands[name]
				op.lower = min(e.range.lower, op.lower)
				op.upper = max(e.range.upper, op.upper)
			else:
				op = Operand(name, e.range.lower, e.range.upper)
				operands[name] = op
	fields = []
	start = 0
	for e in reversed(enc):
		if isinstance(e, arch.BitField):
			name = e.name
			assert name in operands
			op = operands[name]
			if op.length > e.range.length:
				if e.range.length == 1:
					name = name + "{" + str(e.range.lower - op.lower) + "}"
				else:
					name = name + "{" + str(e.range.upper - op.lower) + "-" + str(e.range.lower - op.lower) + "}"
			new = EncodingField(name, start, e.range.length)
			start += e.range.length
		elif isinstance(e, arch.BitVal):
			lower = 0
			new = EncodingField(None, start, e.length, e.value)
			start += e.length
		else:
			assert False
		fields.insert(0, new)
	# print("fields", fields)
	# input(">")
	return operands.values(), fields


def write_riscv_instruction_info(name, real_name, asm_str, ins_str, outs_str, enc, fields, details_str, attrs={}, constraints=[], formats=False):
	# print("write_instruction")
	# print("disass", disass)
	# print("enc", enc)
	# print("fields", fields)

	# TODO: do merge matching predicates!

	if formats:
		instr_template = Template(filename=str(template_dir/'instr_tablegen2.mako'))
	else:
		instr_template = Template(filename=str(template_dir/'instr_tablegen.mako'))

	# asm_str = "x1, $uimmL, $uimmS"

	operands, fields = process_encoding(enc)

	# encoding = [
  #       Encoding("funct7", 25, 7, 0b1010000),
  #       Encoding("rs2", 20, 5),
  #       Encoding("rs1", 15, 5),
  #       Encoding("funct3", 12, 3, 0b101),
  #       Encoding("rd", 7, 5),
  #       Encoding("opcode", 0, 7, 0b1111011),
  #   ]

	# operands = [x for x in encoding if not x.const]
	# fields = [x for x in encoding if x.const and x.name != "opcode" and x.extra is None]
	# temp = ", ".join([f"bits<{field.length}> {field.name}" for field in fields])  # TODO

	# ins_str = "uimm5:$uimmS, uimm12pcrel:$uimmL"
	# print("reads", reads)
	# print("writes", writes)
	# input("<>")
	# print("reads_", reads_)
	# print("ins_str", ins_str)
	# print("outs_str", outs_str)

	attrs = {key: int(value) if isinstance(value, bool) else value for key, value in attrs.items()}

	constraints_str = ", ".join(constraints)

	out_str = instr_template.render(name=name, real_name=real_name, xlen=32, asm_str=asm_str, ins_str=ins_str, outs_str=outs_str, sched_str=str([]), operands=operands, fields=fields, attrs=attrs, constraints_str=constraints_str)

	if len(details_str) > 0:
		out_str = details_str + " in {\n" + "\n".join([("  " + line) if len(line) > 0 else line for line in out_str.split("\n")]) + "\n} // " + details_str + "\n"

	logger.info("writing TableGen for instruction %s", name)

	return out_str


def combine_instruction_info(tablegen_per_insn, predicates_str):
	# print("combine_instruction_info")
	out_str = "\n".join(tablegen_per_insn.values())
	if len(predicates_str) > 0:
		out_str = predicates_str + " in {\n" + "\n".join([("  " + line) if len(line) > 0 else line for line in out_str.split("\n")]) + "\n} // " + predicates_str + "\n"
	return out_str


def combine_patterns(tablegen_patterns, predicates_str):
	# print("combine_patterns")
	out_str = "\n".join(tablegen_patterns)
	if len(predicates_str) > 0:
		out_str = predicates_str + " in {\n" + "\n".join([("  " + line) if len(line) > 0 else line for line in out_str.split("\n")]) + "\n} // " + predicates_str + "\n"
	return out_str


class Extension:
    def __init__(self, name, full_name, desc=None, comment=None):
        self.name = name
        self.full_name = full_name
        if desc:
            self.desc = desc
        else:
            if self.full_name:
                self.desc = f"{full_name} Extension"
            else:
                self.desc = None
        self.comment = comment


class Group:
    def __init__(self, name, full_name, desc=None, comment=None):
        self.name = name
        self.full_name = full_name
        if desc:
            self.desc = desc
        else:
            if self.full_name:
                self.desc = f"{full_name} Extension Group"
            else:
                self.desc = None
        self.comment = comment
        self.extensions = []

    def add(self, x):
        self.extensions.append(x)


def _create_groups(features):
	group = Group("Generated", "Generated Instructions")  # TODO: replace
	extensions = [Extension(feature, None) for feature in features]
	group.extensions = extensions
	groups = [group]
	return groups


def write_riscv_features_patch(features):
	content_template = Template(filename=str(template_dir / 'riscv_features_tablegen.mako'))
	groups = _create_groups(features)

	content_text = content_template.render(groups=groups)
	content_text = content_text.rstrip("\n")
	return content_text
	# print("content_text", content_text)
	# patch_template = Template(filename=str(template_dir / 'git_patch.mako'))

	#	content_pre = """// tuning CPU names.
	#def Feature32Bit
	#    : SubtargetFeature<"32bit", "HasRV32", "true", "Implements RV32">;"""
	#	content_pre = "\n".join([f" {line}" for line in content_pre.split("\n")])
	#	content_main = "\n".join([f"+{line}" for line in content_text.split("\n")])
	#	content_post = """def Feature64Bit
	#    : SubtargetFeature<"64bit", "HasRV64", "true", "Implements RV64">;
	#def IsRV64 : Predicate<"Subtarget->is64Bit()">,"""
	#	content_post = "\n".join([f" {line}" for line in content_post.split("\n")])
	#	content = "\n".join([content_pre, content_main, content_post])
	#	filename = "llvm/lib/Target/RISCV/RISCV.td"
	#	label = ""
	#	start = 410
	#	length_before = len(content_pre.split("\n")) + len(content_post.split("\n"))
	#	length_after = length_before + len(content_main.split("\n"))
	#	patch_text = patch_template.render(filename=filename, patches=[(start, length_before, length_after, label, content)])

	return patch_text


def write_riscv_instruction_info_patch(includes):
	# patch_template = Template(filename=str(template_dir / 'git_patch.mako'))

	#	content_pre = """include \"RISCVInstrInfoV.td\"
	#include \"RISCVInstrInfoZfh.td\"
	#include \"RISCVInstrInfoZicbo.td\""""
	#	content_pre = "\n".join([f" {line}" for line in content_pre.split("\n")])
	content_text = "\n".join([f"include \"{inc}\"" for inc in includes])
	return content_text
	# content_main = "\n".join([f"+{line}" for line in content_text.split("\n")])
	# content_post = None
	# content = "\n".join([content_pre, content_main, *([content_post] if content_post is not None else [])])
	# filename = "llvm/lib/Target/RISCV/RISCVInstrInfo.td"
	# label = ""
	# start = 1778
	# length_before = len(content_pre.split("\n")) + (len(content_post.split("\n")) if content_post is not None else 0)
	# length_after = length_before + len(content_main.split("\n"))
	# patch_text = patch_template.render(filename=filename, patches=[(start, length_before, length_after, label, content)])
	# return patch_text
