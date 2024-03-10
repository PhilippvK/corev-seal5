# SPDX-License-Identifier: Apache-2.0
#
# This file is part of the M2-ISA-R project: https://github.com/tum-ei-eda/M2-ISA-R
#
# Copyright (C) 2022
# Chair of Electrical Design Automation
# Technical University of Munich

"""Viewer tool to visualize an M2-ISA-R model hierarchy."""

import re
import argparse
import logging
import pathlib
import pickle
import yaml
from typing import Optional, List
from enum import Enum
# import tkinter as tk
from collections import defaultdict
# from tkinter import ttk
from anytree import RenderTree

from m2isar.backends.viewer.utils import TreeGenContext

# Metamodel
from ...metamodel import arch, utils, patch_model
from ...metamodel.utils.expr_preprocessor import (process_functions,
                                                  process_instructions)

# Utilities
from .utils import VisitorContext, DAGBuilderContext, EliminateScalarsContext, CollectThrowsContext, CollectRWContext, AnalysisContext, get_asm_tokens
from .types import *  # TODO: dont do this
from . import tablegen_writer, cpp_writer, visitor, dag_builder, collect_reads_writes, analyze_behavior

# Transformations
from .transform import (
	eliminate_mod_rfs,
	eliminate_cmp_zero,
	eliminate_gpr_index,
	fold_field_sign,
	annotate_field_type,
	eliminate_scalar_assignments,
	find_unused_scalars,
	eliminate_unused_scalars,
	fold_slice_conv,
	# insert_powers,
	detect_slices,
	find_minmax,
	detect_arith_shift,
	type_inference,
	simplify_trivial_slice,
	simplify_modulo,
	simplify_conds,
	replace_slices,
	simplify_typed_binary_operations,
	eliminate_throws,
)


logger = logging.getLogger("llvmgen")



def get_legalizations_for_ext(ext_names):
	ret = {}
	for ext_name in ext_names:
		ret[ext_name] = []
		if ext_name == "XCoreVAlu":
			for node in [ISD.SETLE, ISD.SETULE]:
				cc_action = CondCodeAction(node, MVT.XLenVT, LegalizeAction.Legal)
				ret[ext_name].append(cc_action)
			for node in [ISD.ABS]:
				op_action = OperationAction(node, MVT.XLenVT, LegalizeAction.Legal)
				ret[ext_name].append(op_action)
			for node in [ISD.SMIN, ISD.UMIN, ISD.SMAX, ISD.UMAX]:
				op_action = OperationAction(node, MVT.XLenVT, LegalizeAction.Legal)
				ret[ext_name].append(op_action)
			for node in [ISD.SIGN_EXTEND_INREG]:
				for vt in [MVT.i8, MVT.i16]:
					op_action = OperationAction(node, vt, LegalizeAction.Legal)
					ret[ext_name].append(op_action)
		elif ext_name == "XCoreVMem":
			for is_store in [False, True]:
				for vt in [MVT.i8, MVT.i16, MVT.i32]:
					indexed_action = IndexedModeAction(ISD.POST_INC, vt, LegalizeAction.Legal, is_store=is_store)
					ret[ext_name].append(indexed_action)
		elif ext_name == "XCoreVSimd":
			# TODO: get rid of this hack
			ret[(ext_name, "XCoreVAlu")] = []
			for node in [ISD.SIGN_EXTEND_INREG]:
				for vt in [MVT.i8, MVT.i16]:
					op_action = OperationAction(node, vt, LegalizeAction.Expand)
					ret[(ext_name, "XCoreVAlu")].append(op_action)
			for node in [ISD.SIGN_EXTEND_INREG]:
				for vt in [MVT.i8, MVT.i16]:
					op_action = OperationAction(node, vt, LegalizeAction.Legal)
					ret[ext_name].append(op_action)
			# TODO: defaults
			for vt in [MVT.v4i8, MVT.v2i16]:
				op_action = OperationAction([ISD.ADD, ISD.SUB, ISD.AND, ISD.OR, ISD.XOR, ISD.SHL, ISD.SRA, ISD.SRL, ISD.BITCAST, ISD.VECREDUCE_ADD, ISD.SPLAT_VECTOR, ISD.SETCC, ISD.VECTOR_SHUFFLE, ISD.EXTRACT_VECTOR_ELT, ISD.BUILD_VECTOR, ISD.UNDEF, ISD.ABS, ISD.UMAX, ISD.UMIN, ISD.SMAX, ISD.SMIN], vt, LegalizeAction.Legal)
				ret[ext_name].append(op_action)
			op_action = OperationAction(ISD.SETCC, MVT.v4i1, LegalizeAction.Promote)
			ret[ext_name].append(op_action)
			op_action = OperationAction(ISD.SETCC, MVT.v2i1, LegalizeAction.Promote)
			ret[ext_name].append(op_action)
			# TODO: move to class
			promoted_type = OperationPromotedToType(ISD.SETCC, MVT.v4i1, MVT.v4i8)
			ret[ext_name].append(promoted_type)
			promoted_type = OperationPromotedToType(ISD.SETCC, MVT.v2i1, MVT.v2i16)
			ret[ext_name].append(promoted_type)
			for supported_vt in [MVT.v4i8, MVT.v2i16]:
				for other_vt in INTEGER_FIXEDLEN_VECTOR_VALUETYPES:
					if other_vt == supported_vt:
						continue
					trunc_store_action = TruncStoreAction(supported_vt, other_vt, LegalizeAction.Expand)
					ret[ext_name].append(trunc_store_action)
					load_ext_action = LoadExtAction([ISD.EXTLOAD, ISD.SEXTLOAD, ISD.ZEXTLOAD], supported_vt, other_vt, LegalizeAction.Expand)
					ret[ext_name].append(load_ext_action)
	return ret


def gen_legalization_tablegen(ext_actions):
	out = ""
	def join(inp: List[ISD]):
		assert isinstance(inp, list)
		assert len(inp) > 0
		if len(inp) == 1:
			return inp[0]
		else:
			return "{" + ", ".join(list(map(str, inp))) + "}"

	for exts, actions in ext_actions.items():
		if len(actions) == 0:
			continue
		if isinstance(exts, str):
			exts = tuple([exts])
		assert isinstance(exts, tuple)
		def util(ext):
			return ext.lower().replace("xcorev", "xcv")  # TODO: get from class
		temp = [f"Subtarget.hasExt{util(ext)}()" for ext in exts]
		temp_joined = " || ".join(temp)
		out += f"  if ({temp_joined}) {{\n"
		for action in actions:
			if isinstance(action, OperationAction):
				out += f"    setOperationAction({join(action.nodes)}, {action.vt}, {action.action});\n"
			elif isinstance(action, IndexedModeAction):
				if action.is_store:
					out += f"    setIndexedStoreAction({join(action.nodes)}, {action.vt}, {action.action});\n"
				else:
					out += f"    setIndexedLoadAction({join(action.nodes)}, {action.vt}, {action.action});\n"
			elif isinstance(action, CondCodeAction):
				out += f"    setCondCodeAction({join(action.nodes)}, {action.vt}, {action.action});\n"
			elif isinstance(action, OperationPromotedToType):
				out += f"    setOperationPromotedToType({join(action.nodes)}, {action.vt}, {action.vt_});\n"
			elif isinstance(action, TruncStoreAction):
				out += f"    setTruncStoreAction({action.vt}, {action.vt_}, {action.action});\n"
			elif isinstance(action, LoadExtAction):
				out += f"    setLoadExtAction({join(action.nodes)}, {action.vt}, {action.vt_}, {action.action});\n"
			else:
				raise RuntimeError(f"Invalid action: {action}")

		out += "  }\n"
	return out




def sort_instruction(entry: "tuple[tuple[int, int], arch.Instruction]"):
	"""Instruction sort key function. Sorts most restrictive encoding first."""
	(code, mask), _ = entry
	return bin(mask).count("1"), code
	#return code, bin(mask).count("1")

def main():
	"""Main app entrypoint."""

	# read command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('top_level', help="A .m2isarmodel file containing the models to generate.")
	parser.add_argument("--output-dir", "-o", default=None, help="TODO")
	parser.add_argument('-s', '--separate', action='store_true', help="Generate separate .cpp files for each instruction set.")
	parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
	parser.add_argument("--template", choices=["RISCV"], default=None, required=True, help="TODO")
	parser.add_argument("--core", type=str, nargs=1, action="append", default=[], help="TODO")
	parser.add_argument("--set", type=str, nargs=1, action="append", default=[], help="TODO")
	parser.add_argument("--insn", type=str, nargs=1, action="append", default=[], help="TODO")
	parser.add_argument("--config", "--yaml", type=str, nargs=1, action="append", default=[], help="TODO")
	parser.add_argument("--experimental", action='store_true', help="TODO")
	# parser.add_argument("--skip-patterns", action='store_true', help="TODO")
	parser.add_argument("--gen-dlr-yaml", action="store_true", help="TODO")
	parser.add_argument("--gen-features", action="store_true", help="TODO")
	parser.add_argument("--gen-insn-formats", action="store_true", help="TODO")
	parser.add_argument("--gen-insn-info", action="store_true", help="TODO")
	parser.add_argument("--gen-isel-patterns", action="store_true", help="TODO")
	parser.add_argument("--gen-asm-parser", action="store_true", help="TODO")
	parser.add_argument("--gen-isa-info", action="store_true", help="TODO")
	parser.add_argument("--gen-legalizations", action="store_true", help="TODO")
	parser.add_argument("--gen-costs", action="store_true", help="TODO")
	parser.add_argument("--gen-transform-info", action="store_true", help="TODO")
	parser.add_argument("--gen-index", action="store_true", help="TODO")
	args = parser.parse_args()
	args.core = sum(args.core, [])
	args.set = sum(args.set, [])
	args.insn = sum(args.insn, [])

	if args.gen_legalizations:
		ext_actions = {}
	if args.gen_transform_info:
		raise NotImplementedError  # TODO: make this a warning
	if args.gen_costs:
		raise NotImplementedError  # TODO: make this a warning

	configs = {}  # TODO: transform to class
	for yaml_file in sum(args.config, []):
		with open(yaml_file, "r") as f:
			yaml_data = yaml.safe_load(f)
		# merging nor supported!
		for key in yaml_data.keys():
			if key in configs.keys():
				logger.warning(f"Instruction set '{key}' defined muliple times in provided YAML files. Old values will be overwritten!")
		configs.update(yaml_data)

	assert args.template == "RISCV", "Only RISC-V architecture is supported!"

	# print("args", args)
	# input("<>")

	# initialize logging
	logging.basicConfig(level=getattr(logging, args.log.upper()))

	# resolve model paths
	top_level = pathlib.Path(args.top_level)
	abs_top_level = top_level.resolve()
	search_path = abs_top_level.parent.parent
	model_fname = abs_top_level

	if abs_top_level.suffix == ".core_desc":
		logger.warning(".core_desc file passed as input. This is deprecated behavior, please change your scripts!")
		search_path = abs_top_level.parent
		model_path = search_path.joinpath('gen_model')

		if not model_path.exists():
			raise FileNotFoundError('Models not generated!')
		model_fname = model_path / (abs_top_level.stem + '.m2isarmodel')

	if args.output_dir is None:
		output_base_path = search_path.joinpath('gen_patches')
		output_base_path.mkdir(exist_ok=True)
	else:
		output_base_path = pathlib.Path(args.output_dir)
		output_base_path.mkdir(exist_ok=True)

	logger.info("loading models")

	# load models
	with open(model_fname, 'rb') as f:
		models: "dict[str, arch.CoreDef]" = pickle.load(f)

	"""
	required_files = [
		"llvm/lib/Target/RISCV/RISCVSubtarget.h",
		"llvm/lib/Target/RISCV/RISCV.td",
		"llvm/lib/Target/RISCV/RISCVInstrFormatsCOREV.td",
		"llvm/lib/Target/RISCV/RISCVInstrFormats.td",
		"llvm/lib/Target/RISCV/RISCVInstrInfoCOREV.td",
		"llvm/lib/Target/RISCV/MCTargetDesc/RISCVBaseInfo.h",
		"llvm/lib/Target/RISCV/AsmParser/RISCVAsmParser.cpp",
		"llvm/lib/Support/RISCVISAInfo.cpp",
	]
	"""
	required_files = []
	required_patches = []

	generated_extensions = []
	# dummy_extension = Extension("Custom", description="My custom extension")
	# file_artifact = File("path/to/my/file", "BLA")
	# patch_artifact = NamedPatch("path/to_other_file", "my_patch", "foobar", append=True)
	# dummy_extension.add_artifact(file_artifact)
	# dummy_extension.add_artifact(patch_artifact)
	to_include = []


	# Global
	isa_new_archs = {}
	new_riscv_features = []

	processed_instruction_sets = []

	if args.gen_dlr_yaml:
		default_attrs = {
			"hasSideEffects": False,
			"mayLoad": False,
			"mayStore": False,
			"isTerminator": False,
		}
		default_operands = {
			"rd": {
				"output": True,
				"type": "GPR",
			},
			"rs1": {
				"output": False,
				"type": "GPR",
			},
			"rs2": {
				"output": False,
				"type": "GPR",
			},
		}
		default_types = {
			"i32": {
				"cType": "i32",
				"fType": "Li",
				# "ice": "?",  # TODO: ????
			},
			# "GPR": {?}  # TODO: ????
		}
		types_yaml_data = {}

		yaml_data = {
			"options": default_attrs,
			"types": default_types,
			"operands": default_operands,
			"extensions": [
			],
		}

	# preprocess model
	for core_name, core in models.items():
		logger.info("preprocessing model %s", core_name)
		if len(args.core) > 0 and core_name not in args.core:
			continue
		process_functions(core)
		process_instructions(core)

	existing_field_types = ["GPR", "simm12", "uimm5", "simm6"]  # TODO: add more
	used_field_types = []
	existing_complex_patterns = ["AddrRegImm"]
	used_complex_patterns = []

	# TODO: Allow Instruction Set Groups (for now only subgroups allowed)
	tablegen_per_set = {}
	for core_name, core_def in sorted(models.items()):

		if len(args.core) > 0 and core_name not in args.core:
			continue
		logger.info("processing core %s", core_name)
		# print("core_def", core_def)
		# print("core_def.instructions_by_ext", core_def.instructions_by_ext)
		intruction_set_names = core_def.instructions_by_ext.keys()
		# print("intruction_set_names", intruction_set_names)
		# input(">>>")

		for instruction_set_name in intruction_set_names:
			if len(args.set) > 0 and instruction_set_name not in args.set:
				continue
			assert instruction_set_name not in processed_instruction_sets, "A instruction set may not be processed twice"
			processed_instruction_sets.append(instruction_set_name)
			logger.info("processing set %s", instruction_set_name)
			if args.gen_legalizations:
				# TODO: get automatically
				ext_actions.update(get_legalizations_for_ext([instruction_set_name]))
			instructions = core_def.instructions_by_ext[instruction_set_name]

			isa_arch_name = configs.get(instruction_set_name, {}).get("arch", instruction_set_name.lower())
			if args.gen_dlr_yaml:
				extension_yaml_data = {
					"name": instruction_set_name,
					# "identifier": instruction_set_name.lower().replace("xcorev", "xcv").replace("branchimmediate", "bi"),
					"identifier": isa_arch_name,
					"fileName": instruction_set_name,
					"instructions": {},
				}
			extension = Extension(instruction_set_name, identifier=isa_arch_name, description=f"{instruction_set_name} Extension", experimental=args.experimental)

			# Filter
			instructions = {key: insn for key, insn in instructions.items() if len(args.insn) == 0 or insn.name in args.insn}

			if args.gen_insn_info:
				tablegen_per_insn = {}
			if args.gen_isel_patterns:
				tablegen_patterns = []

			if len(instructions) == 0:
				continue
			# TODO: instruction set groups!
			feature_name = configs.get(instruction_set_name, {}).get("feature", instruction_set_name.lower())
			# TODO: use yml
			# isa_arch_name = isa_arch_name.replace("xcorev", "xcv").replace("branchimmediate", "bi")
			isa_arch_version = tuple(configs.get(instruction_set_name, {}).get("version", "1.0").split("."))
			if isa_arch_name not in isa_new_archs:
				isa_new_archs[isa_arch_name] = isa_arch_version
			if instruction_set_name not in new_riscv_features:
				# new_riscv_features.append(instruction_set_name)  # TODO: convert to dict for handling subgroups
				new_riscv_features.append(feature_name)  # TODO: convert to dict for handling subgroups


			logger.debug("collecting predicates")
			predicates = []
			# predicates.append(f"HasExt{instruction_set_name}")
			predicates.append(f"HasExt{feature_name}")
			logger.debug("predicates %s", predicates)
			predicates_str = ""
			if len(predicates) > 0:
				temp = ", ".join(predicates)
				predicates_str = f"let Predicates = [{temp}]"
			# print("instructions", instructions)
			for _, instr in instructions.items():

				if args.gen_dlr_yaml:
					instruction_yaml_data = {
						# "operands": None,
						# "argstring": None,
						# "mayLoad": None,
						# "mayStore": None,
						# "hasSideEffects": None,
						# "isBranch": None,
						# "isTerminator": None,
					}

				logger.info("processing insn %s", instr.name)
				# print("instr", instr)

				logger.debug("compatibility check")
				assert instr.size == 32, "Only 32-bit instructions supported for now."

				logger.debug("collecting encoding")
				# code, mask = enc
				# logger.debug("instr_encoding %s", instr.encoding)
				# logger.debug("instr_fields %s", instr.fields)

				logger.debug("collecting assembly string")
				# logger.debug("instr_disass %s", instr.assembly)

				logger.info("collecting attributes")
				is_branch = False  # TODO: use
				is_conditional_branch = False  # TODO: use
				# print("instr.attributes", instr.attributes)
				# input("?")
				if len(instr.attributes) > 0:
					for attr in instr.attributes:
						if attr == arch.InstrAttribute.NO_CONT:
							is_branch = True
						elif attr == arch.InstrAttribute.COND:
							assert is_branch, "arch.InstrAttribute.COND implies arch.InstrAttribute.NO_CONT"
							is_conditional_branch = True
						else:
							raise NotImplementedError(f"Instruction Attribute {attr} currently not supported.")


				# print("instr_comp_types", instr.comp_types)
				# print("instr_comps", instr.comps)

				logger.debug("instr_operation %s", instr.operation)
				context = VisitorContext()
				instr.operation.statements = instr.operation.statements[1:]  # remove implicit PC intrement
				logger.debug("running type inference")
				patch_model(type_inference)
				instr.operation.generate(context)
				###
				logger.debug("running simplify_trivial_slice")
				patch_model(simplify_trivial_slice)
				instr.operation.generate(context)
				logger.debug("simplify_modulo")
				patch_model(simplify_modulo)
				instr.operation.generate(context)
				# input("A")
				# logger.debug("simplify_conds")
				# patch_model(simplify_conds)
				# instr.operation.generate(context)
				###

				logger.debug("applying transformations")
				# logger.debug("eliminating % RFS")
				# patch_model(eliminate_mod_rfs)
				# instr.operation.generate(context)
				logger.debug("eliminating rd != 0")
				patch_model(eliminate_cmp_zero)
				instr.operation.generate(context)
				logger.debug("eliminating X[r_]")
				patch_model(eliminate_gpr_index)
				instr.operation.generate(context)
				logger.debug("folding field signs")
				patch_model(fold_field_sign)
				instr.operation.generate(context)
				logger.debug("annotating field types")
				patch_model(annotate_field_type)
				instr.operation.generate(context)
				logger.debug("simplifying expressions")
				patch_model(utils.expr_simplifier)  # TODO
				instr.operation.generate(context)
				if len(instr.scalars) > 0:
					logger.debug("Eliminating scalars (if possible)")
					context = EliminateScalarsContext()
					patch_model(eliminate_scalar_assignments)
					instr.operation.generate(context)
					# context = VisitorContext()
					patch_model(find_unused_scalars)
					instr.operation.generate(context)
					patch_model(eliminate_unused_scalars)
					instr.operation.generate(context)
					# logger.debug("context.used %s", context.used)
					# logger.debug("instr.scalars %s", instr.scalars)
					scalars = {}
					for key, value in instr.scalars.items():
						if key in context.used:
							scalars[key] = value
					instr.scalars = scalars
				assert len(instr.scalars) == 0
				logger.debug("applying sext_inreg (where possible)")
				patch_model(fold_slice_conv)
				instr.operation.generate(context)
				# logger.debug("inserting powers (WIP)")
				# patch_model(insert_powers)
				# instr.operation.generate(context)
				logger.debug("detecting slices")
				patch_model(detect_slices)
				# instr.operation.generate(context)
				logger.debug("inserting min/max (WIP)")
				patch_model(find_minmax)
				instr.operation.generate(context)
				logger.debug("converting shr/sra (WIP)")
				patch_model(detect_arith_shift)
				instr.operation.generate(context)
				logger.debug("replace slice with shifts (WIP)")
				patch_model(replace_slices)
				instr.operation.generate(context)
				logger.debug("running type inference 2")
				patch_model(type_inference)
				instr.operation.generate(context)
				logger.debug("simplify typed bops (WIP)")
				patch_model(simplify_typed_binary_operations)
				# instr.operation.generate(context)
				if instr.throws:
					logger.debug("eliminate throws (WIP)")
					context = CollectThrowsContext()
					patch_model(eliminate_throws)
					instr.operation.generate(context)
					logger.debug("simplifying expressions")
					patch_model(utils.expr_simplifier)  # TODO
					instr.operation.generate(context)

				# input("continue?")
				logger.debug("collecting reads/writes")
				context = CollectRWContext()
				patch_model(collect_reads_writes)
				instr.operation.generate(context)
				reads = context.reads
				writes = context.writes
				reads = list(set(reads))
				writes = list(set(writes))
				logger.debug("adding writeback constraints")
				# TODO: further constraints
				constraints = []
				for i, w in enumerate(writes):
					if w in reads:
						t = None
						if ":" in w:
							t, w = w.split(":", 1)
						w_ = f"{w}_wb"
						assert w_ not in writes
						constraint = f"{w} = {w_}"
						constraints.append(constraint)
						if t is not None:
							w_ = f"{t}:{w_}"
						writes[i] = w_
				# writes = [f"{x}_wb" if x in reads else x for x in writes]
				logger.debug("reads %s", reads)
				logger.debug("writes %s", writes)
				logger.debug("constraints %s", constraints)
				asm_str = instr.assembly
				if asm_str.startswith("\""):
					asm_str = asm_str[1:-1]
				asm_str = re.sub(r'{([a-zA-Z0-9]+)}', r'$\g<1>', re.sub(r'{([a-zA-Z0-9]+):[#0-9a-zA-Z\.]+}', r'{\g<1>}', re.sub(r'name\(([a-zA-Z0-9]+)\)', r'\g<1>', asm_str)))
				if args.gen_dlr_yaml:
					instruction_yaml_data["argstring"] = asm_str
				tokens = get_asm_tokens(asm_str)
				# print("tokens", tokens)
				asm_order = re.compile(r"(\$[a-zA-Z0-9]+)").findall(asm_str)
				# add {} around tokens if required by asm parser
				for op in asm_order:
					if f"{op}(" in asm_str or f"{op})" in asm_str or f"{op}!" in asm_str or f"!{op}" in asm_str:
						asm_str = asm_str.replace(op, "${" + op[1:] + "}")
				if len(reads) == 0 and len(writes) == 0:
					logger.warning("Unable to collect reads/writes from behavior. Parsing assembly instead...")
					assert len(asm_order) > 0, "Assembly arguments empty during fallback"
					for op in asm_order:
						assert op[0] == "$"
						assert len(op) > 1
						# op_ = op[1:].lower()
						op_ = op[1:]
						if op_[:2].lower() == "rs":
							reads.append(f"GPR:${op_}")
						elif op_[:2].lower() == "rd":
							writes.append(f"GPR:${op_}")
						elif "imm" in op_.lower():
							ty = op_.lower()
							pre, post = ty.split("imm")
							if len(pre) == 0:
								pre = "s"  # Assume signed by default
							elif len(pre) == 2:
								if pre[0].lower() in ["i", "l"]:
									pre = pre[1]
							if len(post) == 0:
								raise NotImplementedError("Detect sign based on encoding")
							ty = pre + "imm" + post
							reads.append(f"{ty}:${op_}")
						# elif op_.lower() in ["is2", "is3"]:  # TODO: get rid of hardcoded workaround here
						# 	ty = "uimm5"
						# 	reads.append(f"{ty}:${op_}")
						else:
							logger.error("Unsupported op name in asm_string during fallback: %s", op_)
							assert False
				reads_ = [(x.split(":", 1)[1] if ":" in x else x) for x in reads]
				writes_ = [(x.split(":", 1)[1] if ":" in x else x).replace("_wb", "") for x in writes]
				# check that each assembly operand is either a read or written
				for op in asm_order:
					if op in reads_:
						continue
					if op in writes_:
						continue
					if op == "$rd" and len(writes_) == 0:
						# automatic fallback for rd
						writes.append(f"GPR:{op}")
						writes_.append(op)
						continue
					# if op in ["$rs1", "$rs2"]:
					# 	# automatic fallback for rs1,rs2
					# 	reads.append(f"GPR:{op}")
					# 	reads_append(op)
					raise RuntimeError(f"Operand {op} is not used?")

				if args.gen_dlr_yaml:
					operands_yaml_data = {}
					yaml_ins = [tuple(map(lambda x: x.replace("$", ""), reads[reads_.index(x)].split(":"))) for x in asm_order if x in reads_]
					yaml_outs = [tuple(map(lambda x: x.replace("$", ""), writes[writes_.index(x)].split(":"))) for x in asm_order if x in writes_]
					for ty, name in yaml_ins:
						operand_yaml_data = {
							"output": False,
							"type": ty,
						}
						operands_yaml_data[name] = operand_yaml_data
					for ty, name in yaml_outs:
						operand_yaml_data = {
							"output": True,
							"type": ty,
						}
						operands_yaml_data[name] = operand_yaml_data
					instruction_yaml_data["operands"] = {key: value for key, value in operands_yaml_data.items() if key not in default_operands or value != default_operands[key]}
				ins_str = ", ".join([reads[reads_.index(x)] for x in asm_order if x in reads_])
				outs_str = ", ".join([writes[writes_.index(x)] for x in asm_order if x in writes_])

				logger.debug("collecting details")
				details = []
				context = AnalysisContext()
				patch_model(analyze_behavior)
				instr.operation.generate(context)
				has_side_effects = is_branch or context.has_side_effects
				may_load = context.may_load
				may_store = context.may_store
				is_terminator = False  # TODO
				details_str = ""
				# logger.debug("details_str %s", details_str)

				# print("has_side_effects", has_side_effects)
				# print("may_load", may_load)
				# print("may_store", may_store)
				attrs = {
					"hasSideEffects": has_side_effects,
					"mayLoad": may_load,
					"mayStore": may_store,
					"isTerminator": is_terminator,
				}
				if args.gen_dlr_yaml:
					instruction_yaml_data.update({key: value for key, value in attrs.items() if key not in default_attrs or value != default_attrs[key]})

				# print("dir", dir(instr))
				# """
				# print("============================")
				# text = ""
				# for pre, fill, node in RenderTree(context.tree):
				# 	if hasattr(node, "ref_"):
				# 		print("%s%s [%s]" % (pre, node.name, node.ref_))
				# 		text += " " + node.ref_
				# 	elif hasattr(node, "ref"):
				# 		print("%s%s [%s]" % (pre, node.name, node.ref))
				# 	else:
				# 		print("%s%s" % (pre, node.name))
				# print("============================")
				# """
				# logger.debug("text %s", text)
				# print("instr.name", instr.name)
				# input(",,,")
				# real_name = instr.name.lower().replace("_", ".")
				real_name = instr.mnemonic
				if args.gen_insn_info:
					for x in reads + writes:
						ty = x.split(":")[0]
						if ty not in used_field_types:
							used_field_types.append(ty)
					tablegen_str = tablegen_writer.write_riscv_instruction_info(instr.name, real_name, asm_str, ins_str, outs_str, instr.encoding, instr.fields, details_str, attrs=attrs, constraints=constraints, formats=args.gen_insn_formats)
					"""
					print("============================")
					logger.debug("tablegen_str %s", tablegen_str)
					print("============================")
					"""
					tablegen_per_insn[instr.name] = tablegen_str
				if args.gen_isel_patterns:
					logger.debug("collecting patterns")
					context = DAGBuilderContext()
					patch_model(dag_builder)
					instr.operation.generate(context)
					patterns = context.patterns
					complex_patterns = context.complex_patterns
					if len(complex_patterns) > 0:
						for pat_name in complex_patterns:
							# check for duplicates
							found = False
							for used in used_complex_patterns:
								if used.name == pat_name:
									found = True
									break
							if found:
								continue
							cpat = ComplexPattern(pat_name, f"Select{pat_name}")  # TODO: XCoreVMem only!
							used_complex_patterns.append(cpat)
					# print("patterns", patterns)
					# input("p")
					if len(patterns) > 1:
						# print("patterns", patterns)
						logger.warning("Can not generate tablegen pattern for multi-output insn %s", instr.name)
					elif len(patterns) == 1:
						logger.info("Found pattern for insn %s", instr.name)
						target, dag = list(patterns.items())[0]
						# TODO: assert target in outputs
						# if len(ins_str) == 0 and len(outs_str2):
						# 	args_str = ""  # Not allowed?
						# elif len(ins_str) == 0:
						# 	args_str = outs_str2
						# elif len(outs_str2) == 0:
						# 	args_str = ins_str
						# else:
						# 	args_str = ins_str + ", " + outs_str2
						args_str = ins_str
						pat = f"def : Pat<{dag}, ({instr.name} {args_str})>;"  # TODO: move to helper
						tablegen_patterns.append(pat)
					else:
						# raise RuntimeError("No pattern found")
						logger.error("No pattern found!")
					# print("pat0", pat)
					# input(".")
				if args.gen_dlr_yaml:
					extension_yaml_data["instructions"][instr.name] = instruction_yaml_data
			if args.gen_dlr_yaml:
				yaml_data["extensions"].append(extension_yaml_data)

			# print("set_tablegen_patterns", set_tablegen_patterns)
			# input("11")

			# set_tablegen = set_tablegen_info + "\n" + set_tablegen_patterns
			# set_tablegen = (set_tablegen_info, set_tablegen_patterns)

			# tablegen_per_set[instruction_set_name] = set_tablegen
			if args.gen_insn_info:
				set_tablegen_info = tablegen_writer.combine_instruction_info(tablegen_per_insn, predicates_str)
				insn_info_artifact = File(f"llvm/lib/Target/RISCV/RISCVInstrInfo{extension.name}.td", set_tablegen_info)
				extension.add_artifact(insn_info_artifact)
				to_include.append(f"RISCVInstrInfo{extension.name}.td")
			if args.gen_isel_patterns:
				set_tablegen_patterns = tablegen_writer.combine_patterns(tablegen_patterns, predicates_str)
				isel_patterns_artifact = File(f"llvm/lib/Target/RISCV/RISCVInstrInfo{extension.name}Patterns.td", set_tablegen_patterns)
				extension.add_artifact(isel_patterns_artifact)
				to_include.append(f"RISCVInstrInfo{extension.name}Patterns.td")

			generated_extensions.append(extension)

	out = {}

	# print("tablegen_per_set", tablegen_per_set)
	new_insn_formats = []
	new_opcodes = []

	# global_extension = Extension("Global", description="Dummy")  # TODO: improve
	global_artifacts = []
	new_field_types = [x for x in used_field_types if x not in existing_field_types]

	# print("isa_new_archs", isa_new_archs)
	if args.gen_isa_info:
		if len(generated_extensions) > 0 :
			isa_info_str, experimental_isa_info_str = cpp_writer.write_isa_info_patches(generated_extensions)
			if len(isa_info_str) > 0:
				isa_info_artifact = NamedPatch("llvm/lib/Support/RISCVISAInfo.cpp", isa_info_str, "isa_info", append=True)
				global_artifacts.append(isa_info_artifact)
			if len(experimental_isa_info_str) > 0:
				experimental_isa_info_artifact = NamedPatch("llvm/lib/Support/RISCVISAInfo.cpp", experimental_isa_info_str, "experimental_isa_info", append=True)
				global_artifacts.append(experimental_isa_info_artifact)

	if len(new_field_types) > 0 and (args.gen_asm_parser or args.gen_insn_info):
		immops = []
		for ty in new_field_types:
			if "imm" in ty:
				signed, bits = ty.split("imm")
				bits = int(bits)
				# if len(signed) == 2 and signed[0].upper() in ["I", "L"]:
				# 	signed = signed[1]
				assert len(signed) == 1
				assert signed.upper() in ["U", "S"]
				signed = signed.upper() == "S"
				immop = ImmOperand(ty, bits, signed)
				immops.append(immop)
			else:
				raise NotImplementedError
		if args.gen_asm_parser:
			asm_parser_field_types_str = cpp_writer.write_asm_parser_patch(immops)
			asm_parser_field_types_artifact = NamedPatch("llvm/lib/Target/RISCV/AsmParser/RISCVAsmParser.cpp", asm_parser_field_types_str, "riscv_operands", append=True)
			global_artifacts.append(asm_parser_field_types_artifact)
		if args.gen_insn_info:  # TODO: add dependency on gen_insn_formats as well
			instr_info_field_types_str = ""
			base_info_riscv_operands_str = ""
			for immop in immops:
				instr_info_field_types_str += immop.gen_tablegen()
				base_info_riscv_operands_str += immop.gen_cpp()

			instr_info_field_types_artifact = NamedPatch("llvm/lib/Target/RISCV/RISCVInstrInfo.td", instr_info_field_types_str, "field_types", append=True)
			global_artifacts.append(instr_info_field_types_artifact)
			base_info_riscv_operands_artifact = NamedPatch("llvm/lib/Target/RISCV/MCTargetDesc/RISCVBaseInfo.h", base_info_riscv_operands_str, "riscv_operands", append=True)
			global_artifacts.append(base_info_riscv_operands_artifact)

	if args.gen_insn_formats:
		if len(new_insn_formats) > 0:
			base_info_insn_formats_str = cpp_writer.write_riscv_base_info_insn_formats_patch(new_insn_formats)
			base_info_insn_formats_artifact = NamedPatch("llvm/lib/Target/RISCV/MCTargetDesc/RISCVBaseInfo.h", base_info_insn_formats_str, "insn_formats", append=True)
			global_artifacts.append(base_info_insn_formats_artifact)

	if args.gen_insn_formats:
		global_insn_formats = []
		for insn_format in new_insn_formats:
			if insn_format.is_global():
				global_insn_formats.append(insn_format)
		if len(global_insn_formats) > 0:
			insn_formats_str = tablegen_writer.write_riscv_formats_patch(global_insn_formats)
			insn_formats_artifact = NamedPatch("llvm/lib/Target/RISCV/RISCVIntrFormats.td", insn_formats_str, "insn_formats_global")
			global_artifacts.append(insn_formats_artifact)
	if len(new_opcodes) > 0:
		# only global opcodes supported
		opcodes_str = tablegen_writer.write_riscv_formats_opcode_patch(new_opcodes)
		opcodes_artifact = NamedPatch("llvm/lib/Target/RISCV/RISCVIntrFormats.td", insn_formats_str, "opcodes")
		global_artifacts.append(opcodes_artifact)

	if args.gen_features:
		if len(new_riscv_features) > 0:
			features_str = tablegen_writer.write_riscv_features_patch(new_riscv_features)
			features_artifact = NamedPatch("llvm/lib/Target/RISCV/RISCVFeatures.td", features_str, "features", append=True)
			global_artifacts.append(features_artifact)
			# Not required for latest llvm?
			# out["llvm/lib/Target/RISCV/RISCVSubTarget.h.patch"] = cpp_writer.write_subtarget_patch(new_riscv_features)


	# print("to_include", to_include)
	# input("abc")
	if len(to_include) > 0:
		includes_str = tablegen_writer.write_riscv_instruction_info_patch(to_include)
		insn_info_artifact = NamedPatch("llvm/lib/Target/RISCV/RISCVInstrInfo.td", includes_str, "insn_info_includes", append=True)
		global_artifacts.append(insn_info_artifact)

	if args.gen_legalizations:
		legalizations_str = gen_legalization_tablegen(ext_actions)
		legalization_artifact = NamedPatch("llvm/lib/Target/RISCV/RISCVISelLowering.cpp", legalizations_str, "legal_ops", append=True)
		global_artifacts.append(legalization_artifact)

	new_complex_patterns = [x for x in used_complex_patterns if x.name not in existing_complex_patterns]

	if args.gen_isel_patterns:
		if len(new_complex_patterns) > 0:
			# TODO: keep track which extension requires pattern instead of defining it globaly
			complex_patterns_strs = []
			complex_patterns_str = "\n".join(list(map(lambda x: x.gen_tablegen(), new_complex_patterns)))
			complex_patterns_artifact = NamedPatch("llvm/lib/Target/RISCV/RISCVInstrInfo.td", complex_patterns_str, "complex_patterns", append=True)
			global_artifacts.append(complex_patterns_artifact)


	for ext in generated_extensions:
		for artifact in ext.artifacts:
			out[artifact.out_path] = artifact.content
	for artifact in global_artifacts:
		out[artifact.out_path] = artifact.content


	if args.gen_index:
		extensions_yaml_data = []
		for ext in generated_extensions:
			extension_yaml_data = ext.to_dict()
			extensions_yaml_data.append(extension_yaml_data)
		index_yaml_data = {
			"artifacts": list(map(lambda a: a.to_dict(), global_artifacts)),
			"extensions": extensions_yaml_data,
		}
		out["index.yaml"] = yaml.safe_dump(index_yaml_data)


	if args.gen_dlr_yaml:
		out["dlr.yaml"] = yaml.safe_dump(yaml_data)
		# print("yaml_data", yaml.safe_dump(yaml_data, default_flow_style=False))
	for path, content in out.items():
		combined_path = output_base_path / path
		combined_path.parent.mkdir(exist_ok=True, parents=True)
		logger.info(f"Exporting file '{path}' to '{output_base_path}'")
		with open(combined_path, "w") as handle:
			handle.write(content)

if __name__ == "__main__":
	main()
