# SPDX-License-Identifier: Apache-2.0
#
# This file is part of the M2-ISA-R project: https://github.com/tum-ei-eda/M2-ISA-R
#
# Copyright (C) 2022
# Chair of Electrical Design Automation
# Technical University of Munich

"""TODO"""

# import tkinter as tk

from ... import flatten
from ...metamodel import arch, behav
from .utils import VisitorContext, DAGAssignment, DAGOperand, DAGType, DAGUnary, DAGBinary, DAGTernary, DAGQuaternary, DAGImm
from anytree import Node

# pylint: disable=unused-argument

def operation(self: behav.Operation, context: "VisitorContext"):
	# print("operation")

	for i, stmt in enumerate(self.statements):
		stmt.generate(context)

	return self


def binary_operation(self: behav.BinaryOperation, context: "VisitorContext"):

	self.left.generate(context)

	self.right.generate(context)

	return self

def slice_operation(self: behav.SliceOperation, context: "VisitorContext"):
	# print("slice_operation")
	ref_ = None
	for ann in self.annotations:
		ref_ = ann
		break

	self.expr.generate(context)

	self.left.generate(context)

	self.right.generate(context)

	return self

def concat_operation(self: behav.ConcatOperation, context: "VisitorContext"):
	# print("concat_operation")
	context.print("concat_operation")

	self.left.generate(context)

	self.right.generate(context)

	return self

def number_literal(self: behav.IntLiteral, context: "VisitorContext"):
	return self

def int_literal(self: behav.IntLiteral, context: "VisitorContext"):
	return self

def scalar_definition(self: behav.ScalarDefinition, context: "VisitorContext"):
	return self

def assignment(self: behav.Assignment, context: "VisitorContext"):

  	# There are no nested assignments, therefore this should work in theory
	# for register reads/writes and memory loads/stores
	ret_ = None

	context.is_write = True
	self.target.generate(context)

	context.is_write = False

	context.is_read = True
	self.expr.generate(context)
	context.is_read = False

	return self

def conditional(self: behav.Conditional, context: "VisitorContext"):

	ret = self
	for cond in self.conds:
		context.is_read = True
		cond.generate(context)
		context.is_read = False

	if ret == self:
		for op in self.stmts:
			# print("op", op)
			for stmt in flatten(op):
				# print("stmt", stmt)
				stmt.generate(context)
				is_branch = False
				is_conditional = True
				is_relative = False

	return ret

def loop(self: behav.Loop, context: "VisitorContext"):

	context.is_read = True
	self.cond.generate(context)
	context.is_read = False
	self.stmts = [x.generate(context) for x in self.stmts]
	return self

def ternary(self: behav.Ternary, context: "VisitorContext"):

	self.cond.generate(context)

	self.then_expr.generate(context)

	self.else_expr.generate(context)

	return self

def return_(self: behav.Return, context: "VisitorContext"):

	if self.expr is not None:
		self.expr.generate(context)

	return self

def unary_operation(self: behav.UnaryOperation, context: "VisitorContext"):
	# print("unary_operation")

	self.right.generate(context)

	return self

def named_reference(self: behav.NamedReference, context: "VisitorContext"):
	# print("named_reference")
	if context.is_read:
		context.reads.append(self.reference.name)
	elif context.is_write:
		context.writes.append(self.reference.name)
	ret = self
	if ret == self:
		pass
	else:
		ret.generate(context)

	return ret

def indexed_reference(self: behav.IndexedReference, context: "VisitorContext"):
	# print("indexed_reference")
	# print("dir", dir(self.reference), self.reference.attributes)
	# def check_gpr(node: behav.IndexedReference):
	#		ref = node.reference
	assert isinstance(self.reference, arch.Memory)
	name = self.reference.name
	if arch.MemoryAttribute.IS_MAIN_MEM in self.reference.attributes:
		if context.is_write:
			# print("STORE")
			context.may_store = True
			context.is_read = True
			self.index.generate(context)
			context.is_read = False
			# input("~")
		elif context.is_read:
			# print("LOAD")
			context.may_load = True
			self.index.generate(context)
		# print("self.index", self.index)

			# print("ret_", ret_)
			# input("123")
	elif arch.MemoryAttribute.IS_MAIN_REG in self.reference.attributes:
		pass  # handled in IndexedReference (should eliminate Index?)  # TODO
	else:  # CSR, custom memory,...
		raise NotImplementedError(f"Unhandled reference: {name} (Side effects?)")

	# self.index.generate(context)

	return self

def type_conv(self: behav.TypeConv, context: "VisitorContext"):

	self.expr.generate(context)

	return self

def callable_(self: behav.Callable, context: "VisitorContext"):

	for arg, arg_descr in zip(self.args, self.ref_or_name.args):
		arg.generate(context)

	return self

def group(self: behav.Group, context: "VisitorContext"):
	self.expr.generate(context)

	return self

def dereference(self: behav.DeReference, context: "VisitorContext"):

	return self

def union_definition(self: behav.UnionDefinition, context: "VisitorContext"):

	return self
