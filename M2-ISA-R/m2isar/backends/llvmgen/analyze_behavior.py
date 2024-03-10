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
from .utils import VisitorContext
from anytree import Node

# pylint: disable=unused-argument

LOOKUP = {
	"+": "add",
	"*": "mul",
	"-": "sub",
	"%": "srem/urem",
	"&": "and",
	"|": "or",
	"^": "xor",
	"<<": "shl",
	">>": "srl",
	">>>": "sra",
	"**": "pow",  # TODO
	"<": "setlt",
	">": "setgt",
	"<=": "setle",
	">=": "setge",
	"==": "seteq",
	"!=": "setne",
}

def operation(self: behav.Operation, context: "VisitorContext"):
	# print("operation")
	context.push(Node("Operation", parent=context.parent, ref=self))

	for i, stmt in enumerate(self.statements):
		ret = stmt.generate(context)

	context.pop()
	return self


def binary_operation(self: behav.BinaryOperation, context: "VisitorContext"):
	# print("binary_operation")
	context.push(Node("Operation", parent=context.parent, ref=self))
	op_name = LOOKUP.get(self.op.value, "UNKNOWN")

	context.push(Node("Binary Operation", parent=context.parent, ref=self, ref_=op_name))

	context.push(Node("Left", parent=context.parent))
	self.left.generate(context)
	context.pop()

	context.push(Node("Right", parent=context.parent))
	self.right.generate(context)
	context.pop()

	context.pop()
	return self

def slice_operation(self: behav.SliceOperation, context: "VisitorContext"):
	# print("slice_operation")
	ref_ = None
	for ann in self.annotations:
		ref_ = ann
		break
	context.push(Node("Slice Operation", parent=context.parent, ref=self, ref_=ref_ if ref_ else ""))

	context.push(Node("Expr", parent=context.parent))
	self.expr.generate(context)
	context.pop()

	context.push(Node("Left", parent=context.parent))
	self.left.generate(context)
	context.pop()

	context.push(Node("Right", parent=context.parent))
	self.right.generate(context)
	context.pop()

	context.pop()

	return self

def concat_operation(self: behav.ConcatOperation, context: "VisitorContext"):
	# print("concat_operation")
	context.print("concat_operation")
	context.push(Node("Concat Operation", parent=context.parent, ref=self))

	context.push(Node("Left", parent=context.parent))
	self.left.generate(context)
	context.pop()

	context.push(Node("Right", parent=context.parent))
	self.right.generate(context)
	context.pop()

	context.pop()
	return self

def number_literal(self: behav.IntLiteral, context: "VisitorContext"):
	context.push(Node("Number Literal", parent=context.parent, ref=self, ref_=f"{self.value}"))
	context.pop()
	return self

def int_literal(self: behav.IntLiteral, context: "VisitorContext"):
	context.push(Node("Int Literal", parent=context.parent, ref=self))
	context.pop()
	return self

def scalar_definition(self: behav.ScalarDefinition, context: "VisitorContext"):
	context.push(Node("Scalar Definition", parent=context.parent, ref=self))
	context.pop()
	return self

def assignment(self: behav.Assignment, context: "VisitorContext"):
	# print("assignment")
	# if isinstance(self.target, behav.ScalarDefinition):
	#		context.assignments[self.target] = self.expr
	#		return

	context.push(Node("Assignment", parent=context.parent, ref=self))

	context.push(Node("Target", parent=context.parent))

  # There are no nested assignments, therefore this should work in theory
	# for register reads/writes and memory loads/stores

	context.is_write = True
	target = self.target.generate(context)
	# print("target_", target_)

	context.is_write = False
	context.pop()

	context.push(Node("Expr", parent=context.parent))
	context.is_read = True
	expr = self.expr.generate(context)
	# print("expr_", expr_)
	context.is_read = False
	context.pop()

	context.pop()
	return self

def conditional(self: behav.Conditional, context: "VisitorContext"):
	context.push(Node("Conditional", parent=context.parent, ref=self))

	context.push(Node("Conds", parent=context.parent))
	ret = self
	for cond in self.conds:
		context.is_read = True
		cond.generate(context)
		context.is_read = False
		# if isinstance(ret, behav.IntLiteral):
		#	assert len(self.conds) == 1
		#	if not ret.signed and ret.value > 0:  # TODO
		#		ret = self.stmts[0][0].generate(context)  # TODO
		#	else:
		#		ret = self.stmts[1][0].generate(context)  # TODO
		#	break
		# print("ret3", ret)
		# input("?????")
	context.pop()

	if ret == self:
		context.push(Node("Stmts", parent=context.parent))
		for op in self.stmts:
			# print("op", op)
			context.push(Node("Stmt", parent=context.parent))
			for stmt in flatten(op):
				# print("stmt", stmt)
				stmt.generate(context)
				is_branch = False
				is_conditional = True
				is_relative = False
        # TODO: see below
				# raise NotImplementedError("DETECT BRANCH")
			context.pop()
		context.pop()

	context.pop()
	return ret

def loop(self: behav.Loop, context: "VisitorContext"):
	context.push(Node("Loop", parent=context.parent, ref=self))
	context.push(Node("Cond", parent=context.parent))

	context.is_read = True
	self.cond.generate(context)
	context.is_read = False
	context.pop()
	context.push(Node("Statements", parent=context.parent))
	self.stmts = [x.generate(context) for x in self.stmts]
	context.pop()
	context.pop()
	return self

def ternary(self: behav.Ternary, context: "VisitorContext"):
	context.push(Node("Ternary", parent=context.parent, ref=self))

	context.push(Node("Cond", parent=context.parent))
	self.cond.generate(context)
	context.pop()

	context.push(Node("Then Expr", parent=context.parent))
	self.then_expr.generate(context)
	context.pop()

	context.push(Node("Else Expr", parent=context.parent))
	self.else_expr.generate(context)
	context.pop()

	context.pop()
	return self

def return_(self: behav.Return, context: "VisitorContext"):
	context.push(Node("Return", parent=context.parent, ref=self))

	if self.expr is not None:
		context.push(Node("Expr", parent=context.parent))
		self.expr.generate(context)
		context.pop()

	context.pop()
	return self

def unary_operation(self: behav.UnaryOperation, context: "VisitorContext"):
	# print("unary_operation")
	context.push(Node("Unary Expression", parent=context.parent, ref=self))

	context.push(Node("Right", parent=context.parent))
	self.right.generate(context)
	context.pop()

	context.pop()
	return self

def named_reference(self: behav.NamedReference, context: "VisitorContext"):
	# print("named_reference")
	if ":" in self.reference.name:
		ty, name = self.reference.name.split(":", 1)
		if "$" in name:
			name = name[1:]
	else:
		ty = "pc" if arch.MemoryAttribute.IS_PC in self.reference.attributes else "?"
		name = self.reference.name
	ret = self
	if ret == self:
		context.push(Node("Named Reference", parent=context.parent, ref=self, ref_=self.reference.name))
		context.pop()
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
		new_op = None
		if context.is_write:
			# print("STORE")
			new_op = "dummy_store" # missing value!
			context.may_store = True
			# input("~")
		elif context.is_read:
			# print("LOAD")
			new_op = "load"
			context.may_load = True
		# print("self.index", self.index)
		self.index.generate(context)
	elif arch.MemoryAttribute.IS_MAIN_REG in self.reference.attributes:
		pass  # handled in IndexedReference (should eliminate Index?)  # TODO
	else:  # CSR, custom memory,...
		raise NotImplementedError(f"Unhandled reference: {name} (Side effects?)")

	#	return arch.MemoryAttribute.IS_MAIN_REG in ref.attributes  # or ref.name == "X"
	# is_gpr = check_gpr(self)
	# if not is_gpr:
	#	context.has_side_effects = True

	# context.print("> is_gpr", is_gpr)
	context.push(Node("Indexed Reference", parent=context.parent, ref=self))


	context.push(Node("Reference", parent=context.parent))
	context.push(Node(name, parent=context.parent))
	context.pop()
	context.pop()
	context.push(Node("Index", parent=context.parent))
	self.index.generate(context)
	context.pop()

	context.pop()
	return self

def type_conv(self: behav.TypeConv, context: "VisitorContext"):

	context.push(Node("Type Conv", parent=context.parent, ref=self, ref_="?"))


	context.push(Node("Expr", parent=context.parent))
	self.expr.generate(context)
	context.pop()

	context.pop()
	return self

def callable_(self: behav.Callable, context: "VisitorContext"):
	context.push(Node("Callable", parent=context.parent, ref=self))

	for arg, arg_descr in zip(self.args, self.ref_or_name.args):
		context.push(Node("Arg", parent=context.parent))
		arg.generate(context)
		context.pop()

	context.pop()
	return self

def group(self: behav.Group, context: "VisitorContext"):
	context.push(Node("Group", parent=context.parent, ref=self, ref_="()"))
	context.push(Node("Expr", parent=context.parent))
	self.expr.generate(context)
	context.pop()

	context.pop()
	return self

def dereference(self: behav.DeReference, context: "VisitorContext"):
	context.push(Node("DeReference", parent=context.parent, ref=self))

	context.pop()
	return self

def union_definition(self: behav.UnionDefinition, context: "VisitorContext"):
	context.push(Node("Union Definition", parent=context.parent, ref=self))

	context.pop()
	return self
