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

	patterns = {}  # destination : DAGOperation
	prev = None
	for i, stmt in enumerate(self.statements):
		ret, ret2 = stmt.generate(context)
		# print("ret", ret)
		# print("ret2", ret2)  # DAGAssignment
		dest = f"pat{i}"
		expr = ret2
		if isinstance(ret2, DAGAssignment):
			assert isinstance(ret.target, behav.NamedReference)
			dest = ret.target.reference.name
			expr = ret2.expr
		patterns[dest] = expr

	# check for indexed stores
	for dest, expr in patterns.items():
		if isinstance(expr, DAGBinary):
			# print("expr", expr)
			if expr.name in ["truncstorei8", "truncstorei16", "store"]:  # TODO: more generic
				val, ptr = expr.operands
				# print("ptr", ptr)
				if isinstance(ptr, DAGOperand):
					ptr_reg = str(ptr)
					# print("ptr_reg", ptr_reg)
					if ptr_reg in patterns:
						pre = list(patterns.keys()).index(ptr_reg) < list(patterns.keys()).index(dest)
						# print("pre", pre)
						comp = patterns[ptr_reg]
						# print("comp", comp)
						if isinstance(comp, DAGBinary):
							op_name = comp.name
							# print("op_name", op_name)
							if op_name in ["add", "sub"]:
								# print("A")
								lhs, rhs = comp.operands
								if str(lhs) == ptr_reg:  # TODO: also allow swap for +
									# print("IF")
									new_name = expr.name
									if "truncstore" in new_name:
										new_name = new_name.replace("store", "st")
									prefix = "pre" if pre else "post"
									new_name = f"{prefix}_{new_name}"
									new_op = DAGTernary(new_name, val, ptr, rhs)
									patterns[ptr_reg] = new_op
									# print("ASSIGNED")
									del patterns[dest]
									break


	context.patterns = patterns
	# input("abc")

	context.pop()
	# return self, None
	return self


def binary_operation(self: behav.BinaryOperation, context: "VisitorContext"):
	# print("binary_operation")
	context.push(Node("Operation", parent=context.parent, ref=self))
	op_name = LOOKUP.get(self.op.value, "UNKNOWN")

	# TODO: remove this hacks
	# if self.op.value == "%":
	#		if isinstance(self.right, behav.NumberLiteral):
	#			if self.right.value == 32:
	#			self.left.generate(context)
	#			return self.left
	# elif self.op.value == "!=":
	#	if isinstance(self.right, behav.NumberLiteral):
	#		if self.right.value == 0:
	#			# self.left.generate(context)
	#			lit = behav.IntLiteral(1)
	#			lit.generate(context)  # Will this work?
	#			return lit

	# input("qqq")
	context.push(Node("Binary Operation", parent=context.parent, ref=self, ref_=op_name))

	context.push(Node("Left", parent=context.parent))
	_, left_ = self.left.generate(context)
	context.pop()

	context.push(Node("Right", parent=context.parent))
	_, right_ = self.right.generate(context)
	context.pop()

	context.pop()
	if op_name == "srl":
		if isinstance(right_, DAGImm):
			# needs explicit type
			ty_name = "i32"  # TODO: do not hardcode
			right_ = DAGUnary(ty_name, right_)

	ret_ = DAGBinary(op_name, left_, right_)
	return self, ret_

def slice_operation(self: behav.SliceOperation, context: "VisitorContext"):
	# print("slice_operation")
	ref_ = None
	for ann in self.annotations:
		ref_ = ann
		break
	context.push(Node("Slice Operation", parent=context.parent, ref=self, ref_=ref_ if ref_ else ""))

	context.push(Node("Expr", parent=context.parent))
	_, expr_ = self.expr.generate(context)
	context.pop()

	context.push(Node("Left", parent=context.parent))
	_, left_ = self.left.generate(context)
	context.pop()

	context.push(Node("Right", parent=context.parent))
	_, right_ = self.right.generate(context)
	context.pop()

	context.pop()

	ret_ = DAGTernary("slice", expr_, left_, right_)  # TODO
	return self, ret_

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
	return self, None

def number_literal(self: behav.IntLiteral, context: "VisitorContext"):
	context.push(Node("Number Literal", parent=context.parent, ref=self, ref_=f"{self.value}"))
	context.pop()
	ret_ = DAGImm(self.value)
	return self, ret_

def int_literal(self: behav.IntLiteral, context: "VisitorContext"):
	context.push(Node("Int Literal", parent=context.parent, ref=self))
	context.pop()
	return self, None

def scalar_definition(self: behav.ScalarDefinition, context: "VisitorContext"):
	context.push(Node("Scalar Definition", parent=context.parent, ref=self))
	context.pop()
	return self, None

def assignment(self: behav.Assignment, context: "VisitorContext"):
	# print("assignment")
	# if isinstance(self.target, behav.ScalarDefinition):
	#		context.assignments[self.target] = self.expr
	#		return

	context.push(Node("Assignment", parent=context.parent, ref=self))

	context.push(Node("Target", parent=context.parent))

  # There are no nested assignments, therefore this should work in theory
	# for register reads/writes and memory loads/stores
	ret_ = None

	context.is_write = True
	target, target_ = self.target.generate(context)
	# print("target_", target_)

	context.is_write = False
	context.pop()

	context.push(Node("Expr", parent=context.parent))
	context.is_read = True
	expr, expr_ = self.expr.generate(context)
	# print("expr_", expr_)
	context.is_read = False
	context.pop()

	context.pop()
	if isinstance(target_, DAGUnary):
		if "dummy_store" in target_.name:
			new_name = target_.name.replace("dummy_", "")
			ty = expr.inferred_type
			assert ty is not None
			assert isinstance(ty, arch.IntegerType)
			signed = ty.signed
			width = ty.width
			out_type = f"i{width}"
			if isinstance(expr_, DAGUnary):
				if expr_.name == out_type:
					expr_ = expr_.operands[0]

			if width == 32:  # TODO: do not hardcode XLEN
				pass
			elif width in [8, 16]:
				new_name += out_type
				new_name = "trunc" + new_name
			else:
				raise NotImplementedError

			ret_ = DAGBinary(new_name, expr_, target_.operands[0])
	if not ret_:
		ret_ = DAGAssignment(target_, expr_)
	# print("ret_", ret_)
	# input("@#")
	return self, ret_

def conditional(self: behav.Conditional, context: "VisitorContext"):
	context.push(Node("Conditional", parent=context.parent, ref=self))

	context.push(Node("Conds", parent=context.parent))
	ret = self
	conds_ = []
	for cond in self.conds:
		context.is_read = True
		_, cond_ = cond.generate(context)
		context.is_read = False
		conds_.append(cond_)
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
				_, stmt_ = stmt.generate(context)
				is_branch = False
				is_conditional = True
				is_relative = False
				if isinstance(stmt_, DAGAssignment):
					if isinstance(stmt_.target, DAGOperand):
						if stmt_.target.name == "PC":
							is_branch = True
							if isinstance(stmt_.expr, DAGBinary):
								if stmt_.expr.name == "add":
									# TODO: allow swapped lhs and rhs + sub
									lhs, rhs = stmt_.expr.operands
									if isinstance(lhs, DAGOperand):
										if lhs.name == "PC":
											is_relative = True
											assert len(conds_) == 1
											assert isinstance(conds_[0], DAGBinary)
											cmp_name = conds_[0].name
											assert cmp_name in ["seteq", "setne", "setlt", "setgt", "setle", "setge"]
											cmp_name = cmp_name.upper()
											lhs_, rhs_ = conds_[0].operands
											new_op_name = "riscv_brcc"
											new_op = DAGQuaternary(new_op_name, lhs_, rhs_, cmp_name, rhs)
											# print("new_op", new_op)
											return ret, new_op
					# print("stmt_.target", stmt_.target)
					# print("stmt_.expr", stmt_.expr)
					# if stmts.target
				# input("aaa")
			context.pop()
		context.pop()

	context.pop()
	return ret, None

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
	return self, None

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
	return self, None

def return_(self: behav.Return, context: "VisitorContext"):
	context.push(Node("Return", parent=context.parent, ref=self))

	if self.expr is not None:
		context.push(Node("Expr", parent=context.parent))
		self.expr.generate(context)
		context.pop()

	context.pop()
	return self, None

def unary_operation(self: behav.UnaryOperation, context: "VisitorContext"):
	# print("unary_operation")
	context.push(Node("Unary Expression", parent=context.parent, ref=self))

	context.push(Node("Right", parent=context.parent))
	_, right_ = self.right.generate(context)
	context.pop()

	context.pop()
	ret_ = DAGUnary(op_name, right_)
	return self, ret_

def named_reference(self: behav.NamedReference, context: "VisitorContext"):
	# print("named_reference")
	if ":" in self.reference.name:
		ty, name = self.reference.name.split(":", 1)
		if "$" in name:
			name = name[1:]
	else:
		ty = "pc" if arch.MemoryAttribute.IS_PC in self.reference.attributes else "?"
		name = self.reference.name
	ret_ = DAGOperand(name, DAGType(ty))
	# if "GPR" in self.reference.name:
	if context.is_write:
		context.writes.append(self.reference.name)
	elif context.is_read:
		context.reads.append(self.reference.name)
	# else:
	# print("NOT GPR", self.reference.name)
	ret = self
	# for key, value in context.assignments.items():
	#		if isinstance(key, behav.ScalarDefinition):
	#		if key.scalar.name == self.reference.name:
	#			print("key", key, key.scalar)
	#			print("value", value)
	#			input("?")
	#			ret = value
	if ret == self:
		context.push(Node("Named Reference", parent=context.parent, ref=self, ref_=self.reference.name))
		context.pop()
	else:
		ret.generate(context)

	# print("ret_", ret_)
	return ret, ret_

def indexed_reference(self: behav.IndexedReference, context: "VisitorContext"):
	# print("indexed_reference")
	# print("dir", dir(self.reference), self.reference.attributes)
	# def check_gpr(node: behav.IndexedReference):
	#		ref = node.reference
	assert isinstance(self.reference, arch.Memory)
	name = self.reference.name
	ret_ = None
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
		_, index_ = self.index.generate(context)
		assert index_ is not None
		if isinstance(index_, DAGBinary):
			if index_.name == "add":
				op_name = None
				lhs, rhs = index_.operands
				lhs_type = None
				if isinstance(lhs, DAGOperand):
					lhs_type = lhs.ty.name
				rhs_type = None
				if isinstance(rhs, DAGOperand):
					rhs_type = rhs.ty.name
				if lhs_type == "GPR" and rhs_type == "GPR":
					op_name = "AddrRegReg"
				elif lhs_type == "GPR" and rhs_type == "simm12":  # TODO: check if non 12 bit allowed?
					op_name = "AddrRegImm"
				elif rhs_type == "GPR" and lhs_type == "simm12":
					op_name = "AddrRegImm"
					lhs, rhs = rhs, lhs # spwap
				else:
					raise NotImplementedError

				assert op_name is not None
				index_ = DAGBinary(op_name, lhs, rhs)  # TODO: COMPLEX
				# print("index_", index_)

		if new_op:
			ret_ = DAGUnary(new_op, index_)
			# print("ret_", ret_)
			# input("123")
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
	# if is_gpr:
	#	if context.is_write:
	#			context.writes.append(self.index.reference.name)
	# elif context.is_read:
	#	context.reads.append(self.index.reference.name)
	# else:
	#		raise NotImplementedError
	self.index.generate(context)
	# self.right.generate(context)
	context.pop()

	context.pop()
	return self, ret_

def type_conv(self: behav.TypeConv, context: "VisitorContext"):

	target_size = self.size
	if target_size is None:
		ty = self.inferred_type
		target_size = ty.width
	sign_letter = "?"
	signed = None
	if self.data_type in [arch.DataType.S, arch.DataType.U]:
		signed = self.data_type == arch.DataType.S
		if self.data_type == arch.DataType.S:
			sign_letter = "s"
			sign_letter_ = "i"
		elif self.data_type == arch.DataType.U:
			sign_letter = "u"
			# sign_letter_ = "u"
			sign_letter_ = "i"
	for ann in self.annotations:
		if "sext_inreg" in ann:
			sign_letter = ann
	context.push(Node("Type Conv", parent=context.parent, ref=self, ref_=sign_letter))


	context.push(Node("Expr", parent=context.parent))
	_, expr_ = self.expr.generate(context)
	ret_ = None
	assert expr_ is not None
	if isinstance(expr_, DAGUnary):
		if expr_.name == "load":
			assert signed is not None
			load_op = None
			assert target_size is not None
			load_op = "load"
			if target_size == 32:  # TODO: do not hardcode!
				pass
			elif target_size in [8, 16]:
				load_op = "ext" + load_op
				load_op += f"i{target_size}"
				if signed:
					load_op = "s" + load_op
				else:
					load_op = "z" + load_op
					# TODO: also implement extloadi8! (ALT)
			else:
				raise NotImplementedError
			ret_ = expr_
			ret_.name = load_op
			# print("LD")
		# input("++")
	else:
		for ann in self.annotations:
			if "sext_inreg" in ann:
				sign_letter = ann
				op_name, ty =  ann.split(" ", 1)
				ret_ = DAGBinary(op_name, expr_, ty)
	if ret_ is None:
		cast_op = sign_letter_ + str(target_size)
		ret_ = DAGUnary(cast_op, expr_)
	context.pop()

	context.pop()
	# print("ret_", ret_)
	# input("333")
	return self, ret_

def callable_(self: behav.Callable, context: "VisitorContext"):
	context.push(Node("Callable", parent=context.parent, ref=self))

	for arg, arg_descr in zip(self.args, self.ref_or_name.args):
		context.push(Node("Arg", parent=context.parent))
		arg.generate(context)
		context.pop()

	context.pop()
	return self, None

def group(self: behav.Group, context: "VisitorContext"):
	context.push(Node("Group", parent=context.parent, ref=self, ref_="()"))
	context.push(Node("Expr", parent=context.parent))
	_, ret_ = self.expr.generate(context)
	context.pop()

	context.pop()
	return self, ret_

def dereference(self: behav.DeReference, context: "VisitorContext"):
	context.push(Node("DeReference", parent=context.parent, ref=self))

	context.pop()
	return self, None

def union_definition(self: behav.UnionDefinition, context: "VisitorContext"):
	context.push(Node("Union Definition", parent=context.parent, ref=self))

	context.pop()
	return self, None
