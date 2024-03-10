# SPDX-License-Identifier: Apache-2.0
#
# This file is part of the M2-ISA-R project: https://github.com/tum-ei-eda/M2-ISA-R
#
# Copyright (C) 2022
# Chair of Electrical Design Automation
# Technical University of Munich

"""A transformation module for simplifying M2-ISA-R behavior expressions. The following
simplifications are done:

* Resolvable :class:`m2isar.metamodel.arch.Constant` s are replaced by
  `m2isar.metamodel.arch.IntLiteral` s representing their value
* Fully resolvable arithmetic operations are carried out and their results
  represented as a matching :class:`m2isar.metamodel.arch.IntLiteral`
* Conditions and loops with fully resolvable conditions are either discarded entirely
  or transformed into code blocks without any conditions
* Ternaries with fully resolvable conditions are transformed into only the matching part
* Type conversions of :class:`m2isar.metamodel.arch.IntLiteral` s apply the desired
  type directly to the :class:`IntLiteral` and discard the type conversion
"""

import logging

logger = logging.getLogger(__name__)

from m2isar.metamodel import arch, behav

# pylint: disable=unused-argument

def operation(self: behav.Operation, context):
	statements = []
	for stmt in self.statements:
		temp = stmt.generate(context)
		if isinstance(temp, list):
			statements.extend(temp)
		else:
			statements.append(temp)

	self.statements = statements
	return self

def binary_operation(self: behav.BinaryOperation, context):

	self.left = self.left.generate(context)
	self.right = self.right.generate(context)

	return self

def slice_operation(self: behav.SliceOperation, context):
	# print("slice_operation!")
	self.expr = self.expr.generate(context)
	self.left = self.left.generate(context)
	self.right = self.right.generate(context)

	return self

def concat_operation(self: behav.ConcatOperation, context):
	self.left = self.left.generate(context)
	self.right = self.right.generate(context)

	return self

def number_literal(self: behav.IntLiteral, context):
	return self

def int_literal(self: behav.IntLiteral, context):
	return self

def scalar_definition(self: behav.ScalarDefinition, context):
	return self

def assignment(self: behav.Assignment, context):
	self.target = self.target.generate(context)
	self.expr = self.expr.generate(context)

	#if isinstance(self.expr, behav.IntLiteral) and isinstance(self.target, behav.ScalarDefinition):
#		self.target.scalar.value = self.expr.value

	return self

def conditional(self: behav.Conditional, context):
	self.conds = [x.generate(context) for x in self.conds]
	self.stmts = [[y.generate(context) for y in x] for x in self.stmts]

	return self

def loop(self: behav.Loop, context):
	self.cond = self.cond.generate(context)
	self.stmts = [x.generate(context) for x in self.stmts]

	return self

def ternary(self: behav.Ternary, context):
	self.cond = self.cond.generate(context)
	self.then_expr = self.then_expr.generate(context)
	self.else_expr = self.else_expr.generate(context)

	return self

def return_(self: behav.Return, context):
	if self.expr is not None:
		self.expr = self.expr.generate(context)

	return self

def unary_operation(self: behav.UnaryOperation, context):
	self.right = self.right.generate(context)

	return self

def named_reference(self: behav.NamedReference, context):

	return self

def indexed_reference(self: behav.IndexedReference, context):
	self.index = self.index.generate(context)

	return self

def type_conv(self: behav.TypeConv, context):
	# print("type_conv!")
	target_type = self.data_type
	target_size = self.size
	if isinstance(self.expr, behav.SliceOperation):
		# print("conv slice!")
		# always unsigned slice
		left = self.expr.left
		if not isinstance(left, behav.IntLiteral):
			logger.warning("LHS of slice not static: skipping")
			return self
		right = self.expr.right
		if not isinstance(right, behav.IntLiteral):
			logger.warning("RHS of slice not static: skipping")
			return self
		assert left.value >= right.value
		width = left.value - right.value + 1
		# print("width", width)
		if isinstance(self.expr.expr, behav.NamedReference):
			if "GPR" in self.expr.expr.reference.name:
				# TODO: need inferred type here for real register size
				ALLOWED = [8,16,32]  # TODO
				XLEN = 32 # TODO (Get from reg)
				if width in ALLOWED:
					if target_size is None:
						if target_type == arch.DataType.S:
							# print("self", self, dir(self))
							self.annotations.append(f"sext_inreg i{width}")
							self.expr = self.expr.expr.generate(context)
							return self
					else:
						if target_size > width:
							self.annotations.append(f"sext_inreg i{width}")
							self.expr = self.expr.expr.generate(context)
							return self
						else:
							# print("self", self, dir(self))
							# self.annotations.append(f"sext_inreg i{width}")
							# print("target_size", target_size)
							# print("width", width)
							# raise NotImplementedError
							pass



	self.expr = self.expr.generate(context)

	return self

def callable_(self: behav.Callable, context):
	self.args = [stmt.generate(context) for stmt in self.args]

	return self

def group(self: behav.Group, context):
	self.expr = self.expr.generate(context)

	if isinstance(self.expr, behav.IntLiteral):
		return self.expr

	return self
