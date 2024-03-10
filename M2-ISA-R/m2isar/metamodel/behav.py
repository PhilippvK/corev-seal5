# SPDX-License-Identifier: Apache-2.0
#
# This file is part of the M2-ISA-R project: https://github.com/tum-ei-eda/M2-ISA-R
#
# Copyright (C) 2022
# Chair of Electrical Design Automation
# Technical University of Munich

"""This module contains classes for modeling the behavioral part
of an M2-ISA-R model, this means the functional behavior of functions
and instructions. Behavior is modeled as a tree of instances of the classes
in this module. This object tree can then be traversed with transformation
functions to generate code or transform the tree.
"""

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
	from .arch import BitFieldDescr, Constant, FnParam, Function, Memory, Scalar, Composite, Union, Struct

# pylint: disable=abstract-method

class BaseNode:
	"""The base class for all behavior model classes. Only implements an
	empty generate function which raises a :exc:`NotImplementedError` if it is
	not overridden."""

	def __init__(self):
		self.annotations = []
		self.inferred_type = None

	def generate(self, context):
		raise NotImplementedError()

class CodeLiteral(BaseNode):
	def __init__(self, val) -> None:
		super().__init__()
		self.val = val

class Operator(BaseNode):
	"""Class representing an operator (of either a :class:`.UnaryOperation` or a
	:class:`.BinaryOperation`)."""

	def __init__(self, op: str):
		super().__init__()
		self.value = op

class Operation(BaseNode):
	"""Top-level collection class containing a list of actual operations."""

	def __init__(self, statements: "list[BaseNode]") -> None:
		super().__init__()
		self.statements = statements

class BinaryOperation(BaseNode):
	"""A binary operation with a left-hand and a right-hand operand as well
	as an operator."""

	def __init__(self, left: BaseNode, op: Operator, right: BaseNode):
		super().__init__()
		self.left = left
		self.op = op
		self.right = right

class SliceOperation(BaseNode):
	"""A slicing operation for extracting bit runs from scalar values."""

	def __init__(self, expr: BaseNode, left: BaseNode, right: BaseNode):
		super().__init__()
		self.expr = expr
		self.left = left
		self.right = right

class ConcatOperation(BaseNode):
	"""A concatenating operation."""

	def __init__(self, left: BaseNode, right: BaseNode) -> None:
		super().__init__()
		self.left = left
		self.right = right

class NumberLiteral(BaseNode):
	"""A class holding a generic number literal."""

	def __init__(self, value):
		super().__init__()
		self.value = value

class IntLiteral(NumberLiteral):
	"""A more precise class holding only integer literals."""

	def __init__(self, value: int, bit_size: int=None, signed: bool=None):
		super().__init__(value)

		if bit_size is None:
			self.bit_size = value.bit_length()
		else:
			self.bit_size = bit_size

		self.bit_size = max(1, self.bit_size)

		if signed is None:
			self.signed = value < 0
		else:
			self.signed = signed

class Assignment(BaseNode):
	"""An assignment statement."""

	def __init__(self, target: BaseNode, expr: BaseNode):
		super().__init__()
		self.target = target
		self.expr = expr

class Conditional(BaseNode):
	"""A conditional statement with multiple conditions and statement blocks.

	Each statement block has a corresponding condition statement. The exception
	from this is the very last statement block, which when no corresponding
	condition is present is treated as an else statement.
	"""

	def __init__(self, conds: "list[BaseNode]", stmts: "list[list[BaseNode]]"):
		super().__init__()
		self.conds = conds
		self.stmts = stmts

class Loop(BaseNode):
	"""A loop statement, representing while and do .. while loops. `post_test`
	differentiates between normal while (post_test = False) and do .. while
	(post_test=True) loops."""

	def __init__(self, cond: BaseNode, stmts: "list[BaseNode]", post_test: bool):
		super().__init__()
		self.cond = cond
		self.stmts = stmts if stmts is not None else []
		self.post_test = post_test

class Ternary(BaseNode):
	"""A ternary expression."""

	def __init__(self, cond: BaseNode, then_expr: BaseNode, else_expr: BaseNode):
		super().__init__()
		self.cond = cond
		self.then_expr = then_expr
		self.else_expr = else_expr

class ScalarDefinition(BaseNode):
	"""A scalar declaration without initialization. To initialize the scalar while
	declaring it, use the scalar definition as LHS of an assignment statement.
	"""

	def __init__(self, scalar: "Scalar"):
		super().__init__()
		self.scalar = scalar

class CompositeDefinition(BaseNode):
	"""TODO"""

	def __init__(self, composite: "Composite"):
		super().__init__()
		self.composite = composite


class UnionDefinition(CompositeDefinition):
	"""TODO"""

	def __init__(self, union: "Union"):
		super().__init__(union)

class StructDefinition(CompositeDefinition):
	"""TODO"""

	def __init__(self, struct: "Struct"):
		super().__init__(struct)

class Return(BaseNode):
	"""A return expression."""

	def __init__(self, expr: BaseNode):
		super().__init__()
		self.expr = expr

class UnaryOperation(BaseNode):
	"""An unary operation, whith an operator and a right hand operand."""

	def __init__(self, op: Operator, right: BaseNode):
		super().__init__()
		self.op = op
		self.right = right

class NamedReference(BaseNode):
	"""A named reference to a :class:`arch.Memory`, BitFieldDescr, Scalar, Constant or FnParam."""

	def __init__(self, reference: Union["Memory", "BitFieldDescr", "Scalar", "Union", "Struct", "Constant", "FnParam"]):
		super().__init__()
		self.reference = reference

class CompositeReference(BaseNode):
	"""TODO"""

	def __init__(self, reference: Union["Union", "Struct"]):
		super().__init__()
		self.reference = reference

class CompositeTypeReference(BaseNode):
	"""TODO"""

	def __init__(self, reference: Union["UnionType", "StructType"]):
		super().__init__()
		self.reference = reference

class IndexedReference(BaseNode):
	"""An indexed reference to a :class:`..arch.Memory`. Can optionally specify a range of indices
	using the `right` parameter."""

	def __init__(self, reference: "Memory", index: BaseNode, right: BaseNode=None):
		super().__init__()
		self.reference = reference
		self.index = index
		self.right = right

class DeReference(BaseNode):
	"""TODO"""

	def __init__(self, reference: BaseNode, item: str):
		super().__init__()
		self.reference = reference
		self.item = item

class TypeConv(BaseNode):
	"""A type conversion. Size can be None, in this case only the signedness is affected."""
	def __init__(self, data_type, size, expr: BaseNode):
		super().__init__()
		self.data_type = data_type
		self.size = size
		self.expr = expr

		if self.size is not None:
			self.actual_size = 1 << (self.size - 1).bit_length()
			self.actual_size = max(self.actual_size, 8)

		else:
			self.actual_size = None

class CompositeTypeConv(BaseNode):
	"""TODO"""
	def __init__(self, comp_type, expr: BaseNode):
		super().__init__()
		self.comp_type = comp_type
		self.expr = expr
		self.size = size = sum([x.size for x in self.comp_type.children])
		self.actual_size = 1 << (self.size - 1).bit_length()
		self.actual_size = max(self.actual_size, 8)

class UnionTypeConv(CompositeTypeConv):
	pass

class Callable(BaseNode):
	"""A generic invocation of a callable."""

	def __init__(self, ref_or_name: Union[str, "Function"], args: "list[BaseNode]") -> None:
		super().__init__()
		self.ref_or_name = ref_or_name
		self.args = args if args is not None else []

class FunctionCall(Callable):
	"""A function (method with return value) call."""

class ProcedureCall(Callable):
	"""A procedure (method without return value) call."""

class Group(BaseNode):
	"""A group of expressions, used e.g. for parenthesized expressions."""
	def __init__(self, expr: BaseNode):
		super().__init__()
		self.expr = expr
