# SPDX-License-Identifier: Apache-2.0
#
# This file is part of the M2-ISA-R project: https://github.com/tum-ei-eda/M2-ISA-R
#
# Copyright (C) 2022
# Chair of Electrical Design Automation
# Technical University of Munich

"""Utility stuff for M2-ISA-R viewer"""

from typing import TYPE_CHECKING
from anytree import Node

def get_asm_tokens(asm_str):
	def tokenize(asm):
		tokens = []
		current = None
		for c in asm_str:
			if current is None:
				if c == " ":
					continue
				elif c == "$":
					current = c
				else:
					tokens.append(c)
			else:
				if c.isalnum():
					current += c
				else:
					tokens.append(current)
					if c == "$":
						current = c
					else:
						current = None
						tokens.append(c)
		if current:
		    tokens.append(current)
		return tokens

	def helper(x):
			if x.startswith("$rs") or x.startswith("$rd"):
					return "GPR"
			elif x.startswith("$imm"):
					return "IMM"
			elif x.startswith("$"):
					return "???"
			else:
					return x

	tokens = tokenize(asm_str)

	tokens_ = [helper(t) for t in tokens]

	return tokens_

class DAGNode:
	pass


class DAGLeaf(DAGNode):
	pass


class DAGType(DAGNode):
	def __init__(self, name):
		# print("DAGType")
		assert isinstance(name, str)
		self.name : str = name

	def __repr__(self):
		return self.name


class DAGImm(DAGLeaf):
	def __init__(self, value):
		# print("DAGImm")
		self.value = value

	def __repr__(self):
		return str(self.value)


class DAGOperand(DAGLeaf):
	def __init__(self, name, ty):
		# print("DAGOperand")
		self.name = name
		self.ty : DAGType = ty

	def __repr__(self):
		return f"{self.ty}:${self.name}"


class DAGOperation(DAGNode):
	def __init__(self, name, operands):
		# print("DAGOperation")
		self.name : str = name
		self.operands : list = operands
		assert len(self.operands) > 0

	def __repr__(self):
		return f"({self.name} " + ", ".join(map(str, self.operands)) + ")"


class DAGAssignment(DAGNode):
	def __init__(self, target, expr):
		# print("DAGAssignment")
		self.target = target
		self.expr = expr

	def __repr__(self):
		return f"{self.target} <- {self.expr}"


class DAGUnary(DAGOperation):
	def __init__(self, name, right):
		super().__init__(name, [right])


class DAGBinary(DAGOperation):
	def __init__(self, name, left, right):
		super().__init__(name, [left, right])


class DAGTernary(DAGOperation):
	def __init__(self, name, first, second, third):
		super().__init__(name, [first, second, third])


class DAGQuaternary(DAGOperation):
	def __init__(self, name, first, second, third, fourth):
		super().__init__(name, [first, second, third, fourth])

class EliminateScalarsContext:

	def __init__(self) -> None:
		self.assignments = {}
		self.used = []

	@property
	def unused(self):
		return [x for x in self.assignments if x not in self.used]

class CollectThrowsContext:

	def __init__(self) -> None:
		self.throw_reasons = {}

class CollectRWContext:

	def __init__(self) -> None:
		self.reads = []
		self.writes = []
		self.is_read = False
		self.is_write = False

class VisitorContext:

	def __init__(self) -> None:
		self.nodes = [Node("Tree")]
		self.parent_stack = [0]
		self.layer = 0
		self.reads = []
		self.writes = []
		self.is_read = False
		self.is_write = False

	@property
	def parent(self):
		return self.nodes[self.parent_stack[-1]]

	@property
	def tree(self):
		return self.nodes[0]

	def push(self, node):
		self.layer += 1
		self.parent_stack.append(len(self.nodes))
		self.nodes.append(node)


	def pop(self):
		self.layer -= 1
		return self.parent_stack.pop()

	def print(self, *args, **kwargs):
		print("   " * self.layer, *args, **kwargs)

class DAGBuilderContext(CollectRWContext):
	def __init__(self) -> None:
		super().__init__()
		self.patterns = []
		self.complex_patterns = []

class AnalysisContext(CollectRWContext):
	def __init__(self) -> None:
		super().__init__()
		self.may_load = False
		self.may_store = False
		self.has_side_effects = False
		self.is_branch = False
		self.is_terminator = False
		self.is_barrier = False
		self.is_return = False
		self.is_call = False
		self.is_re_materializable = False
		self.is_as_cheap_as_a_move = False
		self.nodes = [Node("Tree")]
		self.parent_stack = [0]
		self.layer = 0

	@property
	def parent(self):
		return self.nodes[self.parent_stack[-1]]

	@property
	def tree(self):
		return self.nodes[0]

	def push(self, node):
		self.layer += 1
		self.parent_stack.append(len(self.nodes))
		self.nodes.append(node)


	def pop(self):
		self.layer -= 1
		return self.parent_stack.pop()

		print("   " * self.layer, *args, **kwargs)
