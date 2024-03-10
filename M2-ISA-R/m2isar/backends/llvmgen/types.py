# TODO
from typing import Optional
from enum import Enum

class Artifact:
	def __init__(self, path: str, content: str, append: bool = False):
		self.path: str = path
		self.content: str = content
		self.append: bool = append

	@property
	def out_path(self):
		raise NotImplementedError

	def to_dict(self) -> dict:
		return {key: value for key, value in vars(self).items() if key not in ["content"]}

class GitPatch(Artifact):
	def __init__(self, path: str, content: str, message=None, append: bool = False):
		super().__init__(path, content, append=append)
		self.message = "???" if message is None else message
		# TODO: context
		raise NotImplementedError

class Patch(Artifact):
	def __init__(self, path: str, content: str, append: bool = False):
		super().__init__(path, content, append=append)

class NamedPatch(Patch):
	def __init__(self, path: str, content: str, key: str, append: bool = False):
		super().__init__(path, content, append=append)
		self.key: str = key
		self.src: str = self.out_path

	@property
	def out_path(self):
		return self.path + "." + self.key

class IndexedPatch(Patch):
	def __init__(self, path: str, content: str, line: int, append: bool = False):
		super().__init__(path, content, append=append)
		# 0: start of file, -1: end of file
		self.line: int = line
		raise NotImplementedError

class RangedPatch(Patch):
	def __init__(self, path: str, content: str, start: str, end: str, append: bool = False):
		super().__init__(path, content, append=append)
		# Patch is added between matching lines
		self.start: str = start
		self.end: str = end
		raise NotImplementedError

class File(Artifact):
	def __init__(self, path: str, content: str, append: bool = False):
		super().__init__(path, content, append=append)

	@property
	def out_path(self):
		return self.path


class Version:
	def __init__(self, major: int, minor: int):
		self.major = major
		self.minor = minor


class Extension:

	def __init__(self, name: str, identifier: Optional[str] = None, description: Optional[str] = None, experimental: bool = False, custom: bool = True, version: Optional[Version] = None):
		self.name = name
		assert isinstance(self.name, str)
		assert self.name != self.name.lower(), "Extension names should not be lower-case only"
		self.identifier = self.name.lower() if identifier is None else identifier
		assert isinstance(self.name, str)
		assert self.identifier == self.identifier.lower(), "Extension names should be lower case"
		self.description = "Unknown" if description is None else description
		assert isinstance(self.description, str)
		self.experimental = experimental
		self.custom = custom  # If x-Prefix should be used
		if self.custom:
			if self.name.upper()[0] != "X":
				self.name = "X" + self.name
			if self.identifier.lower()[0] != "x":
				self.identifier = "x" + self.identifier
		else:
			if self.name.upper()[0] == "X":
				logger.warning("X-Prefix is used for name of non-custom extension")
			if self.identifier.lower()[0] == "x":
				logger.warning("x-Prefix is used for identifier of non-custom extension")
		self.version = Version(1, 0) if version is None else version

		self.artifacts = []

	def add_artifact(self, artifact: Artifact):
		self.artifacts.append(artifact)

	def to_dict(self):
		return {
			"name": self.name,
			"identifier": self.identifier,
			"description": self.description,
			"experimental": self.experimental,
			"artifacts": list(map(lambda artifact: artifact.to_dict(), self.artifacts)),
		}


class ImmOperand:
	def __init__(self, name, bits, signed=False):
		self.name = name  # TODO: automatically?
		self.bits = bits
		self.signed = signed
		self.sign_letter = "s" if self.signed else "u"
		self.sign_letter_upper = self.sign_letter.upper()

	def gen_tablegen(self):  # TODO: move somewhere else
		return f"""def {self.name} : Operand<XLenVT>, ImmLeaf<XLenVT, [{{return is{self.sign_letter_upper}Int<{self.bits}>(Imm);}}]> {{
  let ParserMatchClass = {self.sign_letter_upper}ImmAsmOperand<{self.bits}>;
  let DecoderMethod = "decode{self.sign_letter_upper}ImmOperand<{self.bits}>";
  let OperandType = "OPERAND_{self.sign_letter_upper}IMM{self.bits}";
  let OperandNamespace = "RISCVOp";
}}
"""

	def gen_cpp(self):
		return f"OPERAND_{self.sign_letter_upper}IMM{self.bits},\n"

class LegalizeAction(Enum):
	Legal = 0
	Expand = 1
	Custom = 2
	Promote = 3

	def __str__(self):
		return self.name

class MVT(Enum):  # MachineValueType
	XLenVT = 0
	i8 = 1
	i16 = 2
	i32 = 3
	i1 = 4
	# integer vector types
	v1i1 = 17
	v2i1 = 18
	v4i1 = 19
	v8i1 = 20
	v16i1 = 21
	v32i1 = 22
	v64i1 = 23
	v128i1 = 24
	v256i1 = 25
	v512i1 = 26
	v1024i1 = 27
	v2048i1 = 28
	v128i2 = 29
	v256i2 = 30
	v64i4 = 31
	v128i4 = 32
	v1i8 = 33
	v2i8 = 34
	v4i8 = 35
	v8i8 = 36
	v16i8 = 37
	v32i8 = 38
	v64i8 = 39
	v128i8 = 40
	v256i8 = 41
	v512i8 = 42
	v1024i8 = 43
	v1i16 = 44
	v2i16 = 45
	v3i16 = 46
	v4i16 = 47
	v8i16 = 48
	v16i16 = 49
	v32i16 = 50
	v64i16 = 51
	v128i16 = 52
	v256i16 = 53
	v512i16 = 54
	v1i32 = 55
	v2i32 = 56
	v3i32 = 57
	v4i32 = 58
	v5i32 = 59
	v6i32 = 60
	v7i32 = 61
	v8i32 = 62
	v9i32 = 63
	v10i32 = 64
	v11i32 = 65
	v12i32 = 66
	v16i32 = 67
	v32i32 = 68
	v64i32 = 69
	v128i32 = 70
	v256i32 = 71
	v512i32 = 72
	v1024i32 = 73
	v2048i32 = 74
	v1i64 = 75
	v2i64 = 76
	v3i64 = 77
	v4i64 = 78
	v8i64 = 79
	v16i64 = 80
	v32i64 = 81
	v64i64 = 82
	v128i64 = 83
	v256i64 = 84
	v1i128 = 85

	def __str__(self):
		if self.name == "XLenVT":
			return self.name
		return f"MVT::{self.name}"

INTEGER_FIXEDLEN_VECTOR_VALUETYPES = [MVT.v1i1, MVT.v2i1, MVT.v4i1, MVT.v8i1, MVT.v16i1, MVT.v32i1, MVT.v64i1, MVT.v128i1, MVT.v256i1, MVT.v512i1, MVT.v1024i1, MVT.v2048i1, MVT.v128i2, MVT.v256i2, MVT.v64i4, MVT.v128i4, MVT.v1i8, MVT.v2i8, MVT.v4i8, MVT.v8i8, MVT.v16i8, MVT.v32i8, MVT.v64i8, MVT.v128i8, MVT.v256i8, MVT.v512i8, MVT.v1024i8, MVT.v1i16, MVT.v2i16, MVT.v3i16, MVT.v4i16, MVT.v8i16, MVT.v16i16, MVT.v32i16, MVT.v64i16, MVT.v128i16, MVT.v256i16, MVT.v512i16, MVT.v1i32, MVT.v2i32, MVT.v3i32, MVT.v4i32, MVT.v5i32, MVT.v6i32, MVT.v7i32, MVT.v8i32, MVT.v9i32, MVT.v10i32, MVT.v11i32, MVT.v12i32, MVT.v16i32, MVT.v32i32, MVT.v64i32, MVT.v128i32, MVT.v256i32, MVT.v512i32, MVT.v1024i32, MVT.v2048i32, MVT.v1i64, MVT.v2i64, MVT.v3i64, MVT.v4i64, MVT.v8i64, MVT.v16i64, MVT.v32i64, MVT.v64i64, MVT.v128i64, MVT.v256i64, MVT.v1i128]

class ISD(Enum):
	POST_INC = 0
	INTRINSIC_W_CHAIN = 1
	SIGN_EXTEND_INREG = 2
	UMAX = 3
	SMAX = 4
	UMIN = 5
	SMIN = 6
	ABS = 7
	SETULE = 8
	SETLE = 9
	ADD = 10
	SUB = 11
	AND = 12
	OR = 13
	XOR = 14
	SHL = 15
	SRA = 16
	SRL = 17
	BITCAST = 18
	VECREDUCE_ADD = 19
	SPLAT_VECTOR = 20
	SETCC = 21
	VECTOR_SHUFFLE = 22
	EXTRACT_VECTOR_ELT = 23
	BUILD_VECTOR = 24
	UNDEF = 25
	EXTLOAD = 26
	SEXTLOAD = 27
	ZEXTLOAD = 28


	def __str__(self):
		return f"ISD::{self.name}"


class Legalization:
	def __init__(self):
		pass

class OperationAction(Legalization):
	def __init__(self, nodes, vt, action):
		self.nodes = nodes if isinstance(nodes, list) else [nodes]
		self.vt = vt
		self.action = action

class IndexedModeAction(Legalization):
	def __init__(self, nodes, vt, action, is_store=False):
		self.nodes = nodes if isinstance(nodes, list) else [nodes]
		self.vt = vt
		self.action = action
		self.is_store = is_store

class CondCodeAction(Legalization):
	def __init__(self, nodes, vt, action, is_store=False):
		self.nodes = nodes if isinstance(nodes, list) else [nodes]
		self.vt = vt
		self.action = action

class LoadExtAction(Legalization):
	def __init__(self, nodes, vt, vt_, action):
		self.nodes = nodes if isinstance(nodes, list) else [nodes]
		self.vt = vt
		self.vt_ = vt_
		self.action = action

class TruncStoreAction(Legalization):
	def __init__(self, vt, vt_, action):
		self.vt = vt
		self.vt_ = vt_
		self.action = action

class OperationPromotedToType(Legalization):
	def __init__(self, nodes, vt, vt_):
		self.nodes = nodes if isinstance(nodes, list) else [nodes]
		self.vt = vt
		self.vt_ = vt_


# class TargetDAGCombine
# class RegClass

class ComplexPattern:
	def __init__(self, name, func):
		self.name = name
		self.func = func
		# TODO: find out what these are
		self.foo = "iPTR"
		self.bar = 2

	def gen_tablegen(self):
		return f"def {self.name} : ComplexPattern<{self.foo}, {self.bar}, \"{self.func}\">;"
