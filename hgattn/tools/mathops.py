from enum import StrEnum

class BinaryOp(StrEnum):
	ADD = "add"
	MUL = "mul" 
	INTDIV = "intdiv" 
	MOD = "mod" 

class UnaryOp(StrEnum):
	ABS = "abs" 
	SIGN = "sign" 
	RELU = "relu" 
	POW2 = "pow2"
	POW3 = "pow3"

