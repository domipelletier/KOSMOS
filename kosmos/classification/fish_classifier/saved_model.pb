Â&
®
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
3
Square
x"T
y"T"
Ttype:
2
	
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-0-g919f693420e8Ùü!
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
:*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:*
dtype0

batchNormalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatchNormalization_1/gamma

.batchNormalization_1/gamma/Read/ReadVariableOpReadVariableOpbatchNormalization_1/gamma*
_output_shapes
:*
dtype0

batchNormalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatchNormalization_1/beta

-batchNormalization_1/beta/Read/ReadVariableOpReadVariableOpbatchNormalization_1/beta*
_output_shapes
:*
dtype0

 batchNormalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" batchNormalization_1/moving_mean

4batchNormalization_1/moving_mean/Read/ReadVariableOpReadVariableOp batchNormalization_1/moving_mean*
_output_shapes
:*
dtype0
 
$batchNormalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batchNormalization_1/moving_variance

8batchNormalization_1/moving_variance/Read/ReadVariableOpReadVariableOp$batchNormalization_1/moving_variance*
_output_shapes
:*
dtype0

conv_1b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_1b/kernel
y
"conv_1b/kernel/Read/ReadVariableOpReadVariableOpconv_1b/kernel*&
_output_shapes
: *
dtype0
p
conv_1b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_1b/bias
i
 conv_1b/bias/Read/ReadVariableOpReadVariableOpconv_1b/bias*
_output_shapes
: *
dtype0

batchNormalization_1b/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatchNormalization_1b/gamma

/batchNormalization_1b/gamma/Read/ReadVariableOpReadVariableOpbatchNormalization_1b/gamma*
_output_shapes
: *
dtype0

batchNormalization_1b/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatchNormalization_1b/beta

.batchNormalization_1b/beta/Read/ReadVariableOpReadVariableOpbatchNormalization_1b/beta*
_output_shapes
: *
dtype0

!batchNormalization_1b/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batchNormalization_1b/moving_mean

5batchNormalization_1b/moving_mean/Read/ReadVariableOpReadVariableOp!batchNormalization_1b/moving_mean*
_output_shapes
: *
dtype0
¢
%batchNormalization_1b/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batchNormalization_1b/moving_variance

9batchNormalization_1b/moving_variance/Read/ReadVariableOpReadVariableOp%batchNormalization_1b/moving_variance*
_output_shapes
: *
dtype0
~
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv_2/kernel
w
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*&
_output_shapes
: @*
dtype0
n
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_2/bias
g
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes
:@*
dtype0

conv_2b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv_2b/kernel
y
"conv_2b/kernel/Read/ReadVariableOpReadVariableOpconv_2b/kernel*&
_output_shapes
:@@*
dtype0
p
conv_2b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_2b/bias
i
 conv_2b/bias/Read/ReadVariableOpReadVariableOpconv_2b/bias*
_output_shapes
:@*
dtype0

batchNormalization_2b/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatchNormalization_2b/gamma

/batchNormalization_2b/gamma/Read/ReadVariableOpReadVariableOpbatchNormalization_2b/gamma*
_output_shapes
:@*
dtype0

batchNormalization_2b/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatchNormalization_2b/beta

.batchNormalization_2b/beta/Read/ReadVariableOpReadVariableOpbatchNormalization_2b/beta*
_output_shapes
:@*
dtype0

!batchNormalization_2b/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batchNormalization_2b/moving_mean

5batchNormalization_2b/moving_mean/Read/ReadVariableOpReadVariableOp!batchNormalization_2b/moving_mean*
_output_shapes
:@*
dtype0
¢
%batchNormalization_2b/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batchNormalization_2b/moving_variance

9batchNormalization_2b/moving_variance/Read/ReadVariableOpReadVariableOp%batchNormalization_2b/moving_variance*
_output_shapes
:@*
dtype0

conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_3/kernel
x
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*'
_output_shapes
:@*
dtype0
o
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/bias
h
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes	
:*
dtype0

batchNormalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatchNormalization_3/gamma

.batchNormalization_3/gamma/Read/ReadVariableOpReadVariableOpbatchNormalization_3/gamma*
_output_shapes	
:*
dtype0

batchNormalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatchNormalization_3/beta

-batchNormalization_3/beta/Read/ReadVariableOpReadVariableOpbatchNormalization_3/beta*
_output_shapes	
:*
dtype0

 batchNormalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" batchNormalization_3/moving_mean

4batchNormalization_3/moving_mean/Read/ReadVariableOpReadVariableOp batchNormalization_3/moving_mean*
_output_shapes	
:*
dtype0
¡
$batchNormalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batchNormalization_3/moving_variance

8batchNormalization_3/moving_variance/Read/ReadVariableOpReadVariableOp$batchNormalization_3/moving_variance*
_output_shapes	
:*
dtype0

conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4/kernel
y
!conv_4/kernel/Read/ReadVariableOpReadVariableOpconv_4/kernel*(
_output_shapes
:*
dtype0
o
conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4/bias
h
conv_4/bias/Read/ReadVariableOpReadVariableOpconv_4/bias*
_output_shapes	
:*
dtype0

batchNormalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatchNormalization_4/gamma

.batchNormalization_4/gamma/Read/ReadVariableOpReadVariableOpbatchNormalization_4/gamma*
_output_shapes	
:*
dtype0

batchNormalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatchNormalization_4/beta

-batchNormalization_4/beta/Read/ReadVariableOpReadVariableOpbatchNormalization_4/beta*
_output_shapes	
:*
dtype0

 batchNormalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" batchNormalization_4/moving_mean

4batchNormalization_4/moving_mean/Read/ReadVariableOpReadVariableOp batchNormalization_4/moving_mean*
_output_shapes	
:*
dtype0
¡
$batchNormalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batchNormalization_4/moving_variance

8batchNormalization_4/moving_variance/Read/ReadVariableOpReadVariableOp$batchNormalization_4/moving_variance*
_output_shapes	
:*
dtype0

batchNormalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatchNormalization_5/gamma

.batchNormalization_5/gamma/Read/ReadVariableOpReadVariableOpbatchNormalization_5/gamma*
_output_shapes	
:*
dtype0

batchNormalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatchNormalization_5/beta

-batchNormalization_5/beta/Read/ReadVariableOpReadVariableOpbatchNormalization_5/beta*
_output_shapes	
:*
dtype0

 batchNormalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" batchNormalization_5/moving_mean

4batchNormalization_5/moving_mean/Read/ReadVariableOpReadVariableOp batchNormalization_5/moving_mean*
_output_shapes	
:*
dtype0
¡
$batchNormalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batchNormalization_5/moving_variance

8batchNormalization_5/moving_variance/Read/ReadVariableOpReadVariableOp$batchNormalization_5/moving_variance*
_output_shapes	
:*
dtype0

dense_final/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_namedense_final/kernel
z
&dense_final/kernel/Read/ReadVariableOpReadVariableOpdense_final/kernel*
_output_shapes
:	*
dtype0
x
dense_final/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedense_final/bias
q
$dense_final/bias/Read/ReadVariableOpReadVariableOpdense_final/bias*
_output_shapes
:*
dtype0
j
Adamax/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdamax/iter
c
Adamax/iter/Read/ReadVariableOpReadVariableOpAdamax/iter*
_output_shapes
: *
dtype0	
n
Adamax/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_1
g
!Adamax/beta_1/Read/ReadVariableOpReadVariableOpAdamax/beta_1*
_output_shapes
: *
dtype0
n
Adamax/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_2
g
!Adamax/beta_2/Read/ReadVariableOpReadVariableOpAdamax/beta_2*
_output_shapes
: *
dtype0
l
Adamax/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/decay
e
 Adamax/decay/Read/ReadVariableOpReadVariableOpAdamax/decay*
_output_shapes
: *
dtype0
|
Adamax/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdamax/learning_rate
u
(Adamax/learning_rate/Read/ReadVariableOpReadVariableOpAdamax/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adamax/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdamax/conv_1/kernel/m

*Adamax/conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv_1/kernel/m*&
_output_shapes
:*
dtype0

Adamax/conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdamax/conv_1/bias/m
y
(Adamax/conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv_1/bias/m*
_output_shapes
:*
dtype0

#Adamax/batchNormalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adamax/batchNormalization_1/gamma/m

7Adamax/batchNormalization_1/gamma/m/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_1/gamma/m*
_output_shapes
:*
dtype0

"Adamax/batchNormalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adamax/batchNormalization_1/beta/m

6Adamax/batchNormalization_1/beta/m/Read/ReadVariableOpReadVariableOp"Adamax/batchNormalization_1/beta/m*
_output_shapes
:*
dtype0

Adamax/conv_1b/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdamax/conv_1b/kernel/m

+Adamax/conv_1b/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv_1b/kernel/m*&
_output_shapes
: *
dtype0

Adamax/conv_1b/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdamax/conv_1b/bias/m
{
)Adamax/conv_1b/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv_1b/bias/m*
_output_shapes
: *
dtype0
 
$Adamax/batchNormalization_1b/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batchNormalization_1b/gamma/m

8Adamax/batchNormalization_1b/gamma/m/Read/ReadVariableOpReadVariableOp$Adamax/batchNormalization_1b/gamma/m*
_output_shapes
: *
dtype0

#Adamax/batchNormalization_1b/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adamax/batchNormalization_1b/beta/m

7Adamax/batchNormalization_1b/beta/m/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_1b/beta/m*
_output_shapes
: *
dtype0

Adamax/conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdamax/conv_2/kernel/m

*Adamax/conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv_2/kernel/m*&
_output_shapes
: @*
dtype0

Adamax/conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdamax/conv_2/bias/m
y
(Adamax/conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv_2/bias/m*
_output_shapes
:@*
dtype0

Adamax/conv_2b/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdamax/conv_2b/kernel/m

+Adamax/conv_2b/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv_2b/kernel/m*&
_output_shapes
:@@*
dtype0

Adamax/conv_2b/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdamax/conv_2b/bias/m
{
)Adamax/conv_2b/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv_2b/bias/m*
_output_shapes
:@*
dtype0
 
$Adamax/batchNormalization_2b/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adamax/batchNormalization_2b/gamma/m

8Adamax/batchNormalization_2b/gamma/m/Read/ReadVariableOpReadVariableOp$Adamax/batchNormalization_2b/gamma/m*
_output_shapes
:@*
dtype0

#Adamax/batchNormalization_2b/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adamax/batchNormalization_2b/beta/m

7Adamax/batchNormalization_2b/beta/m/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_2b/beta/m*
_output_shapes
:@*
dtype0

Adamax/conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdamax/conv_3/kernel/m

*Adamax/conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv_3/kernel/m*'
_output_shapes
:@*
dtype0

Adamax/conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdamax/conv_3/bias/m
z
(Adamax/conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv_3/bias/m*
_output_shapes	
:*
dtype0

#Adamax/batchNormalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adamax/batchNormalization_3/gamma/m

7Adamax/batchNormalization_3/gamma/m/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_3/gamma/m*
_output_shapes	
:*
dtype0

"Adamax/batchNormalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adamax/batchNormalization_3/beta/m

6Adamax/batchNormalization_3/beta/m/Read/ReadVariableOpReadVariableOp"Adamax/batchNormalization_3/beta/m*
_output_shapes	
:*
dtype0

Adamax/conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdamax/conv_4/kernel/m

*Adamax/conv_4/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv_4/kernel/m*(
_output_shapes
:*
dtype0

Adamax/conv_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdamax/conv_4/bias/m
z
(Adamax/conv_4/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv_4/bias/m*
_output_shapes	
:*
dtype0

#Adamax/batchNormalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adamax/batchNormalization_4/gamma/m

7Adamax/batchNormalization_4/gamma/m/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_4/gamma/m*
_output_shapes	
:*
dtype0

"Adamax/batchNormalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adamax/batchNormalization_4/beta/m

6Adamax/batchNormalization_4/beta/m/Read/ReadVariableOpReadVariableOp"Adamax/batchNormalization_4/beta/m*
_output_shapes	
:*
dtype0

#Adamax/batchNormalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adamax/batchNormalization_5/gamma/m

7Adamax/batchNormalization_5/gamma/m/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_5/gamma/m*
_output_shapes	
:*
dtype0

"Adamax/batchNormalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adamax/batchNormalization_5/beta/m

6Adamax/batchNormalization_5/beta/m/Read/ReadVariableOpReadVariableOp"Adamax/batchNormalization_5/beta/m*
_output_shapes	
:*
dtype0

Adamax/dense_final/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameAdamax/dense_final/kernel/m

/Adamax/dense_final/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/dense_final/kernel/m*
_output_shapes
:	*
dtype0

Adamax/dense_final/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdamax/dense_final/bias/m

-Adamax/dense_final/bias/m/Read/ReadVariableOpReadVariableOpAdamax/dense_final/bias/m*
_output_shapes
:*
dtype0

Adamax/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdamax/conv_1/kernel/v

*Adamax/conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv_1/kernel/v*&
_output_shapes
:*
dtype0

Adamax/conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdamax/conv_1/bias/v
y
(Adamax/conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv_1/bias/v*
_output_shapes
:*
dtype0

#Adamax/batchNormalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adamax/batchNormalization_1/gamma/v

7Adamax/batchNormalization_1/gamma/v/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_1/gamma/v*
_output_shapes
:*
dtype0

"Adamax/batchNormalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adamax/batchNormalization_1/beta/v

6Adamax/batchNormalization_1/beta/v/Read/ReadVariableOpReadVariableOp"Adamax/batchNormalization_1/beta/v*
_output_shapes
:*
dtype0

Adamax/conv_1b/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdamax/conv_1b/kernel/v

+Adamax/conv_1b/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv_1b/kernel/v*&
_output_shapes
: *
dtype0

Adamax/conv_1b/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdamax/conv_1b/bias/v
{
)Adamax/conv_1b/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv_1b/bias/v*
_output_shapes
: *
dtype0
 
$Adamax/batchNormalization_1b/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adamax/batchNormalization_1b/gamma/v

8Adamax/batchNormalization_1b/gamma/v/Read/ReadVariableOpReadVariableOp$Adamax/batchNormalization_1b/gamma/v*
_output_shapes
: *
dtype0

#Adamax/batchNormalization_1b/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adamax/batchNormalization_1b/beta/v

7Adamax/batchNormalization_1b/beta/v/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_1b/beta/v*
_output_shapes
: *
dtype0

Adamax/conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdamax/conv_2/kernel/v

*Adamax/conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv_2/kernel/v*&
_output_shapes
: @*
dtype0

Adamax/conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdamax/conv_2/bias/v
y
(Adamax/conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv_2/bias/v*
_output_shapes
:@*
dtype0

Adamax/conv_2b/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdamax/conv_2b/kernel/v

+Adamax/conv_2b/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv_2b/kernel/v*&
_output_shapes
:@@*
dtype0

Adamax/conv_2b/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdamax/conv_2b/bias/v
{
)Adamax/conv_2b/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv_2b/bias/v*
_output_shapes
:@*
dtype0
 
$Adamax/batchNormalization_2b/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adamax/batchNormalization_2b/gamma/v

8Adamax/batchNormalization_2b/gamma/v/Read/ReadVariableOpReadVariableOp$Adamax/batchNormalization_2b/gamma/v*
_output_shapes
:@*
dtype0

#Adamax/batchNormalization_2b/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adamax/batchNormalization_2b/beta/v

7Adamax/batchNormalization_2b/beta/v/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_2b/beta/v*
_output_shapes
:@*
dtype0

Adamax/conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdamax/conv_3/kernel/v

*Adamax/conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv_3/kernel/v*'
_output_shapes
:@*
dtype0

Adamax/conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdamax/conv_3/bias/v
z
(Adamax/conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv_3/bias/v*
_output_shapes	
:*
dtype0

#Adamax/batchNormalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adamax/batchNormalization_3/gamma/v

7Adamax/batchNormalization_3/gamma/v/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_3/gamma/v*
_output_shapes	
:*
dtype0

"Adamax/batchNormalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adamax/batchNormalization_3/beta/v

6Adamax/batchNormalization_3/beta/v/Read/ReadVariableOpReadVariableOp"Adamax/batchNormalization_3/beta/v*
_output_shapes	
:*
dtype0

Adamax/conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdamax/conv_4/kernel/v

*Adamax/conv_4/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv_4/kernel/v*(
_output_shapes
:*
dtype0

Adamax/conv_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdamax/conv_4/bias/v
z
(Adamax/conv_4/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv_4/bias/v*
_output_shapes	
:*
dtype0

#Adamax/batchNormalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adamax/batchNormalization_4/gamma/v

7Adamax/batchNormalization_4/gamma/v/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_4/gamma/v*
_output_shapes	
:*
dtype0

"Adamax/batchNormalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adamax/batchNormalization_4/beta/v

6Adamax/batchNormalization_4/beta/v/Read/ReadVariableOpReadVariableOp"Adamax/batchNormalization_4/beta/v*
_output_shapes	
:*
dtype0

#Adamax/batchNormalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adamax/batchNormalization_5/gamma/v

7Adamax/batchNormalization_5/gamma/v/Read/ReadVariableOpReadVariableOp#Adamax/batchNormalization_5/gamma/v*
_output_shapes	
:*
dtype0

"Adamax/batchNormalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adamax/batchNormalization_5/beta/v

6Adamax/batchNormalization_5/beta/v/Read/ReadVariableOpReadVariableOp"Adamax/batchNormalization_5/beta/v*
_output_shapes	
:*
dtype0

Adamax/dense_final/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameAdamax/dense_final/kernel/v

/Adamax/dense_final/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/dense_final/kernel/v*
_output_shapes
:	*
dtype0

Adamax/dense_final/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdamax/dense_final/bias/v

-Adamax/dense_final/bias/v/Read/ReadVariableOpReadVariableOpAdamax/dense_final/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Á«
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ûª
valueðªBìª Bäª
þ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
layer-21
layer_with_weights-12
layer-22
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api

$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*	variables
+trainable_variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api

7axis
	8gamma
9beta
:moving_mean
;moving_variance
<regularization_losses
=	variables
>trainable_variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
R
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
R
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
h

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
R
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api

\axis
	]gamma
^beta
_moving_mean
`moving_variance
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
h

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
R
kregularization_losses
l	variables
mtrainable_variables
n	keras_api

oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
R
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
j

|kernel
}bias
~regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
 kernel
	¡bias
¢regularization_losses
£	variables
¤trainable_variables
¥	keras_api
Ù
	¦iter
§beta_1
¨beta_2

©decay
ªlearning_ratem®m¯%m°&m±-m².m³8m´9mµ@m¶Am·Rm¸Sm¹]mº^m»em¼fm½pm¾qm¿|mÀ}mÁ	mÂ	mÃ	mÄ	mÅ	 mÆ	¡mÇvÈvÉ%vÊ&vË-vÌ.vÍ8vÎ9vÏ@vÐAvÑRvÒSvÓ]vÔ^vÕevÖfv×pvØqvÙ|vÚ}vÛ	vÜ	vÝ	vÞ	vß	 và	¡vá
 
°
0
1
%2
&3
'4
(5
-6
.7
88
99
:10
;11
@12
A13
R14
S15
]16
^17
_18
`19
e20
f21
p22
q23
r24
s25
|26
}27
28
29
30
31
32
33
34
35
 36
¡37
Ì
0
1
%2
&3
-4
.5
86
97
@8
A9
R10
S11
]12
^13
e14
f15
p16
q17
|18
}19
20
21
22
23
 24
¡25
²
«non_trainable_variables
regularization_losses
 ¬layer_regularization_losses
­layer_metrics
®metrics
¯layers
	variables
trainable_variables
 
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
²
°non_trainable_variables
 regularization_losses
 ±layer_regularization_losses
²layer_metrics
³metrics
´layers
!	variables
"trainable_variables
 
ec
VARIABLE_VALUEbatchNormalization_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatchNormalization_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE batchNormalization_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE$batchNormalization_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1
'2
(3

%0
&1
²
µnon_trainable_variables
)regularization_losses
 ¶layer_regularization_losses
·layer_metrics
¸metrics
¹layers
*	variables
+trainable_variables
ZX
VARIABLE_VALUEconv_1b/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_1b/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
²
ºnon_trainable_variables
/regularization_losses
 »layer_regularization_losses
¼layer_metrics
½metrics
¾layers
0	variables
1trainable_variables
 
 
 
²
¿non_trainable_variables
3regularization_losses
 Àlayer_regularization_losses
Álayer_metrics
Âmetrics
Ãlayers
4	variables
5trainable_variables
 
fd
VARIABLE_VALUEbatchNormalization_1b/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatchNormalization_1b/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batchNormalization_1b/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batchNormalization_1b/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

80
91
:2
;3

80
91
²
Änon_trainable_variables
<regularization_losses
 Ålayer_regularization_losses
Ælayer_metrics
Çmetrics
Èlayers
=	variables
>trainable_variables
YW
VARIABLE_VALUEconv_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
²
Énon_trainable_variables
Bregularization_losses
 Êlayer_regularization_losses
Ëlayer_metrics
Ìmetrics
Ílayers
C	variables
Dtrainable_variables
 
 
 
²
Înon_trainable_variables
Fregularization_losses
 Ïlayer_regularization_losses
Ðlayer_metrics
Ñmetrics
Òlayers
G	variables
Htrainable_variables
 
 
 
²
Ónon_trainable_variables
Jregularization_losses
 Ôlayer_regularization_losses
Õlayer_metrics
Ömetrics
×layers
K	variables
Ltrainable_variables
 
 
 
²
Ønon_trainable_variables
Nregularization_losses
 Ùlayer_regularization_losses
Úlayer_metrics
Ûmetrics
Ülayers
O	variables
Ptrainable_variables
ZX
VARIABLE_VALUEconv_2b/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_2b/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
²
Ýnon_trainable_variables
Tregularization_losses
 Þlayer_regularization_losses
ßlayer_metrics
àmetrics
álayers
U	variables
Vtrainable_variables
 
 
 
²
ânon_trainable_variables
Xregularization_losses
 ãlayer_regularization_losses
älayer_metrics
åmetrics
ælayers
Y	variables
Ztrainable_variables
 
fd
VARIABLE_VALUEbatchNormalization_2b/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatchNormalization_2b/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batchNormalization_2b/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batchNormalization_2b/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1
_2
`3

]0
^1
²
çnon_trainable_variables
aregularization_losses
 èlayer_regularization_losses
élayer_metrics
êmetrics
ëlayers
b	variables
ctrainable_variables
YW
VARIABLE_VALUEconv_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
²
ìnon_trainable_variables
gregularization_losses
 ílayer_regularization_losses
îlayer_metrics
ïmetrics
ðlayers
h	variables
itrainable_variables
 
 
 
²
ñnon_trainable_variables
kregularization_losses
 òlayer_regularization_losses
ólayer_metrics
ômetrics
õlayers
l	variables
mtrainable_variables
 
ec
VARIABLE_VALUEbatchNormalization_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatchNormalization_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE batchNormalization_3/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE$batchNormalization_3/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

p0
q1
r2
s3

p0
q1
²
önon_trainable_variables
tregularization_losses
 ÷layer_regularization_losses
ølayer_metrics
ùmetrics
úlayers
u	variables
vtrainable_variables
 
 
 
²
ûnon_trainable_variables
xregularization_losses
 ülayer_regularization_losses
ýlayer_metrics
þmetrics
ÿlayers
y	variables
ztrainable_variables
YW
VARIABLE_VALUEconv_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

|0
}1

|0
}1
³
non_trainable_variables
~regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
 
 
 
µ
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
 
fd
VARIABLE_VALUEbatchNormalization_4/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatchNormalization_4/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE batchNormalization_4/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$batchNormalization_4/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
0
1
2
3

0
1
µ
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
 
 
 
µ
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
 
fd
VARIABLE_VALUEbatchNormalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatchNormalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE batchNormalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$batchNormalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
0
1
2
3

0
1
µ
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
 
 
 
µ
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
_]
VARIABLE_VALUEdense_final/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdense_final/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
¡1

 0
¡1
µ
non_trainable_variables
¢regularization_losses
 layer_regularization_losses
 layer_metrics
¡metrics
¢layers
£	variables
¤trainable_variables
JH
VARIABLE_VALUEAdamax/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdamax/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdamax/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdamax/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEAdamax/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
Z
'0
(1
:2
;3
_4
`5
r6
s7
8
9
10
11
 
 

£0
¤1
®
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
 
 
 
 
 

'0
(1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

:0
;1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

_0
`1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

r0
s1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

¥total

¦count
§	variables
¨	keras_api
I

©total

ªcount
«
_fn_kwargs
¬	variables
­	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

¥0
¦1

§	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

©0
ª1

¬	variables
~|
VARIABLE_VALUEAdamax/conv_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adamax/batchNormalization_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamax/conv_1b/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/conv_1b/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adamax/batchNormalization_1b/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_1b/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamax/conv_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamax/conv_2b/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/conv_2b/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adamax/batchNormalization_2b/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_2b/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamax/conv_3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv_3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_3/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adamax/batchNormalization_3/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamax/conv_4/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv_4/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_4/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adamax/batchNormalization_4/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_5/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adamax/batchNormalization_5/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdamax/dense_final/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdamax/dense_final/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamax/conv_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adamax/batchNormalization_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamax/conv_1b/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/conv_1b/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adamax/batchNormalization_1b/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_1b/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamax/conv_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamax/conv_2b/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/conv_2b/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adamax/batchNormalization_2b/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_2b/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamax/conv_3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv_3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_3/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adamax/batchNormalization_3/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamax/conv_4/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv_4/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_4/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adamax/batchNormalization_4/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adamax/batchNormalization_5/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adamax/batchNormalization_5/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdamax/dense_final/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdamax/dense_final/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv_1_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ``
Ù

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv_1_inputconv_1/kernelconv_1/biasbatchNormalization_1/gammabatchNormalization_1/beta batchNormalization_1/moving_mean$batchNormalization_1/moving_varianceconv_1b/kernelconv_1b/biasbatchNormalization_1b/gammabatchNormalization_1b/beta!batchNormalization_1b/moving_mean%batchNormalization_1b/moving_varianceconv_2/kernelconv_2/biasconv_2b/kernelconv_2b/biasbatchNormalization_2b/gammabatchNormalization_2b/beta!batchNormalization_2b/moving_mean%batchNormalization_2b/moving_varianceconv_3/kernelconv_3/biasbatchNormalization_3/gammabatchNormalization_3/beta batchNormalization_3/moving_mean$batchNormalization_3/moving_varianceconv_4/kernelconv_4/biasbatchNormalization_4/gammabatchNormalization_4/beta batchNormalization_4/moving_mean$batchNormalization_4/moving_variance$batchNormalization_5/moving_variancebatchNormalization_5/gamma batchNormalization_5/moving_meanbatchNormalization_5/betadense_final/kerneldense_final/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_50906
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp.batchNormalization_1/gamma/Read/ReadVariableOp-batchNormalization_1/beta/Read/ReadVariableOp4batchNormalization_1/moving_mean/Read/ReadVariableOp8batchNormalization_1/moving_variance/Read/ReadVariableOp"conv_1b/kernel/Read/ReadVariableOp conv_1b/bias/Read/ReadVariableOp/batchNormalization_1b/gamma/Read/ReadVariableOp.batchNormalization_1b/beta/Read/ReadVariableOp5batchNormalization_1b/moving_mean/Read/ReadVariableOp9batchNormalization_1b/moving_variance/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp"conv_2b/kernel/Read/ReadVariableOp conv_2b/bias/Read/ReadVariableOp/batchNormalization_2b/gamma/Read/ReadVariableOp.batchNormalization_2b/beta/Read/ReadVariableOp5batchNormalization_2b/moving_mean/Read/ReadVariableOp9batchNormalization_2b/moving_variance/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp.batchNormalization_3/gamma/Read/ReadVariableOp-batchNormalization_3/beta/Read/ReadVariableOp4batchNormalization_3/moving_mean/Read/ReadVariableOp8batchNormalization_3/moving_variance/Read/ReadVariableOp!conv_4/kernel/Read/ReadVariableOpconv_4/bias/Read/ReadVariableOp.batchNormalization_4/gamma/Read/ReadVariableOp-batchNormalization_4/beta/Read/ReadVariableOp4batchNormalization_4/moving_mean/Read/ReadVariableOp8batchNormalization_4/moving_variance/Read/ReadVariableOp.batchNormalization_5/gamma/Read/ReadVariableOp-batchNormalization_5/beta/Read/ReadVariableOp4batchNormalization_5/moving_mean/Read/ReadVariableOp8batchNormalization_5/moving_variance/Read/ReadVariableOp&dense_final/kernel/Read/ReadVariableOp$dense_final/bias/Read/ReadVariableOpAdamax/iter/Read/ReadVariableOp!Adamax/beta_1/Read/ReadVariableOp!Adamax/beta_2/Read/ReadVariableOp Adamax/decay/Read/ReadVariableOp(Adamax/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adamax/conv_1/kernel/m/Read/ReadVariableOp(Adamax/conv_1/bias/m/Read/ReadVariableOp7Adamax/batchNormalization_1/gamma/m/Read/ReadVariableOp6Adamax/batchNormalization_1/beta/m/Read/ReadVariableOp+Adamax/conv_1b/kernel/m/Read/ReadVariableOp)Adamax/conv_1b/bias/m/Read/ReadVariableOp8Adamax/batchNormalization_1b/gamma/m/Read/ReadVariableOp7Adamax/batchNormalization_1b/beta/m/Read/ReadVariableOp*Adamax/conv_2/kernel/m/Read/ReadVariableOp(Adamax/conv_2/bias/m/Read/ReadVariableOp+Adamax/conv_2b/kernel/m/Read/ReadVariableOp)Adamax/conv_2b/bias/m/Read/ReadVariableOp8Adamax/batchNormalization_2b/gamma/m/Read/ReadVariableOp7Adamax/batchNormalization_2b/beta/m/Read/ReadVariableOp*Adamax/conv_3/kernel/m/Read/ReadVariableOp(Adamax/conv_3/bias/m/Read/ReadVariableOp7Adamax/batchNormalization_3/gamma/m/Read/ReadVariableOp6Adamax/batchNormalization_3/beta/m/Read/ReadVariableOp*Adamax/conv_4/kernel/m/Read/ReadVariableOp(Adamax/conv_4/bias/m/Read/ReadVariableOp7Adamax/batchNormalization_4/gamma/m/Read/ReadVariableOp6Adamax/batchNormalization_4/beta/m/Read/ReadVariableOp7Adamax/batchNormalization_5/gamma/m/Read/ReadVariableOp6Adamax/batchNormalization_5/beta/m/Read/ReadVariableOp/Adamax/dense_final/kernel/m/Read/ReadVariableOp-Adamax/dense_final/bias/m/Read/ReadVariableOp*Adamax/conv_1/kernel/v/Read/ReadVariableOp(Adamax/conv_1/bias/v/Read/ReadVariableOp7Adamax/batchNormalization_1/gamma/v/Read/ReadVariableOp6Adamax/batchNormalization_1/beta/v/Read/ReadVariableOp+Adamax/conv_1b/kernel/v/Read/ReadVariableOp)Adamax/conv_1b/bias/v/Read/ReadVariableOp8Adamax/batchNormalization_1b/gamma/v/Read/ReadVariableOp7Adamax/batchNormalization_1b/beta/v/Read/ReadVariableOp*Adamax/conv_2/kernel/v/Read/ReadVariableOp(Adamax/conv_2/bias/v/Read/ReadVariableOp+Adamax/conv_2b/kernel/v/Read/ReadVariableOp)Adamax/conv_2b/bias/v/Read/ReadVariableOp8Adamax/batchNormalization_2b/gamma/v/Read/ReadVariableOp7Adamax/batchNormalization_2b/beta/v/Read/ReadVariableOp*Adamax/conv_3/kernel/v/Read/ReadVariableOp(Adamax/conv_3/bias/v/Read/ReadVariableOp7Adamax/batchNormalization_3/gamma/v/Read/ReadVariableOp6Adamax/batchNormalization_3/beta/v/Read/ReadVariableOp*Adamax/conv_4/kernel/v/Read/ReadVariableOp(Adamax/conv_4/bias/v/Read/ReadVariableOp7Adamax/batchNormalization_4/gamma/v/Read/ReadVariableOp6Adamax/batchNormalization_4/beta/v/Read/ReadVariableOp7Adamax/batchNormalization_5/gamma/v/Read/ReadVariableOp6Adamax/batchNormalization_5/beta/v/Read/ReadVariableOp/Adamax/dense_final/kernel/v/Read/ReadVariableOp-Adamax/dense_final/bias/v/Read/ReadVariableOpConst*p
Tini
g2e	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_52866
Ò
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasbatchNormalization_1/gammabatchNormalization_1/beta batchNormalization_1/moving_mean$batchNormalization_1/moving_varianceconv_1b/kernelconv_1b/biasbatchNormalization_1b/gammabatchNormalization_1b/beta!batchNormalization_1b/moving_mean%batchNormalization_1b/moving_varianceconv_2/kernelconv_2/biasconv_2b/kernelconv_2b/biasbatchNormalization_2b/gammabatchNormalization_2b/beta!batchNormalization_2b/moving_mean%batchNormalization_2b/moving_varianceconv_3/kernelconv_3/biasbatchNormalization_3/gammabatchNormalization_3/beta batchNormalization_3/moving_mean$batchNormalization_3/moving_varianceconv_4/kernelconv_4/biasbatchNormalization_4/gammabatchNormalization_4/beta batchNormalization_4/moving_mean$batchNormalization_4/moving_variancebatchNormalization_5/gammabatchNormalization_5/beta batchNormalization_5/moving_mean$batchNormalization_5/moving_variancedense_final/kerneldense_final/biasAdamax/iterAdamax/beta_1Adamax/beta_2Adamax/decayAdamax/learning_ratetotalcounttotal_1count_1Adamax/conv_1/kernel/mAdamax/conv_1/bias/m#Adamax/batchNormalization_1/gamma/m"Adamax/batchNormalization_1/beta/mAdamax/conv_1b/kernel/mAdamax/conv_1b/bias/m$Adamax/batchNormalization_1b/gamma/m#Adamax/batchNormalization_1b/beta/mAdamax/conv_2/kernel/mAdamax/conv_2/bias/mAdamax/conv_2b/kernel/mAdamax/conv_2b/bias/m$Adamax/batchNormalization_2b/gamma/m#Adamax/batchNormalization_2b/beta/mAdamax/conv_3/kernel/mAdamax/conv_3/bias/m#Adamax/batchNormalization_3/gamma/m"Adamax/batchNormalization_3/beta/mAdamax/conv_4/kernel/mAdamax/conv_4/bias/m#Adamax/batchNormalization_4/gamma/m"Adamax/batchNormalization_4/beta/m#Adamax/batchNormalization_5/gamma/m"Adamax/batchNormalization_5/beta/mAdamax/dense_final/kernel/mAdamax/dense_final/bias/mAdamax/conv_1/kernel/vAdamax/conv_1/bias/v#Adamax/batchNormalization_1/gamma/v"Adamax/batchNormalization_1/beta/vAdamax/conv_1b/kernel/vAdamax/conv_1b/bias/v$Adamax/batchNormalization_1b/gamma/v#Adamax/batchNormalization_1b/beta/vAdamax/conv_2/kernel/vAdamax/conv_2/bias/vAdamax/conv_2b/kernel/vAdamax/conv_2b/bias/v$Adamax/batchNormalization_2b/gamma/v#Adamax/batchNormalization_2b/beta/vAdamax/conv_3/kernel/vAdamax/conv_3/bias/v#Adamax/batchNormalization_3/gamma/v"Adamax/batchNormalization_3/beta/vAdamax/conv_4/kernel/vAdamax/conv_4/bias/v#Adamax/batchNormalization_4/gamma/v"Adamax/batchNormalization_4/beta/v#Adamax/batchNormalization_5/gamma/v"Adamax/batchNormalization_5/beta/vAdamax/dense_final/kernel/vAdamax/dense_final/bias/v*o
Tinh
f2d*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_53173Â®
¹
©
G__inference_sequential_5_layer_call_and_return_conditional_losses_50693
conv_1_input&
conv_1_50578:
conv_1_50580:(
batchnormalization_1_50583:(
batchnormalization_1_50585:(
batchnormalization_1_50587:(
batchnormalization_1_50589:'
conv_1b_50592: 
conv_1b_50594: )
batchnormalization_1b_50598: )
batchnormalization_1b_50600: )
batchnormalization_1b_50602: )
batchnormalization_1b_50604: &
conv_2_50607: @
conv_2_50609:@'
conv_2b_50615:@@
conv_2b_50617:@)
batchnormalization_2b_50621:@)
batchnormalization_2b_50623:@)
batchnormalization_2b_50625:@)
batchnormalization_2b_50627:@'
conv_3_50630:@
conv_3_50632:	)
batchnormalization_3_50636:	)
batchnormalization_3_50638:	)
batchnormalization_3_50640:	)
batchnormalization_3_50642:	(
conv_4_50646:
conv_4_50648:	)
batchnormalization_4_50652:	)
batchnormalization_4_50654:	)
batchnormalization_4_50656:	)
batchnormalization_4_50658:	)
batchnormalization_5_50662:	)
batchnormalization_5_50664:	)
batchnormalization_5_50666:	)
batchnormalization_5_50668:	$
dense_final_50672:	
dense_final_50674:
identity

identity_1¢,batchNormalization_1/StatefulPartitionedCall¢-batchNormalization_1b/StatefulPartitionedCall¢-batchNormalization_2b/StatefulPartitionedCall¢,batchNormalization_3/StatefulPartitionedCall¢,batchNormalization_4/StatefulPartitionedCall¢,batchNormalization_5/StatefulPartitionedCall¢conv_1/StatefulPartitionedCall¢conv_1b/StatefulPartitionedCall¢conv_2/StatefulPartitionedCall¢conv_2b/StatefulPartitionedCall¢conv_3/StatefulPartitionedCall¢conv_4/StatefulPartitionedCall¢#dense_final/StatefulPartitionedCall¢4dense_final/kernel/Regularizer/Square/ReadVariableOp
conv_1/StatefulPartitionedCallStatefulPartitionedCallconv_1_inputconv_1_50578conv_1_50580*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_493722 
conv_1/StatefulPartitionedCallµ
,batchNormalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnormalization_1_50583batchnormalization_1_50585batchnormalization_1_50587batchnormalization_1_50589*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_493952.
,batchNormalization_1/StatefulPartitionedCallÆ
conv_1b/StatefulPartitionedCallStatefulPartitionedCall5batchNormalization_1/StatefulPartitionedCall:output:0conv_1b_50592conv_1b_50594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_1b_layer_call_and_return_conditional_losses_494162!
conv_1b/StatefulPartitionedCall
maxPool_1b/PartitionedCallPartitionedCall(conv_1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_494262
maxPool_1b/PartitionedCall¸
-batchNormalization_1b/StatefulPartitionedCallStatefulPartitionedCall#maxPool_1b/PartitionedCall:output:0batchnormalization_1b_50598batchnormalization_1b_50600batchnormalization_1b_50602batchnormalization_1b_50604*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_494452/
-batchNormalization_1b/StatefulPartitionedCallÂ
conv_2/StatefulPartitionedCallStatefulPartitionedCall6batchNormalization_1b/StatefulPartitionedCall:output:0conv_2_50607conv_2_50609*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_494662 
conv_2/StatefulPartitionedCall
maxPool_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_2_layer_call_and_return_conditional_losses_494762
maxPool_2/PartitionedCallý
dropout_2/PartitionedCallPartitionedCall"maxPool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_494832
dropout_2/PartitionedCall÷
noise_1/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_1_layer_call_and_return_conditional_losses_494892
noise_1/PartitionedCall±
conv_2b/StatefulPartitionedCallStatefulPartitionedCall noise_1/PartitionedCall:output:0conv_2b_50615conv_2b_50617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_2b_layer_call_and_return_conditional_losses_495022!
conv_2b/StatefulPartitionedCall
maxPool_2b/PartitionedCallPartitionedCall(conv_2b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_495122
maxPool_2b/PartitionedCall¸
-batchNormalization_2b/StatefulPartitionedCallStatefulPartitionedCall#maxPool_2b/PartitionedCall:output:0batchnormalization_2b_50621batchnormalization_2b_50623batchnormalization_2b_50625batchnormalization_2b_50627*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_495312/
-batchNormalization_2b/StatefulPartitionedCallÃ
conv_3/StatefulPartitionedCallStatefulPartitionedCall6batchNormalization_2b/StatefulPartitionedCall:output:0conv_3_50630conv_3_50632*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_495522 
conv_3/StatefulPartitionedCall
maxPool_3/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_3_layer_call_and_return_conditional_losses_495622
maxPool_3/PartitionedCall±
,batchNormalization_3/StatefulPartitionedCallStatefulPartitionedCall"maxPool_3/PartitionedCall:output:0batchnormalization_3_50636batchnormalization_3_50638batchnormalization_3_50640batchnormalization_3_50642*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_495812.
,batchNormalization_3/StatefulPartitionedCall
noise_3/PartitionedCallPartitionedCall5batchNormalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_3_layer_call_and_return_conditional_losses_495952
noise_3/PartitionedCall­
conv_4/StatefulPartitionedCallStatefulPartitionedCall noise_3/PartitionedCall:output:0conv_4_50646conv_4_50648*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_496082 
conv_4/StatefulPartitionedCall
maxPool_4/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_4_layer_call_and_return_conditional_losses_496182
maxPool_4/PartitionedCall±
,batchNormalization_4/StatefulPartitionedCallStatefulPartitionedCall"maxPool_4/PartitionedCall:output:0batchnormalization_4_50652batchnormalization_4_50654batchnormalization_4_50656batchnormalization_4_50658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_496372.
,batchNormalization_4/StatefulPartitionedCall
globAvgPool_5/PartitionedCallPartitionedCall5batchNormalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_496522
globAvgPool_5/PartitionedCall­
,batchNormalization_5/StatefulPartitionedCallStatefulPartitionedCall&globAvgPool_5/PartitionedCall:output:0batchnormalization_5_50662batchnormalization_5_50664batchnormalization_5_50666batchnormalization_5_50668*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_492032.
,batchNormalization_5/StatefulPartitionedCall
dropout_6/PartitionedCallPartitionedCall5batchNormalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_496682
dropout_6/PartitionedCall¿
#dense_final/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_final_50672dense_final_50674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_496872%
#dense_final/StatefulPartitionedCall
/dense_final/ActivityRegularizer/PartitionedCallPartitionedCall,dense_final/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *;
f6R4
2__inference_dense_final_activity_regularizer_4935421
/dense_final/ActivityRegularizer/PartitionedCallª
%dense_final/ActivityRegularizer/ShapeShape,dense_final/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2'
%dense_final/ActivityRegularizer/Shape´
3dense_final/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3dense_final/ActivityRegularizer/strided_slice/stack¸
5dense_final/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_1¸
5dense_final/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_2¢
-dense_final/ActivityRegularizer/strided_sliceStridedSlice.dense_final/ActivityRegularizer/Shape:output:0<dense_final/ActivityRegularizer/strided_slice/stack:output:0>dense_final/ActivityRegularizer/strided_slice/stack_1:output:0>dense_final/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-dense_final/ActivityRegularizer/strided_slice¼
$dense_final/ActivityRegularizer/CastCast6dense_final/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$dense_final/ActivityRegularizer/Castâ
'dense_final/ActivityRegularizer/truedivRealDiv8dense_final/ActivityRegularizer/PartitionedCall:output:0(dense_final/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_final/ActivityRegularizer/truediv¿
4dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_final_50672*
_output_shapes
:	*
dtype026
4dense_final/kernel/Regularizer/Square/ReadVariableOpÀ
%dense_final/kernel/Regularizer/SquareSquare<dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%dense_final/kernel/Regularizer/Square
$dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$dense_final/kernel/Regularizer/ConstÊ
"dense_final/kernel/Regularizer/SumSum)dense_final/kernel/Regularizer/Square:y:0-dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/Sum
$dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *-²=2&
$dense_final/kernel/Regularizer/mul/xÌ
"dense_final/kernel/Regularizer/mulMul-dense_final/kernel/Regularizer/mul/x:output:0+dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/mul
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityy

Identity_1Identity+dense_final/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1
NoOpNoOp-^batchNormalization_1/StatefulPartitionedCall.^batchNormalization_1b/StatefulPartitionedCall.^batchNormalization_2b/StatefulPartitionedCall-^batchNormalization_3/StatefulPartitionedCall-^batchNormalization_4/StatefulPartitionedCall-^batchNormalization_5/StatefulPartitionedCall^conv_1/StatefulPartitionedCall ^conv_1b/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^conv_2b/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall$^dense_final/StatefulPartitionedCall5^dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,batchNormalization_1/StatefulPartitionedCall,batchNormalization_1/StatefulPartitionedCall2^
-batchNormalization_1b/StatefulPartitionedCall-batchNormalization_1b/StatefulPartitionedCall2^
-batchNormalization_2b/StatefulPartitionedCall-batchNormalization_2b/StatefulPartitionedCall2\
,batchNormalization_3/StatefulPartitionedCall,batchNormalization_3/StatefulPartitionedCall2\
,batchNormalization_4/StatefulPartitionedCall,batchNormalization_4/StatefulPartitionedCall2\
,batchNormalization_5/StatefulPartitionedCall,batchNormalization_5/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2B
conv_1b/StatefulPartitionedCallconv_1b/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2B
conv_2b/StatefulPartitionedCallconv_2b/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2l
4dense_final/kernel/Regularizer/Square/ReadVariableOp4dense_final/kernel/Regularizer/Square/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameconv_1_input
Ü
Â
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52122

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
b
)__inference_dropout_2_layer_call_fn_51821

inputs
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_500802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ü
Â
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_49872

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

^
B__inference_noise_1_layer_call_and_return_conditional_losses_49489

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½	
Ï
4__inference_batchNormalization_1_layer_call_fn_51564

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_484812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
ú
A__inference_conv_1_layer_call_and_return_conditional_losses_51457

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ``: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
Î
F
*__inference_maxPool_1b_layer_call_fn_51625

inputs
identityé
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_485502
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

²
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_49203

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¿
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51922

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â

P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_48755

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï
ü
A__inference_conv_3_layer_call_and_return_conditional_losses_52021

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51940

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
õ
Ï
4__inference_batchNormalization_1_layer_call_fn_51590

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_501902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
è
û
B__inference_conv_2b_layer_call_and_return_conditional_losses_51857

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
³
¯
G__inference_sequential_5_layer_call_and_return_conditional_losses_50413

inputs&
conv_1_50298:
conv_1_50300:(
batchnormalization_1_50303:(
batchnormalization_1_50305:(
batchnormalization_1_50307:(
batchnormalization_1_50309:'
conv_1b_50312: 
conv_1b_50314: )
batchnormalization_1b_50318: )
batchnormalization_1b_50320: )
batchnormalization_1b_50322: )
batchnormalization_1b_50324: &
conv_2_50327: @
conv_2_50329:@'
conv_2b_50335:@@
conv_2b_50337:@)
batchnormalization_2b_50341:@)
batchnormalization_2b_50343:@)
batchnormalization_2b_50345:@)
batchnormalization_2b_50347:@'
conv_3_50350:@
conv_3_50352:	)
batchnormalization_3_50356:	)
batchnormalization_3_50358:	)
batchnormalization_3_50360:	)
batchnormalization_3_50362:	(
conv_4_50366:
conv_4_50368:	)
batchnormalization_4_50372:	)
batchnormalization_4_50374:	)
batchnormalization_4_50376:	)
batchnormalization_4_50378:	)
batchnormalization_5_50382:	)
batchnormalization_5_50384:	)
batchnormalization_5_50386:	)
batchnormalization_5_50388:	$
dense_final_50392:	
dense_final_50394:
identity

identity_1¢,batchNormalization_1/StatefulPartitionedCall¢-batchNormalization_1b/StatefulPartitionedCall¢-batchNormalization_2b/StatefulPartitionedCall¢,batchNormalization_3/StatefulPartitionedCall¢,batchNormalization_4/StatefulPartitionedCall¢,batchNormalization_5/StatefulPartitionedCall¢conv_1/StatefulPartitionedCall¢conv_1b/StatefulPartitionedCall¢conv_2/StatefulPartitionedCall¢conv_2b/StatefulPartitionedCall¢conv_3/StatefulPartitionedCall¢conv_4/StatefulPartitionedCall¢#dense_final/StatefulPartitionedCall¢4dense_final/kernel/Regularizer/Square/ReadVariableOp¢!dropout_2/StatefulPartitionedCall¢!dropout_6/StatefulPartitionedCall¢noise_1/StatefulPartitionedCall¢noise_3/StatefulPartitionedCall
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_50298conv_1_50300*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_493722 
conv_1/StatefulPartitionedCall³
,batchNormalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnormalization_1_50303batchnormalization_1_50305batchnormalization_1_50307batchnormalization_1_50309*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_501902.
,batchNormalization_1/StatefulPartitionedCallÆ
conv_1b/StatefulPartitionedCallStatefulPartitionedCall5batchNormalization_1/StatefulPartitionedCall:output:0conv_1b_50312conv_1b_50314*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_1b_layer_call_and_return_conditional_losses_494162!
conv_1b/StatefulPartitionedCall
maxPool_1b/PartitionedCallPartitionedCall(conv_1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_494262
maxPool_1b/PartitionedCall¶
-batchNormalization_1b/StatefulPartitionedCallStatefulPartitionedCall#maxPool_1b/PartitionedCall:output:0batchnormalization_1b_50318batchnormalization_1b_50320batchnormalization_1b_50322batchnormalization_1b_50324*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_501312/
-batchNormalization_1b/StatefulPartitionedCallÂ
conv_2/StatefulPartitionedCallStatefulPartitionedCall6batchNormalization_1b/StatefulPartitionedCall:output:0conv_2_50327conv_2_50329*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_494662 
conv_2/StatefulPartitionedCall
maxPool_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_2_layer_call_and_return_conditional_losses_494762
maxPool_2/PartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall"maxPool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_500802#
!dropout_2/StatefulPartitionedCall»
noise_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_1_layer_call_and_return_conditional_losses_500572!
noise_1/StatefulPartitionedCall¹
conv_2b/StatefulPartitionedCallStatefulPartitionedCall(noise_1/StatefulPartitionedCall:output:0conv_2b_50335conv_2b_50337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_2b_layer_call_and_return_conditional_losses_495022!
conv_2b/StatefulPartitionedCall
maxPool_2b/PartitionedCallPartitionedCall(conv_2b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_495122
maxPool_2b/PartitionedCall¶
-batchNormalization_2b/StatefulPartitionedCallStatefulPartitionedCall#maxPool_2b/PartitionedCall:output:0batchnormalization_2b_50341batchnormalization_2b_50343batchnormalization_2b_50345batchnormalization_2b_50347*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_500122/
-batchNormalization_2b/StatefulPartitionedCallÃ
conv_3/StatefulPartitionedCallStatefulPartitionedCall6batchNormalization_2b/StatefulPartitionedCall:output:0conv_3_50350conv_3_50352*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_495522 
conv_3/StatefulPartitionedCall
maxPool_3/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_3_layer_call_and_return_conditional_losses_495622
maxPool_3/PartitionedCall¯
,batchNormalization_3/StatefulPartitionedCallStatefulPartitionedCall"maxPool_3/PartitionedCall:output:0batchnormalization_3_50356batchnormalization_3_50358batchnormalization_3_50360batchnormalization_3_50362*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_499532.
,batchNormalization_3/StatefulPartitionedCallÅ
noise_3/StatefulPartitionedCallStatefulPartitionedCall5batchNormalization_3/StatefulPartitionedCall:output:0 ^noise_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_3_layer_call_and_return_conditional_losses_499172!
noise_3/StatefulPartitionedCallµ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(noise_3/StatefulPartitionedCall:output:0conv_4_50366conv_4_50368*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_496082 
conv_4/StatefulPartitionedCall
maxPool_4/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_4_layer_call_and_return_conditional_losses_496182
maxPool_4/PartitionedCall¯
,batchNormalization_4/StatefulPartitionedCallStatefulPartitionedCall"maxPool_4/PartitionedCall:output:0batchnormalization_4_50372batchnormalization_4_50374batchnormalization_4_50376batchnormalization_4_50378*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_498722.
,batchNormalization_4/StatefulPartitionedCall
globAvgPool_5/PartitionedCallPartitionedCall5batchNormalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_496522
globAvgPool_5/PartitionedCall«
,batchNormalization_5/StatefulPartitionedCallStatefulPartitionedCall&globAvgPool_5/PartitionedCall:output:0batchnormalization_5_50382batchnormalization_5_50384batchnormalization_5_50386batchnormalization_5_50388*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_492632.
,batchNormalization_5/StatefulPartitionedCallÃ
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall5batchNormalization_5/StatefulPartitionedCall:output:0 ^noise_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_498312#
!dropout_6/StatefulPartitionedCallÇ
#dense_final/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_final_50392dense_final_50394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_496872%
#dense_final/StatefulPartitionedCall
/dense_final/ActivityRegularizer/PartitionedCallPartitionedCall,dense_final/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *;
f6R4
2__inference_dense_final_activity_regularizer_4935421
/dense_final/ActivityRegularizer/PartitionedCallª
%dense_final/ActivityRegularizer/ShapeShape,dense_final/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2'
%dense_final/ActivityRegularizer/Shape´
3dense_final/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3dense_final/ActivityRegularizer/strided_slice/stack¸
5dense_final/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_1¸
5dense_final/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_2¢
-dense_final/ActivityRegularizer/strided_sliceStridedSlice.dense_final/ActivityRegularizer/Shape:output:0<dense_final/ActivityRegularizer/strided_slice/stack:output:0>dense_final/ActivityRegularizer/strided_slice/stack_1:output:0>dense_final/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-dense_final/ActivityRegularizer/strided_slice¼
$dense_final/ActivityRegularizer/CastCast6dense_final/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$dense_final/ActivityRegularizer/Castâ
'dense_final/ActivityRegularizer/truedivRealDiv8dense_final/ActivityRegularizer/PartitionedCall:output:0(dense_final/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_final/ActivityRegularizer/truediv¿
4dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_final_50392*
_output_shapes
:	*
dtype026
4dense_final/kernel/Regularizer/Square/ReadVariableOpÀ
%dense_final/kernel/Regularizer/SquareSquare<dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%dense_final/kernel/Regularizer/Square
$dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$dense_final/kernel/Regularizer/ConstÊ
"dense_final/kernel/Regularizer/SumSum)dense_final/kernel/Regularizer/Square:y:0-dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/Sum
$dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *-²=2&
$dense_final/kernel/Regularizer/mul/xÌ
"dense_final/kernel/Regularizer/mulMul-dense_final/kernel/Regularizer/mul/x:output:0+dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/mul
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityy

Identity_1Identity+dense_final/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1
NoOpNoOp-^batchNormalization_1/StatefulPartitionedCall.^batchNormalization_1b/StatefulPartitionedCall.^batchNormalization_2b/StatefulPartitionedCall-^batchNormalization_3/StatefulPartitionedCall-^batchNormalization_4/StatefulPartitionedCall-^batchNormalization_5/StatefulPartitionedCall^conv_1/StatefulPartitionedCall ^conv_1b/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^conv_2b/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall$^dense_final/StatefulPartitionedCall5^dense_final/kernel/Regularizer/Square/ReadVariableOp"^dropout_2/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall ^noise_1/StatefulPartitionedCall ^noise_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,batchNormalization_1/StatefulPartitionedCall,batchNormalization_1/StatefulPartitionedCall2^
-batchNormalization_1b/StatefulPartitionedCall-batchNormalization_1b/StatefulPartitionedCall2^
-batchNormalization_2b/StatefulPartitionedCall-batchNormalization_2b/StatefulPartitionedCall2\
,batchNormalization_3/StatefulPartitionedCall,batchNormalization_3/StatefulPartitionedCall2\
,batchNormalization_4/StatefulPartitionedCall,batchNormalization_4/StatefulPartitionedCall2\
,batchNormalization_5/StatefulPartitionedCall,batchNormalization_5/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2B
conv_1b/StatefulPartitionedCallconv_1b/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2B
conv_2b/StatefulPartitionedCallconv_2b/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2l
4dense_final/kernel/Regularizer/Square/ReadVariableOp4dense_final/kernel/Regularizer/Square/ReadVariableOp2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2B
noise_1/StatefulPartitionedCallnoise_1/StatefulPartitionedCall2B
noise_3/StatefulPartitionedCallnoise_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
÷
Ï
4__inference_batchNormalization_1_layer_call_fn_51577

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_493952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
Ø
d
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_52369

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¿
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51666

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ß
E
)__inference_maxPool_2_layer_call_fn_51794

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_2_layer_call_and_return_conditional_losses_494762
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ00@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
æ*
ì
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_52439

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

b
D__inference_dropout_2_layer_call_and_return_conditional_losses_51799

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Å
µ
G__inference_sequential_5_layer_call_and_return_conditional_losses_50811
conv_1_input&
conv_1_50696:
conv_1_50698:(
batchnormalization_1_50701:(
batchnormalization_1_50703:(
batchnormalization_1_50705:(
batchnormalization_1_50707:'
conv_1b_50710: 
conv_1b_50712: )
batchnormalization_1b_50716: )
batchnormalization_1b_50718: )
batchnormalization_1b_50720: )
batchnormalization_1b_50722: &
conv_2_50725: @
conv_2_50727:@'
conv_2b_50733:@@
conv_2b_50735:@)
batchnormalization_2b_50739:@)
batchnormalization_2b_50741:@)
batchnormalization_2b_50743:@)
batchnormalization_2b_50745:@'
conv_3_50748:@
conv_3_50750:	)
batchnormalization_3_50754:	)
batchnormalization_3_50756:	)
batchnormalization_3_50758:	)
batchnormalization_3_50760:	(
conv_4_50764:
conv_4_50766:	)
batchnormalization_4_50770:	)
batchnormalization_4_50772:	)
batchnormalization_4_50774:	)
batchnormalization_4_50776:	)
batchnormalization_5_50780:	)
batchnormalization_5_50782:	)
batchnormalization_5_50784:	)
batchnormalization_5_50786:	$
dense_final_50790:	
dense_final_50792:
identity

identity_1¢,batchNormalization_1/StatefulPartitionedCall¢-batchNormalization_1b/StatefulPartitionedCall¢-batchNormalization_2b/StatefulPartitionedCall¢,batchNormalization_3/StatefulPartitionedCall¢,batchNormalization_4/StatefulPartitionedCall¢,batchNormalization_5/StatefulPartitionedCall¢conv_1/StatefulPartitionedCall¢conv_1b/StatefulPartitionedCall¢conv_2/StatefulPartitionedCall¢conv_2b/StatefulPartitionedCall¢conv_3/StatefulPartitionedCall¢conv_4/StatefulPartitionedCall¢#dense_final/StatefulPartitionedCall¢4dense_final/kernel/Regularizer/Square/ReadVariableOp¢!dropout_2/StatefulPartitionedCall¢!dropout_6/StatefulPartitionedCall¢noise_1/StatefulPartitionedCall¢noise_3/StatefulPartitionedCall
conv_1/StatefulPartitionedCallStatefulPartitionedCallconv_1_inputconv_1_50696conv_1_50698*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_493722 
conv_1/StatefulPartitionedCall³
,batchNormalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnormalization_1_50701batchnormalization_1_50703batchnormalization_1_50705batchnormalization_1_50707*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_501902.
,batchNormalization_1/StatefulPartitionedCallÆ
conv_1b/StatefulPartitionedCallStatefulPartitionedCall5batchNormalization_1/StatefulPartitionedCall:output:0conv_1b_50710conv_1b_50712*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_1b_layer_call_and_return_conditional_losses_494162!
conv_1b/StatefulPartitionedCall
maxPool_1b/PartitionedCallPartitionedCall(conv_1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_494262
maxPool_1b/PartitionedCall¶
-batchNormalization_1b/StatefulPartitionedCallStatefulPartitionedCall#maxPool_1b/PartitionedCall:output:0batchnormalization_1b_50716batchnormalization_1b_50718batchnormalization_1b_50720batchnormalization_1b_50722*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_501312/
-batchNormalization_1b/StatefulPartitionedCallÂ
conv_2/StatefulPartitionedCallStatefulPartitionedCall6batchNormalization_1b/StatefulPartitionedCall:output:0conv_2_50725conv_2_50727*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_494662 
conv_2/StatefulPartitionedCall
maxPool_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_2_layer_call_and_return_conditional_losses_494762
maxPool_2/PartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall"maxPool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_500802#
!dropout_2/StatefulPartitionedCall»
noise_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_1_layer_call_and_return_conditional_losses_500572!
noise_1/StatefulPartitionedCall¹
conv_2b/StatefulPartitionedCallStatefulPartitionedCall(noise_1/StatefulPartitionedCall:output:0conv_2b_50733conv_2b_50735*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_2b_layer_call_and_return_conditional_losses_495022!
conv_2b/StatefulPartitionedCall
maxPool_2b/PartitionedCallPartitionedCall(conv_2b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_495122
maxPool_2b/PartitionedCall¶
-batchNormalization_2b/StatefulPartitionedCallStatefulPartitionedCall#maxPool_2b/PartitionedCall:output:0batchnormalization_2b_50739batchnormalization_2b_50741batchnormalization_2b_50743batchnormalization_2b_50745*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_500122/
-batchNormalization_2b/StatefulPartitionedCallÃ
conv_3/StatefulPartitionedCallStatefulPartitionedCall6batchNormalization_2b/StatefulPartitionedCall:output:0conv_3_50748conv_3_50750*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_495522 
conv_3/StatefulPartitionedCall
maxPool_3/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_3_layer_call_and_return_conditional_losses_495622
maxPool_3/PartitionedCall¯
,batchNormalization_3/StatefulPartitionedCallStatefulPartitionedCall"maxPool_3/PartitionedCall:output:0batchnormalization_3_50754batchnormalization_3_50756batchnormalization_3_50758batchnormalization_3_50760*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_499532.
,batchNormalization_3/StatefulPartitionedCallÅ
noise_3/StatefulPartitionedCallStatefulPartitionedCall5batchNormalization_3/StatefulPartitionedCall:output:0 ^noise_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_3_layer_call_and_return_conditional_losses_499172!
noise_3/StatefulPartitionedCallµ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(noise_3/StatefulPartitionedCall:output:0conv_4_50764conv_4_50766*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_496082 
conv_4/StatefulPartitionedCall
maxPool_4/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_4_layer_call_and_return_conditional_losses_496182
maxPool_4/PartitionedCall¯
,batchNormalization_4/StatefulPartitionedCallStatefulPartitionedCall"maxPool_4/PartitionedCall:output:0batchnormalization_4_50770batchnormalization_4_50772batchnormalization_4_50774batchnormalization_4_50776*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_498722.
,batchNormalization_4/StatefulPartitionedCall
globAvgPool_5/PartitionedCallPartitionedCall5batchNormalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_496522
globAvgPool_5/PartitionedCall«
,batchNormalization_5/StatefulPartitionedCallStatefulPartitionedCall&globAvgPool_5/PartitionedCall:output:0batchnormalization_5_50780batchnormalization_5_50782batchnormalization_5_50784batchnormalization_5_50786*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_492632.
,batchNormalization_5/StatefulPartitionedCallÃ
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall5batchNormalization_5/StatefulPartitionedCall:output:0 ^noise_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_498312#
!dropout_6/StatefulPartitionedCallÇ
#dense_final/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_final_50790dense_final_50792*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_496872%
#dense_final/StatefulPartitionedCall
/dense_final/ActivityRegularizer/PartitionedCallPartitionedCall,dense_final/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *;
f6R4
2__inference_dense_final_activity_regularizer_4935421
/dense_final/ActivityRegularizer/PartitionedCallª
%dense_final/ActivityRegularizer/ShapeShape,dense_final/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2'
%dense_final/ActivityRegularizer/Shape´
3dense_final/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3dense_final/ActivityRegularizer/strided_slice/stack¸
5dense_final/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_1¸
5dense_final/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_2¢
-dense_final/ActivityRegularizer/strided_sliceStridedSlice.dense_final/ActivityRegularizer/Shape:output:0<dense_final/ActivityRegularizer/strided_slice/stack:output:0>dense_final/ActivityRegularizer/strided_slice/stack_1:output:0>dense_final/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-dense_final/ActivityRegularizer/strided_slice¼
$dense_final/ActivityRegularizer/CastCast6dense_final/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$dense_final/ActivityRegularizer/Castâ
'dense_final/ActivityRegularizer/truedivRealDiv8dense_final/ActivityRegularizer/PartitionedCall:output:0(dense_final/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_final/ActivityRegularizer/truediv¿
4dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_final_50790*
_output_shapes
:	*
dtype026
4dense_final/kernel/Regularizer/Square/ReadVariableOpÀ
%dense_final/kernel/Regularizer/SquareSquare<dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%dense_final/kernel/Regularizer/Square
$dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$dense_final/kernel/Regularizer/ConstÊ
"dense_final/kernel/Regularizer/SumSum)dense_final/kernel/Regularizer/Square:y:0-dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/Sum
$dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *-²=2&
$dense_final/kernel/Regularizer/mul/xÌ
"dense_final/kernel/Regularizer/mulMul-dense_final/kernel/Regularizer/mul/x:output:0+dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/mul
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityy

Identity_1Identity+dense_final/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1
NoOpNoOp-^batchNormalization_1/StatefulPartitionedCall.^batchNormalization_1b/StatefulPartitionedCall.^batchNormalization_2b/StatefulPartitionedCall-^batchNormalization_3/StatefulPartitionedCall-^batchNormalization_4/StatefulPartitionedCall-^batchNormalization_5/StatefulPartitionedCall^conv_1/StatefulPartitionedCall ^conv_1b/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^conv_2b/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall$^dense_final/StatefulPartitionedCall5^dense_final/kernel/Regularizer/Square/ReadVariableOp"^dropout_2/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall ^noise_1/StatefulPartitionedCall ^noise_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,batchNormalization_1/StatefulPartitionedCall,batchNormalization_1/StatefulPartitionedCall2^
-batchNormalization_1b/StatefulPartitionedCall-batchNormalization_1b/StatefulPartitionedCall2^
-batchNormalization_2b/StatefulPartitionedCall-batchNormalization_2b/StatefulPartitionedCall2\
,batchNormalization_3/StatefulPartitionedCall,batchNormalization_3/StatefulPartitionedCall2\
,batchNormalization_4/StatefulPartitionedCall,batchNormalization_4/StatefulPartitionedCall2\
,batchNormalization_5/StatefulPartitionedCall,batchNormalization_5/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2B
conv_1b/StatefulPartitionedCallconv_1b/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2B
conv_2b/StatefulPartitionedCallconv_2b/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2l
4dense_final/kernel/Regularizer/Square/ReadVariableOp4dense_final/kernel/Regularizer/Square/ReadVariableOp2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2B
noise_1/StatefulPartitionedCallnoise_1/StatefulPartitionedCall2B
noise_3/StatefulPartitionedCallnoise_3/StatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameconv_1_input
±
	
,__inference_sequential_5_layer_call_fn_51446

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: *<
_read_only_resource_inputs
	
#$%&*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_504132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
÷
Ð
5__inference_batchNormalization_1b_layer_call_fn_51754

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_501312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
â

P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51904

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

	
#__inference_signature_wrapper_50906
conv_1_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallconv_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_484152
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameconv_1_input
Â
`
'__inference_noise_3_layer_call_fn_52199

inputs
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_3_layer_call_and_return_conditional_losses_499172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_52375

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

a
B__inference_noise_1_layer_call_and_return_conditional_losses_51836

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *>2
random_normal/stddev×
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed¡¶*
seed2â´2$
"random_normal/RandomStandardNormal³
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
á

O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_48437

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¾
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_48481

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¿
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_48629

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_49531

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¥
a
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_48550

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


&__inference_conv_2_layer_call_fn_51774

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_494662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ00 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
Ï
¯
F__inference_dense_final_layer_call_and_return_conditional_losses_49687

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢4dense_final/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
SoftmaxÌ
4dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype026
4dense_final/kernel/Regularizer/Square/ReadVariableOpÀ
%dense_final/kernel/Regularizer/SquareSquare<dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%dense_final/kernel/Regularizer/Square
$dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$dense_final/kernel/Regularizer/ConstÊ
"dense_final/kernel/Regularizer/SumSum)dense_final/kernel/Regularizer/Square:y:0-dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/Sum
$dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *-²=2&
$dense_final/kernel/Regularizer/mul/xÌ
"dense_final/kernel/Regularizer/mulMul-dense_final/kernel/Regularizer/mul/x:output:0+dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/mull
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¶
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4dense_final/kernel/Regularizer/Square/ReadVariableOp4dense_final/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
`
D__inference_maxPool_3_layer_call_and_return_conditional_losses_52040

inputs
identity
MaxPoolMaxPoolinputs*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á	
Ð
5__inference_batchNormalization_2b_layer_call_fn_51971

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_487552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ô
È
J__inference_dense_final_layer_call_and_return_all_conditional_losses_52509

inputs
unknown:	
	unknown_0:
identity

identity_1¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_496872
StatefulPartitionedCall¼
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *;
f6R4
2__inference_dense_final_activity_regularizer_493542
PartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityf

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
F
*__inference_maxPool_2b_layer_call_fn_51886

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_495122
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ó
ý
A__inference_conv_4_layer_call_and_return_conditional_losses_52210

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
`
D__inference_maxPool_3_layer_call_and_return_conditional_losses_49562

inputs
identity
MaxPoolMaxPoolinputs*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
£
G__inference_sequential_5_layer_call_and_return_conditional_losses_49709

inputs&
conv_1_49373:
conv_1_49375:(
batchnormalization_1_49396:(
batchnormalization_1_49398:(
batchnormalization_1_49400:(
batchnormalization_1_49402:'
conv_1b_49417: 
conv_1b_49419: )
batchnormalization_1b_49446: )
batchnormalization_1b_49448: )
batchnormalization_1b_49450: )
batchnormalization_1b_49452: &
conv_2_49467: @
conv_2_49469:@'
conv_2b_49503:@@
conv_2b_49505:@)
batchnormalization_2b_49532:@)
batchnormalization_2b_49534:@)
batchnormalization_2b_49536:@)
batchnormalization_2b_49538:@'
conv_3_49553:@
conv_3_49555:	)
batchnormalization_3_49582:	)
batchnormalization_3_49584:	)
batchnormalization_3_49586:	)
batchnormalization_3_49588:	(
conv_4_49609:
conv_4_49611:	)
batchnormalization_4_49638:	)
batchnormalization_4_49640:	)
batchnormalization_4_49642:	)
batchnormalization_4_49644:	)
batchnormalization_5_49654:	)
batchnormalization_5_49656:	)
batchnormalization_5_49658:	)
batchnormalization_5_49660:	$
dense_final_49688:	
dense_final_49690:
identity

identity_1¢,batchNormalization_1/StatefulPartitionedCall¢-batchNormalization_1b/StatefulPartitionedCall¢-batchNormalization_2b/StatefulPartitionedCall¢,batchNormalization_3/StatefulPartitionedCall¢,batchNormalization_4/StatefulPartitionedCall¢,batchNormalization_5/StatefulPartitionedCall¢conv_1/StatefulPartitionedCall¢conv_1b/StatefulPartitionedCall¢conv_2/StatefulPartitionedCall¢conv_2b/StatefulPartitionedCall¢conv_3/StatefulPartitionedCall¢conv_4/StatefulPartitionedCall¢#dense_final/StatefulPartitionedCall¢4dense_final/kernel/Regularizer/Square/ReadVariableOp
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_49373conv_1_49375*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_493722 
conv_1/StatefulPartitionedCallµ
,batchNormalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batchnormalization_1_49396batchnormalization_1_49398batchnormalization_1_49400batchnormalization_1_49402*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_493952.
,batchNormalization_1/StatefulPartitionedCallÆ
conv_1b/StatefulPartitionedCallStatefulPartitionedCall5batchNormalization_1/StatefulPartitionedCall:output:0conv_1b_49417conv_1b_49419*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_1b_layer_call_and_return_conditional_losses_494162!
conv_1b/StatefulPartitionedCall
maxPool_1b/PartitionedCallPartitionedCall(conv_1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_494262
maxPool_1b/PartitionedCall¸
-batchNormalization_1b/StatefulPartitionedCallStatefulPartitionedCall#maxPool_1b/PartitionedCall:output:0batchnormalization_1b_49446batchnormalization_1b_49448batchnormalization_1b_49450batchnormalization_1b_49452*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_494452/
-batchNormalization_1b/StatefulPartitionedCallÂ
conv_2/StatefulPartitionedCallStatefulPartitionedCall6batchNormalization_1b/StatefulPartitionedCall:output:0conv_2_49467conv_2_49469*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_494662 
conv_2/StatefulPartitionedCall
maxPool_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_2_layer_call_and_return_conditional_losses_494762
maxPool_2/PartitionedCallý
dropout_2/PartitionedCallPartitionedCall"maxPool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_494832
dropout_2/PartitionedCall÷
noise_1/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_1_layer_call_and_return_conditional_losses_494892
noise_1/PartitionedCall±
conv_2b/StatefulPartitionedCallStatefulPartitionedCall noise_1/PartitionedCall:output:0conv_2b_49503conv_2b_49505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_2b_layer_call_and_return_conditional_losses_495022!
conv_2b/StatefulPartitionedCall
maxPool_2b/PartitionedCallPartitionedCall(conv_2b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_495122
maxPool_2b/PartitionedCall¸
-batchNormalization_2b/StatefulPartitionedCallStatefulPartitionedCall#maxPool_2b/PartitionedCall:output:0batchnormalization_2b_49532batchnormalization_2b_49534batchnormalization_2b_49536batchnormalization_2b_49538*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_495312/
-batchNormalization_2b/StatefulPartitionedCallÃ
conv_3/StatefulPartitionedCallStatefulPartitionedCall6batchNormalization_2b/StatefulPartitionedCall:output:0conv_3_49553conv_3_49555*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_495522 
conv_3/StatefulPartitionedCall
maxPool_3/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_3_layer_call_and_return_conditional_losses_495622
maxPool_3/PartitionedCall±
,batchNormalization_3/StatefulPartitionedCallStatefulPartitionedCall"maxPool_3/PartitionedCall:output:0batchnormalization_3_49582batchnormalization_3_49584batchnormalization_3_49586batchnormalization_3_49588*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_495812.
,batchNormalization_3/StatefulPartitionedCall
noise_3/PartitionedCallPartitionedCall5batchNormalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_3_layer_call_and_return_conditional_losses_495952
noise_3/PartitionedCall­
conv_4/StatefulPartitionedCallStatefulPartitionedCall noise_3/PartitionedCall:output:0conv_4_49609conv_4_49611*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_496082 
conv_4/StatefulPartitionedCall
maxPool_4/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_4_layer_call_and_return_conditional_losses_496182
maxPool_4/PartitionedCall±
,batchNormalization_4/StatefulPartitionedCallStatefulPartitionedCall"maxPool_4/PartitionedCall:output:0batchnormalization_4_49638batchnormalization_4_49640batchnormalization_4_49642batchnormalization_4_49644*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_496372.
,batchNormalization_4/StatefulPartitionedCall
globAvgPool_5/PartitionedCallPartitionedCall5batchNormalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_496522
globAvgPool_5/PartitionedCall­
,batchNormalization_5/StatefulPartitionedCallStatefulPartitionedCall&globAvgPool_5/PartitionedCall:output:0batchnormalization_5_49654batchnormalization_5_49656batchnormalization_5_49658batchnormalization_5_49660*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_492032.
,batchNormalization_5/StatefulPartitionedCall
dropout_6/PartitionedCallPartitionedCall5batchNormalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_496682
dropout_6/PartitionedCall¿
#dense_final/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_final_49688dense_final_49690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_496872%
#dense_final/StatefulPartitionedCall
/dense_final/ActivityRegularizer/PartitionedCallPartitionedCall,dense_final/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *;
f6R4
2__inference_dense_final_activity_regularizer_4935421
/dense_final/ActivityRegularizer/PartitionedCallª
%dense_final/ActivityRegularizer/ShapeShape,dense_final/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2'
%dense_final/ActivityRegularizer/Shape´
3dense_final/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3dense_final/ActivityRegularizer/strided_slice/stack¸
5dense_final/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_1¸
5dense_final/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_2¢
-dense_final/ActivityRegularizer/strided_sliceStridedSlice.dense_final/ActivityRegularizer/Shape:output:0<dense_final/ActivityRegularizer/strided_slice/stack:output:0>dense_final/ActivityRegularizer/strided_slice/stack_1:output:0>dense_final/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-dense_final/ActivityRegularizer/strided_slice¼
$dense_final/ActivityRegularizer/CastCast6dense_final/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$dense_final/ActivityRegularizer/Castâ
'dense_final/ActivityRegularizer/truedivRealDiv8dense_final/ActivityRegularizer/PartitionedCall:output:0(dense_final/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_final/ActivityRegularizer/truediv¿
4dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_final_49688*
_output_shapes
:	*
dtype026
4dense_final/kernel/Regularizer/Square/ReadVariableOpÀ
%dense_final/kernel/Regularizer/SquareSquare<dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%dense_final/kernel/Regularizer/Square
$dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$dense_final/kernel/Regularizer/ConstÊ
"dense_final/kernel/Regularizer/SumSum)dense_final/kernel/Regularizer/Square:y:0-dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/Sum
$dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *-²=2&
$dense_final/kernel/Regularizer/mul/xÌ
"dense_final/kernel/Regularizer/mulMul-dense_final/kernel/Regularizer/mul/x:output:0+dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/mul
IdentityIdentity,dense_final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityy

Identity_1Identity+dense_final/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1
NoOpNoOp-^batchNormalization_1/StatefulPartitionedCall.^batchNormalization_1b/StatefulPartitionedCall.^batchNormalization_2b/StatefulPartitionedCall-^batchNormalization_3/StatefulPartitionedCall-^batchNormalization_4/StatefulPartitionedCall-^batchNormalization_5/StatefulPartitionedCall^conv_1/StatefulPartitionedCall ^conv_1b/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^conv_2b/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall$^dense_final/StatefulPartitionedCall5^dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,batchNormalization_1/StatefulPartitionedCall,batchNormalization_1/StatefulPartitionedCall2^
-batchNormalization_1b/StatefulPartitionedCall-batchNormalization_1b/StatefulPartitionedCall2^
-batchNormalization_2b/StatefulPartitionedCall-batchNormalization_2b/StatefulPartitionedCall2\
,batchNormalization_3/StatefulPartitionedCall,batchNormalization_3/StatefulPartitionedCall2\
,batchNormalization_4/StatefulPartitionedCall,batchNormalization_4/StatefulPartitionedCall2\
,batchNormalization_5/StatefulPartitionedCall,batchNormalization_5/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2B
conv_1b/StatefulPartitionedCallconv_1b/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2B
conv_2b/StatefulPartitionedCallconv_2b/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2J
#dense_final/StatefulPartitionedCall#dense_final/StatefulPartitionedCall2l
4dense_final/kernel/Regularizer/Square/ReadVariableOp4dense_final/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ü
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_50080

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed¡¶2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿	
Ð
5__inference_batchNormalization_1b_layer_call_fn_51728

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_486292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ñ

O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52068

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_51811

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed¡¶2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ß
C
'__inference_noise_3_layer_call_fn_52194

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_3_layer_call_and_return_conditional_losses_495952
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
E
)__inference_maxPool_3_layer_call_fn_52050

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_3_layer_call_and_return_conditional_losses_495622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
`
D__inference_maxPool_2_layer_call_and_return_conditional_losses_51779

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
d
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_49165

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


&__inference_conv_4_layer_call_fn_52219

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_496082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
èÕ
×%
G__inference_sequential_5_layer_call_and_return_conditional_losses_51282

inputs?
%conv_1_conv2d_readvariableop_resource:4
&conv_1_biasadd_readvariableop_resource::
,batchnormalization_1_readvariableop_resource:<
.batchnormalization_1_readvariableop_1_resource:K
=batchnormalization_1_fusedbatchnormv3_readvariableop_resource:M
?batchnormalization_1_fusedbatchnormv3_readvariableop_1_resource:@
&conv_1b_conv2d_readvariableop_resource: 5
'conv_1b_biasadd_readvariableop_resource: ;
-batchnormalization_1b_readvariableop_resource: =
/batchnormalization_1b_readvariableop_1_resource: L
>batchnormalization_1b_fusedbatchnormv3_readvariableop_resource: N
@batchnormalization_1b_fusedbatchnormv3_readvariableop_1_resource: ?
%conv_2_conv2d_readvariableop_resource: @4
&conv_2_biasadd_readvariableop_resource:@@
&conv_2b_conv2d_readvariableop_resource:@@5
'conv_2b_biasadd_readvariableop_resource:@;
-batchnormalization_2b_readvariableop_resource:@=
/batchnormalization_2b_readvariableop_1_resource:@L
>batchnormalization_2b_fusedbatchnormv3_readvariableop_resource:@N
@batchnormalization_2b_fusedbatchnormv3_readvariableop_1_resource:@@
%conv_3_conv2d_readvariableop_resource:@5
&conv_3_biasadd_readvariableop_resource:	;
,batchnormalization_3_readvariableop_resource:	=
.batchnormalization_3_readvariableop_1_resource:	L
=batchnormalization_3_fusedbatchnormv3_readvariableop_resource:	N
?batchnormalization_3_fusedbatchnormv3_readvariableop_1_resource:	A
%conv_4_conv2d_readvariableop_resource:5
&conv_4_biasadd_readvariableop_resource:	;
,batchnormalization_4_readvariableop_resource:	=
.batchnormalization_4_readvariableop_1_resource:	L
=batchnormalization_4_fusedbatchnormv3_readvariableop_resource:	N
?batchnormalization_4_fusedbatchnormv3_readvariableop_1_resource:	K
<batchnormalization_5_assignmovingavg_readvariableop_resource:	M
>batchnormalization_5_assignmovingavg_1_readvariableop_resource:	I
:batchnormalization_5_batchnorm_mul_readvariableop_resource:	E
6batchnormalization_5_batchnorm_readvariableop_resource:	=
*dense_final_matmul_readvariableop_resource:	9
+dense_final_biasadd_readvariableop_resource:
identity

identity_1¢#batchNormalization_1/AssignNewValue¢%batchNormalization_1/AssignNewValue_1¢4batchNormalization_1/FusedBatchNormV3/ReadVariableOp¢6batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1¢#batchNormalization_1/ReadVariableOp¢%batchNormalization_1/ReadVariableOp_1¢$batchNormalization_1b/AssignNewValue¢&batchNormalization_1b/AssignNewValue_1¢5batchNormalization_1b/FusedBatchNormV3/ReadVariableOp¢7batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1¢$batchNormalization_1b/ReadVariableOp¢&batchNormalization_1b/ReadVariableOp_1¢$batchNormalization_2b/AssignNewValue¢&batchNormalization_2b/AssignNewValue_1¢5batchNormalization_2b/FusedBatchNormV3/ReadVariableOp¢7batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1¢$batchNormalization_2b/ReadVariableOp¢&batchNormalization_2b/ReadVariableOp_1¢#batchNormalization_3/AssignNewValue¢%batchNormalization_3/AssignNewValue_1¢4batchNormalization_3/FusedBatchNormV3/ReadVariableOp¢6batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1¢#batchNormalization_3/ReadVariableOp¢%batchNormalization_3/ReadVariableOp_1¢#batchNormalization_4/AssignNewValue¢%batchNormalization_4/AssignNewValue_1¢4batchNormalization_4/FusedBatchNormV3/ReadVariableOp¢6batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1¢#batchNormalization_4/ReadVariableOp¢%batchNormalization_4/ReadVariableOp_1¢$batchNormalization_5/AssignMovingAvg¢3batchNormalization_5/AssignMovingAvg/ReadVariableOp¢&batchNormalization_5/AssignMovingAvg_1¢5batchNormalization_5/AssignMovingAvg_1/ReadVariableOp¢-batchNormalization_5/batchnorm/ReadVariableOp¢1batchNormalization_5/batchnorm/mul/ReadVariableOp¢conv_1/BiasAdd/ReadVariableOp¢conv_1/Conv2D/ReadVariableOp¢conv_1b/BiasAdd/ReadVariableOp¢conv_1b/Conv2D/ReadVariableOp¢conv_2/BiasAdd/ReadVariableOp¢conv_2/Conv2D/ReadVariableOp¢conv_2b/BiasAdd/ReadVariableOp¢conv_2b/Conv2D/ReadVariableOp¢conv_3/BiasAdd/ReadVariableOp¢conv_3/Conv2D/ReadVariableOp¢conv_4/BiasAdd/ReadVariableOp¢conv_4/Conv2D/ReadVariableOp¢"dense_final/BiasAdd/ReadVariableOp¢!dense_final/MatMul/ReadVariableOp¢4dense_final/kernel/Regularizer/Square/ReadVariableOpª
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_1/Conv2D/ReadVariableOp¸
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2
conv_1/Conv2D¡
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp¤
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
conv_1/BiasAddu
conv_1/ReluReluconv_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
conv_1/Relu³
#batchNormalization_1/ReadVariableOpReadVariableOp,batchnormalization_1_readvariableop_resource*
_output_shapes
:*
dtype02%
#batchNormalization_1/ReadVariableOp¹
%batchNormalization_1/ReadVariableOp_1ReadVariableOp.batchnormalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02'
%batchNormalization_1/ReadVariableOp_1æ
4batchNormalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp=batchnormalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype026
4batchNormalization_1/FusedBatchNormV3/ReadVariableOpì
6batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?batchnormalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype028
6batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1é
%batchNormalization_1/FusedBatchNormV3FusedBatchNormV3conv_1/Relu:activations:0+batchNormalization_1/ReadVariableOp:value:0-batchNormalization_1/ReadVariableOp_1:value:0<batchNormalization_1/FusedBatchNormV3/ReadVariableOp:value:0>batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
exponential_avg_factor%
×#<2'
%batchNormalization_1/FusedBatchNormV3«
#batchNormalization_1/AssignNewValueAssignVariableOp=batchnormalization_1_fusedbatchnormv3_readvariableop_resource2batchNormalization_1/FusedBatchNormV3:batch_mean:05^batchNormalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02%
#batchNormalization_1/AssignNewValue·
%batchNormalization_1/AssignNewValue_1AssignVariableOp?batchnormalization_1_fusedbatchnormv3_readvariableop_1_resource6batchNormalization_1/FusedBatchNormV3:batch_variance:07^batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02'
%batchNormalization_1/AssignNewValue_1­
conv_1b/Conv2D/ReadVariableOpReadVariableOp&conv_1b_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_1b/Conv2D/ReadVariableOpÞ
conv_1b/Conv2DConv2D)batchNormalization_1/FusedBatchNormV3:y:0%conv_1b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingSAME*
strides
2
conv_1b/Conv2D¤
conv_1b/BiasAdd/ReadVariableOpReadVariableOp'conv_1b_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv_1b/BiasAdd/ReadVariableOp¨
conv_1b/BiasAddBiasAddconv_1b/Conv2D:output:0&conv_1b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
conv_1b/BiasAddx
conv_1b/ReluReluconv_1b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
conv_1b/Relu¼
maxPool_1b/MaxPoolMaxPoolconv_1b/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *
ksize
*
paddingVALID*
strides
2
maxPool_1b/MaxPool¶
$batchNormalization_1b/ReadVariableOpReadVariableOp-batchnormalization_1b_readvariableop_resource*
_output_shapes
: *
dtype02&
$batchNormalization_1b/ReadVariableOp¼
&batchNormalization_1b/ReadVariableOp_1ReadVariableOp/batchnormalization_1b_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batchNormalization_1b/ReadVariableOp_1é
5batchNormalization_1b/FusedBatchNormV3/ReadVariableOpReadVariableOp>batchnormalization_1b_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batchNormalization_1b/FusedBatchNormV3/ReadVariableOpï
7batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batchnormalization_1b_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1ñ
&batchNormalization_1b/FusedBatchNormV3FusedBatchNormV3maxPool_1b/MaxPool:output:0,batchNormalization_1b/ReadVariableOp:value:0.batchNormalization_1b/ReadVariableOp_1:value:0=batchNormalization_1b/FusedBatchNormV3/ReadVariableOp:value:0?batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batchNormalization_1b/FusedBatchNormV3°
$batchNormalization_1b/AssignNewValueAssignVariableOp>batchnormalization_1b_fusedbatchnormv3_readvariableop_resource3batchNormalization_1b/FusedBatchNormV3:batch_mean:06^batchNormalization_1b/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batchNormalization_1b/AssignNewValue¼
&batchNormalization_1b/AssignNewValue_1AssignVariableOp@batchnormalization_1b_fusedbatchnormv3_readvariableop_1_resource7batchNormalization_1b/FusedBatchNormV3:batch_variance:08^batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batchNormalization_1b/AssignNewValue_1ª
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_2/Conv2D/ReadVariableOpÜ
conv_2/Conv2DConv2D*batchNormalization_1b/FusedBatchNormV3:y:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
conv_2/Conv2D¡
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_2/BiasAdd/ReadVariableOp¤
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv_2/BiasAddu
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv_2/Relu¹
maxPool_2/MaxPoolMaxPoolconv_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
maxPool_2/MaxPoolw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout_2/dropout/Const­
dropout_2/dropout/MulMulmaxPool_2/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/dropout/Mul|
dropout_2/dropout/ShapeShapemaxPool_2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeê
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed¡¶20
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_2/dropout/GreaterEqual/yî
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout_2/dropout/GreaterEqual¥
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/dropout/Castª
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/dropout/Mul_1i
noise_1/ShapeShapedropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
noise_1/Shape}
noise_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
noise_1/random_normal/mean
noise_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *>2
noise_1/random_normal/stddevï
*noise_1/random_normal/RandomStandardNormalRandomStandardNormalnoise_1/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed¡¶*
seed2°áÅ2,
*noise_1/random_normal/RandomStandardNormalÓ
noise_1/random_normal/mulMul3noise_1/random_normal/RandomStandardNormal:output:0%noise_1/random_normal/stddev:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
noise_1/random_normal/mulµ
noise_1/random_normalAddV2noise_1/random_normal/mul:z:0#noise_1/random_normal/mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
noise_1/random_normal
noise_1/addAddV2dropout_2/dropout/Mul_1:z:0noise_1/random_normal:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
noise_1/add­
conv_2b/Conv2D/ReadVariableOpReadVariableOp&conv_2b_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv_2b/Conv2D/ReadVariableOpÄ
conv_2b/Conv2DConv2Dnoise_1/add:z:0%conv_2b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv_2b/Conv2D¤
conv_2b/BiasAdd/ReadVariableOpReadVariableOp'conv_2b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv_2b/BiasAdd/ReadVariableOp¨
conv_2b/BiasAddBiasAddconv_2b/Conv2D:output:0&conv_2b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_2b/BiasAddx
conv_2b/ReluReluconv_2b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_2b/Relu¼
maxPool_2b/MaxPoolMaxPoolconv_2b/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
maxPool_2b/MaxPool¶
$batchNormalization_2b/ReadVariableOpReadVariableOp-batchnormalization_2b_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batchNormalization_2b/ReadVariableOp¼
&batchNormalization_2b/ReadVariableOp_1ReadVariableOp/batchnormalization_2b_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batchNormalization_2b/ReadVariableOp_1é
5batchNormalization_2b/FusedBatchNormV3/ReadVariableOpReadVariableOp>batchnormalization_2b_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batchNormalization_2b/FusedBatchNormV3/ReadVariableOpï
7batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batchnormalization_2b_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1ñ
&batchNormalization_2b/FusedBatchNormV3FusedBatchNormV3maxPool_2b/MaxPool:output:0,batchNormalization_2b/ReadVariableOp:value:0.batchNormalization_2b/ReadVariableOp_1:value:0=batchNormalization_2b/FusedBatchNormV3/ReadVariableOp:value:0?batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batchNormalization_2b/FusedBatchNormV3°
$batchNormalization_2b/AssignNewValueAssignVariableOp>batchnormalization_2b_fusedbatchnormv3_readvariableop_resource3batchNormalization_2b/FusedBatchNormV3:batch_mean:06^batchNormalization_2b/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batchNormalization_2b/AssignNewValue¼
&batchNormalization_2b/AssignNewValue_1AssignVariableOp@batchnormalization_2b_fusedbatchnormv3_readvariableop_1_resource7batchNormalization_2b/FusedBatchNormV3:batch_variance:08^batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batchNormalization_2b/AssignNewValue_1«
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_3/Conv2D/ReadVariableOpÝ
conv_3/Conv2DConv2D*batchNormalization_2b/FusedBatchNormV3:y:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_3/Conv2D¢
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_3/BiasAdd/ReadVariableOp¥
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_3/BiasAddv
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_3/Reluº
maxPool_3/MaxPoolMaxPoolconv_3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
maxPool_3/MaxPool´
#batchNormalization_3/ReadVariableOpReadVariableOp,batchnormalization_3_readvariableop_resource*
_output_shapes	
:*
dtype02%
#batchNormalization_3/ReadVariableOpº
%batchNormalization_3/ReadVariableOp_1ReadVariableOp.batchnormalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02'
%batchNormalization_3/ReadVariableOp_1ç
4batchNormalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp=batchnormalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype026
4batchNormalization_3/FusedBatchNormV3/ReadVariableOpí
6batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?batchnormalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1ï
%batchNormalization_3/FusedBatchNormV3FusedBatchNormV3maxPool_3/MaxPool:output:0+batchNormalization_3/ReadVariableOp:value:0-batchNormalization_3/ReadVariableOp_1:value:0<batchNormalization_3/FusedBatchNormV3/ReadVariableOp:value:0>batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2'
%batchNormalization_3/FusedBatchNormV3«
#batchNormalization_3/AssignNewValueAssignVariableOp=batchnormalization_3_fusedbatchnormv3_readvariableop_resource2batchNormalization_3/FusedBatchNormV3:batch_mean:05^batchNormalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02%
#batchNormalization_3/AssignNewValue·
%batchNormalization_3/AssignNewValue_1AssignVariableOp?batchnormalization_3_fusedbatchnormv3_readvariableop_1_resource6batchNormalization_3/FusedBatchNormV3:batch_variance:07^batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02'
%batchNormalization_3/AssignNewValue_1w
noise_3/ShapeShape)batchNormalization_3/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
noise_3/Shape}
noise_3/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
noise_3/random_normal/mean
noise_3/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *>2
noise_3/random_normal/stddevð
*noise_3/random_normal/RandomStandardNormalRandomStandardNormalnoise_3/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed¡¶*
seed2¦Ã2,
*noise_3/random_normal/RandomStandardNormalÔ
noise_3/random_normal/mulMul3noise_3/random_normal/RandomStandardNormal:output:0%noise_3/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
noise_3/random_normal/mul¶
noise_3/random_normalAddV2noise_3/random_normal/mul:z:0#noise_3/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
noise_3/random_normal¤
noise_3/addAddV2)batchNormalization_3/FusedBatchNormV3:y:0noise_3/random_normal:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
noise_3/add¬
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOpÂ
conv_4/Conv2DConv2Dnoise_3/add:z:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_4/Conv2D¢
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_4/BiasAdd/ReadVariableOp¥
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_4/BiasAddv
conv_4/ReluReluconv_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_4/Reluº
maxPool_4/MaxPoolMaxPoolconv_4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
maxPool_4/MaxPool´
#batchNormalization_4/ReadVariableOpReadVariableOp,batchnormalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02%
#batchNormalization_4/ReadVariableOpº
%batchNormalization_4/ReadVariableOp_1ReadVariableOp.batchnormalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02'
%batchNormalization_4/ReadVariableOp_1ç
4batchNormalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp=batchnormalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype026
4batchNormalization_4/FusedBatchNormV3/ReadVariableOpí
6batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?batchnormalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1ï
%batchNormalization_4/FusedBatchNormV3FusedBatchNormV3maxPool_4/MaxPool:output:0+batchNormalization_4/ReadVariableOp:value:0-batchNormalization_4/ReadVariableOp_1:value:0<batchNormalization_4/FusedBatchNormV3/ReadVariableOp:value:0>batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2'
%batchNormalization_4/FusedBatchNormV3«
#batchNormalization_4/AssignNewValueAssignVariableOp=batchnormalization_4_fusedbatchnormv3_readvariableop_resource2batchNormalization_4/FusedBatchNormV3:batch_mean:05^batchNormalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02%
#batchNormalization_4/AssignNewValue·
%batchNormalization_4/AssignNewValue_1AssignVariableOp?batchnormalization_4_fusedbatchnormv3_readvariableop_1_resource6batchNormalization_4/FusedBatchNormV3:batch_variance:07^batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02'
%batchNormalization_4/AssignNewValue_1
$globAvgPool_5/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2&
$globAvgPool_5/Mean/reduction_indices½
globAvgPool_5/MeanMean)batchNormalization_4/FusedBatchNormV3:y:0-globAvgPool_5/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
globAvgPool_5/Mean´
3batchNormalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 25
3batchNormalization_5/moments/mean/reduction_indicesä
!batchNormalization_5/moments/meanMeanglobAvgPool_5/Mean:output:0<batchNormalization_5/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2#
!batchNormalization_5/moments/mean¼
)batchNormalization_5/moments/StopGradientStopGradient*batchNormalization_5/moments/mean:output:0*
T0*
_output_shapes
:	2+
)batchNormalization_5/moments/StopGradientù
.batchNormalization_5/moments/SquaredDifferenceSquaredDifferenceglobAvgPool_5/Mean:output:02batchNormalization_5/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.batchNormalization_5/moments/SquaredDifference¼
7batchNormalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 29
7batchNormalization_5/moments/variance/reduction_indices
%batchNormalization_5/moments/varianceMean2batchNormalization_5/moments/SquaredDifference:z:0@batchNormalization_5/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2'
%batchNormalization_5/moments/varianceÀ
$batchNormalization_5/moments/SqueezeSqueeze*batchNormalization_5/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2&
$batchNormalization_5/moments/SqueezeÈ
&batchNormalization_5/moments/Squeeze_1Squeeze.batchNormalization_5/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2(
&batchNormalization_5/moments/Squeeze_1
*batchNormalization_5/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2,
*batchNormalization_5/AssignMovingAvg/decayä
3batchNormalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp<batchnormalization_5_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype025
3batchNormalization_5/AssignMovingAvg/ReadVariableOpí
(batchNormalization_5/AssignMovingAvg/subSub;batchNormalization_5/AssignMovingAvg/ReadVariableOp:value:0-batchNormalization_5/moments/Squeeze:output:0*
T0*
_output_shapes	
:2*
(batchNormalization_5/AssignMovingAvg/subä
(batchNormalization_5/AssignMovingAvg/mulMul,batchNormalization_5/AssignMovingAvg/sub:z:03batchNormalization_5/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2*
(batchNormalization_5/AssignMovingAvg/mul¨
$batchNormalization_5/AssignMovingAvgAssignSubVariableOp<batchnormalization_5_assignmovingavg_readvariableop_resource,batchNormalization_5/AssignMovingAvg/mul:z:04^batchNormalization_5/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02&
$batchNormalization_5/AssignMovingAvg¡
,batchNormalization_5/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2.
,batchNormalization_5/AssignMovingAvg_1/decayê
5batchNormalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batchnormalization_5_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype027
5batchNormalization_5/AssignMovingAvg_1/ReadVariableOpõ
*batchNormalization_5/AssignMovingAvg_1/subSub=batchNormalization_5/AssignMovingAvg_1/ReadVariableOp:value:0/batchNormalization_5/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2,
*batchNormalization_5/AssignMovingAvg_1/subì
*batchNormalization_5/AssignMovingAvg_1/mulMul.batchNormalization_5/AssignMovingAvg_1/sub:z:05batchNormalization_5/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2,
*batchNormalization_5/AssignMovingAvg_1/mul²
&batchNormalization_5/AssignMovingAvg_1AssignSubVariableOp>batchnormalization_5_assignmovingavg_1_readvariableop_resource.batchNormalization_5/AssignMovingAvg_1/mul:z:06^batchNormalization_5/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02(
&batchNormalization_5/AssignMovingAvg_1
$batchNormalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$batchNormalization_5/batchnorm/add/y×
"batchNormalization_5/batchnorm/addAddV2/batchNormalization_5/moments/Squeeze_1:output:0-batchNormalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2$
"batchNormalization_5/batchnorm/add£
$batchNormalization_5/batchnorm/RsqrtRsqrt&batchNormalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:2&
$batchNormalization_5/batchnorm/RsqrtÞ
1batchNormalization_5/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype023
1batchNormalization_5/batchnorm/mul/ReadVariableOpÚ
"batchNormalization_5/batchnorm/mulMul(batchNormalization_5/batchnorm/Rsqrt:y:09batchNormalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"batchNormalization_5/batchnorm/mulË
$batchNormalization_5/batchnorm/mul_1MulglobAvgPool_5/Mean:output:0&batchNormalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$batchNormalization_5/batchnorm/mul_1Ð
$batchNormalization_5/batchnorm/mul_2Mul-batchNormalization_5/moments/Squeeze:output:0&batchNormalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:2&
$batchNormalization_5/batchnorm/mul_2Ò
-batchNormalization_5/batchnorm/ReadVariableOpReadVariableOp6batchnormalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02/
-batchNormalization_5/batchnorm/ReadVariableOpÖ
"batchNormalization_5/batchnorm/subSub5batchNormalization_5/batchnorm/ReadVariableOp:value:0(batchNormalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2$
"batchNormalization_5/batchnorm/subÚ
$batchNormalization_5/batchnorm/add_1AddV2(batchNormalization_5/batchnorm/mul_1:z:0&batchNormalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$batchNormalization_5/batchnorm/add_1w
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout_6/dropout/Const´
dropout_6/dropout/MulMul(batchNormalization_5/batchnorm/add_1:z:0 dropout_6/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_6/dropout/Mul
dropout_6/dropout/ShapeShape(batchNormalization_5/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shapeð
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed¡¶*
seed220
.dropout_6/dropout/random_uniform/RandomUniform
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2"
 dropout_6/dropout/GreaterEqual/yç
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_6/dropout/GreaterEqual
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_6/dropout/Cast£
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_6/dropout/Mul_1²
!dense_final/MatMul/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!dense_final/MatMul/ReadVariableOp¬
dense_final/MatMulMatMuldropout_6/dropout/Mul_1:z:0)dense_final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_final/MatMul°
"dense_final/BiasAdd/ReadVariableOpReadVariableOp+dense_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dense_final/BiasAdd/ReadVariableOp±
dense_final/BiasAddBiasAdddense_final/MatMul:product:0*dense_final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_final/BiasAdd
dense_final/SoftmaxSoftmaxdense_final/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_final/Softmax«
&dense_final/ActivityRegularizer/SquareSquaredense_final/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&dense_final/ActivityRegularizer/Square
%dense_final/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_final/ActivityRegularizer/ConstÎ
#dense_final/ActivityRegularizer/SumSum*dense_final/ActivityRegularizer/Square:y:0.dense_final/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_final/ActivityRegularizer/Sum
%dense_final/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£<2'
%dense_final/ActivityRegularizer/mul/xÐ
#dense_final/ActivityRegularizer/mulMul.dense_final/ActivityRegularizer/mul/x:output:0,dense_final/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_final/ActivityRegularizer/mul
%dense_final/ActivityRegularizer/ShapeShapedense_final/Softmax:softmax:0*
T0*
_output_shapes
:2'
%dense_final/ActivityRegularizer/Shape´
3dense_final/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3dense_final/ActivityRegularizer/strided_slice/stack¸
5dense_final/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_1¸
5dense_final/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_2¢
-dense_final/ActivityRegularizer/strided_sliceStridedSlice.dense_final/ActivityRegularizer/Shape:output:0<dense_final/ActivityRegularizer/strided_slice/stack:output:0>dense_final/ActivityRegularizer/strided_slice/stack_1:output:0>dense_final/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-dense_final/ActivityRegularizer/strided_slice¼
$dense_final/ActivityRegularizer/CastCast6dense_final/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$dense_final/ActivityRegularizer/CastÑ
'dense_final/ActivityRegularizer/truedivRealDiv'dense_final/ActivityRegularizer/mul:z:0(dense_final/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_final/ActivityRegularizer/truedivØ
4dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes
:	*
dtype026
4dense_final/kernel/Regularizer/Square/ReadVariableOpÀ
%dense_final/kernel/Regularizer/SquareSquare<dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%dense_final/kernel/Regularizer/Square
$dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$dense_final/kernel/Regularizer/ConstÊ
"dense_final/kernel/Regularizer/SumSum)dense_final/kernel/Regularizer/Square:y:0-dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/Sum
$dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *-²=2&
$dense_final/kernel/Regularizer/mul/xÌ
"dense_final/kernel/Regularizer/mulMul-dense_final/kernel/Regularizer/mul/x:output:0+dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/mulx
IdentityIdentitydense_final/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityy

Identity_1Identity+dense_final/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1¶
NoOpNoOp$^batchNormalization_1/AssignNewValue&^batchNormalization_1/AssignNewValue_15^batchNormalization_1/FusedBatchNormV3/ReadVariableOp7^batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1$^batchNormalization_1/ReadVariableOp&^batchNormalization_1/ReadVariableOp_1%^batchNormalization_1b/AssignNewValue'^batchNormalization_1b/AssignNewValue_16^batchNormalization_1b/FusedBatchNormV3/ReadVariableOp8^batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1%^batchNormalization_1b/ReadVariableOp'^batchNormalization_1b/ReadVariableOp_1%^batchNormalization_2b/AssignNewValue'^batchNormalization_2b/AssignNewValue_16^batchNormalization_2b/FusedBatchNormV3/ReadVariableOp8^batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1%^batchNormalization_2b/ReadVariableOp'^batchNormalization_2b/ReadVariableOp_1$^batchNormalization_3/AssignNewValue&^batchNormalization_3/AssignNewValue_15^batchNormalization_3/FusedBatchNormV3/ReadVariableOp7^batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1$^batchNormalization_3/ReadVariableOp&^batchNormalization_3/ReadVariableOp_1$^batchNormalization_4/AssignNewValue&^batchNormalization_4/AssignNewValue_15^batchNormalization_4/FusedBatchNormV3/ReadVariableOp7^batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1$^batchNormalization_4/ReadVariableOp&^batchNormalization_4/ReadVariableOp_1%^batchNormalization_5/AssignMovingAvg4^batchNormalization_5/AssignMovingAvg/ReadVariableOp'^batchNormalization_5/AssignMovingAvg_16^batchNormalization_5/AssignMovingAvg_1/ReadVariableOp.^batchNormalization_5/batchnorm/ReadVariableOp2^batchNormalization_5/batchnorm/mul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_1b/BiasAdd/ReadVariableOp^conv_1b/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_2b/BiasAdd/ReadVariableOp^conv_2b/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp#^dense_final/BiasAdd/ReadVariableOp"^dense_final/MatMul/ReadVariableOp5^dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#batchNormalization_1/AssignNewValue#batchNormalization_1/AssignNewValue2N
%batchNormalization_1/AssignNewValue_1%batchNormalization_1/AssignNewValue_12l
4batchNormalization_1/FusedBatchNormV3/ReadVariableOp4batchNormalization_1/FusedBatchNormV3/ReadVariableOp2p
6batchNormalization_1/FusedBatchNormV3/ReadVariableOp_16batchNormalization_1/FusedBatchNormV3/ReadVariableOp_12J
#batchNormalization_1/ReadVariableOp#batchNormalization_1/ReadVariableOp2N
%batchNormalization_1/ReadVariableOp_1%batchNormalization_1/ReadVariableOp_12L
$batchNormalization_1b/AssignNewValue$batchNormalization_1b/AssignNewValue2P
&batchNormalization_1b/AssignNewValue_1&batchNormalization_1b/AssignNewValue_12n
5batchNormalization_1b/FusedBatchNormV3/ReadVariableOp5batchNormalization_1b/FusedBatchNormV3/ReadVariableOp2r
7batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_17batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_12L
$batchNormalization_1b/ReadVariableOp$batchNormalization_1b/ReadVariableOp2P
&batchNormalization_1b/ReadVariableOp_1&batchNormalization_1b/ReadVariableOp_12L
$batchNormalization_2b/AssignNewValue$batchNormalization_2b/AssignNewValue2P
&batchNormalization_2b/AssignNewValue_1&batchNormalization_2b/AssignNewValue_12n
5batchNormalization_2b/FusedBatchNormV3/ReadVariableOp5batchNormalization_2b/FusedBatchNormV3/ReadVariableOp2r
7batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_17batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_12L
$batchNormalization_2b/ReadVariableOp$batchNormalization_2b/ReadVariableOp2P
&batchNormalization_2b/ReadVariableOp_1&batchNormalization_2b/ReadVariableOp_12J
#batchNormalization_3/AssignNewValue#batchNormalization_3/AssignNewValue2N
%batchNormalization_3/AssignNewValue_1%batchNormalization_3/AssignNewValue_12l
4batchNormalization_3/FusedBatchNormV3/ReadVariableOp4batchNormalization_3/FusedBatchNormV3/ReadVariableOp2p
6batchNormalization_3/FusedBatchNormV3/ReadVariableOp_16batchNormalization_3/FusedBatchNormV3/ReadVariableOp_12J
#batchNormalization_3/ReadVariableOp#batchNormalization_3/ReadVariableOp2N
%batchNormalization_3/ReadVariableOp_1%batchNormalization_3/ReadVariableOp_12J
#batchNormalization_4/AssignNewValue#batchNormalization_4/AssignNewValue2N
%batchNormalization_4/AssignNewValue_1%batchNormalization_4/AssignNewValue_12l
4batchNormalization_4/FusedBatchNormV3/ReadVariableOp4batchNormalization_4/FusedBatchNormV3/ReadVariableOp2p
6batchNormalization_4/FusedBatchNormV3/ReadVariableOp_16batchNormalization_4/FusedBatchNormV3/ReadVariableOp_12J
#batchNormalization_4/ReadVariableOp#batchNormalization_4/ReadVariableOp2N
%batchNormalization_4/ReadVariableOp_1%batchNormalization_4/ReadVariableOp_12L
$batchNormalization_5/AssignMovingAvg$batchNormalization_5/AssignMovingAvg2j
3batchNormalization_5/AssignMovingAvg/ReadVariableOp3batchNormalization_5/AssignMovingAvg/ReadVariableOp2P
&batchNormalization_5/AssignMovingAvg_1&batchNormalization_5/AssignMovingAvg_12n
5batchNormalization_5/AssignMovingAvg_1/ReadVariableOp5batchNormalization_5/AssignMovingAvg_1/ReadVariableOp2^
-batchNormalization_5/batchnorm/ReadVariableOp-batchNormalization_5/batchnorm/ReadVariableOp2f
1batchNormalization_5/batchnorm/mul/ReadVariableOp1batchNormalization_5/batchnorm/mul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2@
conv_1b/BiasAdd/ReadVariableOpconv_1b/BiasAdd/ReadVariableOp2>
conv_1b/Conv2D/ReadVariableOpconv_1b/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2@
conv_2b/BiasAdd/ReadVariableOpconv_2b/BiasAdd/ReadVariableOp2>
conv_2b/Conv2D/ReadVariableOpconv_2b/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
conv_4/BiasAdd/ReadVariableOpconv_4/BiasAdd/ReadVariableOp2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2H
"dense_final/BiasAdd/ReadVariableOp"dense_final/BiasAdd/ReadVariableOp2F
!dense_final/MatMul/ReadVariableOp!dense_final/MatMul/ReadVariableOp2l
4dense_final/kernel/Regularizer/Square/ReadVariableOp4dense_final/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
¿	
Ð
5__inference_batchNormalization_2b_layer_call_fn_51984

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_487992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¥
a
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_51871

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
Â
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_49953

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
`
D__inference_maxPool_4_layer_call_and_return_conditional_losses_49618

inputs
identity
MaxPoolMaxPoolinputs*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
ý
A__inference_conv_4_layer_call_and_return_conditional_losses_49608

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_49637

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
`
'__inference_noise_1_layer_call_fn_51846

inputs
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_1_layer_call_and_return_conditional_losses_500572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

^
B__inference_noise_1_layer_call_and_return_conditional_losses_51825

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

^
B__inference_noise_3_layer_call_and_return_conditional_losses_52178

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

a
B__inference_noise_3_layer_call_and_return_conditional_losses_49917

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *>2
random_normal/stddevØ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed¡¶*
seed2 2$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normali
addAddV2inputsrandom_normal:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
Â
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_49095

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
ú
A__inference_conv_2_layer_call_and_return_conditional_losses_49466

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ00 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
¥
Â
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_48947

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
Ó
4__inference_batchNormalization_5_layer_call_fn_52452

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_492032
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
Ó
4__inference_batchNormalization_5_layer_call_fn_52465

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_492632
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç	
Ó
4__inference_batchNormalization_3_layer_call_fn_52135

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_489032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
E
)__inference_maxPool_4_layer_call_fn_52234

inputs
identityè
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_4_layer_call_and_return_conditional_losses_490162
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
Ó
4__inference_batchNormalization_4_layer_call_fn_52363

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_498722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
`
D__inference_maxPool_4_layer_call_and_return_conditional_losses_49016

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
`
D__inference_maxPool_2_layer_call_and_return_conditional_losses_49476

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ00@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
Ì
E
)__inference_maxPool_2_layer_call_fn_51789

inputs
identityè
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_2_layer_call_and_return_conditional_losses_486982
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ

O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_48903

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á

O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51484

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â

P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51648

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¥
a
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_48720

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
û
B__inference_conv_1b_layer_call_and_return_conditional_losses_51601

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ``: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
÷
Ð
5__inference_batchNormalization_2b_layer_call_fn_52010

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_500122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨

O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52104

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
E
)__inference_maxPool_4_layer_call_fn_52239

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_4_layer_call_and_return_conditional_losses_496182
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_49395

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
è
û
B__inference_conv_2b_layer_call_and_return_conditional_losses_49502

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿	
Ï
4__inference_batchNormalization_1_layer_call_fn_51551

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_484372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
	
,__inference_sequential_5_layer_call_fn_50575
conv_1_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallconv_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: *<
_read_only_resource_inputs
	
#$%&*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_504132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameconv_1_input
¤
`
D__inference_maxPool_4_layer_call_and_return_conditional_losses_52224

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¿
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51702

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00 : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
ñ

O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_49051

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_49831

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÅ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed¡¶2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
B
!__inference__traced_restore_53173
file_prefix8
assignvariableop_conv_1_kernel:,
assignvariableop_1_conv_1_bias:;
-assignvariableop_2_batchnormalization_1_gamma::
,assignvariableop_3_batchnormalization_1_beta:A
3assignvariableop_4_batchnormalization_1_moving_mean:E
7assignvariableop_5_batchnormalization_1_moving_variance:;
!assignvariableop_6_conv_1b_kernel: -
assignvariableop_7_conv_1b_bias: <
.assignvariableop_8_batchnormalization_1b_gamma: ;
-assignvariableop_9_batchnormalization_1b_beta: C
5assignvariableop_10_batchnormalization_1b_moving_mean: G
9assignvariableop_11_batchnormalization_1b_moving_variance: ;
!assignvariableop_12_conv_2_kernel: @-
assignvariableop_13_conv_2_bias:@<
"assignvariableop_14_conv_2b_kernel:@@.
 assignvariableop_15_conv_2b_bias:@=
/assignvariableop_16_batchnormalization_2b_gamma:@<
.assignvariableop_17_batchnormalization_2b_beta:@C
5assignvariableop_18_batchnormalization_2b_moving_mean:@G
9assignvariableop_19_batchnormalization_2b_moving_variance:@<
!assignvariableop_20_conv_3_kernel:@.
assignvariableop_21_conv_3_bias:	=
.assignvariableop_22_batchnormalization_3_gamma:	<
-assignvariableop_23_batchnormalization_3_beta:	C
4assignvariableop_24_batchnormalization_3_moving_mean:	G
8assignvariableop_25_batchnormalization_3_moving_variance:	=
!assignvariableop_26_conv_4_kernel:.
assignvariableop_27_conv_4_bias:	=
.assignvariableop_28_batchnormalization_4_gamma:	<
-assignvariableop_29_batchnormalization_4_beta:	C
4assignvariableop_30_batchnormalization_4_moving_mean:	G
8assignvariableop_31_batchnormalization_4_moving_variance:	=
.assignvariableop_32_batchnormalization_5_gamma:	<
-assignvariableop_33_batchnormalization_5_beta:	C
4assignvariableop_34_batchnormalization_5_moving_mean:	G
8assignvariableop_35_batchnormalization_5_moving_variance:	9
&assignvariableop_36_dense_final_kernel:	2
$assignvariableop_37_dense_final_bias:)
assignvariableop_38_adamax_iter:	 +
!assignvariableop_39_adamax_beta_1: +
!assignvariableop_40_adamax_beta_2: *
 assignvariableop_41_adamax_decay: 2
(assignvariableop_42_adamax_learning_rate: #
assignvariableop_43_total: #
assignvariableop_44_count: %
assignvariableop_45_total_1: %
assignvariableop_46_count_1: D
*assignvariableop_47_adamax_conv_1_kernel_m:6
(assignvariableop_48_adamax_conv_1_bias_m:E
7assignvariableop_49_adamax_batchnormalization_1_gamma_m:D
6assignvariableop_50_adamax_batchnormalization_1_beta_m:E
+assignvariableop_51_adamax_conv_1b_kernel_m: 7
)assignvariableop_52_adamax_conv_1b_bias_m: F
8assignvariableop_53_adamax_batchnormalization_1b_gamma_m: E
7assignvariableop_54_adamax_batchnormalization_1b_beta_m: D
*assignvariableop_55_adamax_conv_2_kernel_m: @6
(assignvariableop_56_adamax_conv_2_bias_m:@E
+assignvariableop_57_adamax_conv_2b_kernel_m:@@7
)assignvariableop_58_adamax_conv_2b_bias_m:@F
8assignvariableop_59_adamax_batchnormalization_2b_gamma_m:@E
7assignvariableop_60_adamax_batchnormalization_2b_beta_m:@E
*assignvariableop_61_adamax_conv_3_kernel_m:@7
(assignvariableop_62_adamax_conv_3_bias_m:	F
7assignvariableop_63_adamax_batchnormalization_3_gamma_m:	E
6assignvariableop_64_adamax_batchnormalization_3_beta_m:	F
*assignvariableop_65_adamax_conv_4_kernel_m:7
(assignvariableop_66_adamax_conv_4_bias_m:	F
7assignvariableop_67_adamax_batchnormalization_4_gamma_m:	E
6assignvariableop_68_adamax_batchnormalization_4_beta_m:	F
7assignvariableop_69_adamax_batchnormalization_5_gamma_m:	E
6assignvariableop_70_adamax_batchnormalization_5_beta_m:	B
/assignvariableop_71_adamax_dense_final_kernel_m:	;
-assignvariableop_72_adamax_dense_final_bias_m:D
*assignvariableop_73_adamax_conv_1_kernel_v:6
(assignvariableop_74_adamax_conv_1_bias_v:E
7assignvariableop_75_adamax_batchnormalization_1_gamma_v:D
6assignvariableop_76_adamax_batchnormalization_1_beta_v:E
+assignvariableop_77_adamax_conv_1b_kernel_v: 7
)assignvariableop_78_adamax_conv_1b_bias_v: F
8assignvariableop_79_adamax_batchnormalization_1b_gamma_v: E
7assignvariableop_80_adamax_batchnormalization_1b_beta_v: D
*assignvariableop_81_adamax_conv_2_kernel_v: @6
(assignvariableop_82_adamax_conv_2_bias_v:@E
+assignvariableop_83_adamax_conv_2b_kernel_v:@@7
)assignvariableop_84_adamax_conv_2b_bias_v:@F
8assignvariableop_85_adamax_batchnormalization_2b_gamma_v:@E
7assignvariableop_86_adamax_batchnormalization_2b_beta_v:@E
*assignvariableop_87_adamax_conv_3_kernel_v:@7
(assignvariableop_88_adamax_conv_3_bias_v:	F
7assignvariableop_89_adamax_batchnormalization_3_gamma_v:	E
6assignvariableop_90_adamax_batchnormalization_3_beta_v:	F
*assignvariableop_91_adamax_conv_4_kernel_v:7
(assignvariableop_92_adamax_conv_4_bias_v:	F
7assignvariableop_93_adamax_batchnormalization_4_gamma_v:	E
6assignvariableop_94_adamax_batchnormalization_4_beta_v:	F
7assignvariableop_95_adamax_batchnormalization_5_gamma_v:	E
6assignvariableop_96_adamax_batchnormalization_5_beta_v:	B
/assignvariableop_97_adamax_dense_final_kernel_v:	;
-assignvariableop_98_adamax_dense_final_bias_v:
identity_100¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98Î7
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ú6
valueÐ6BÍ6dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÙ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ý
valueÓBÐdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¢
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2²
AssignVariableOp_2AssignVariableOp-assignvariableop_2_batchnormalization_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_batchnormalization_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¸
AssignVariableOp_4AssignVariableOp3assignvariableop_4_batchnormalization_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¼
AssignVariableOp_5AssignVariableOp7assignvariableop_5_batchnormalization_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¦
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv_1b_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¤
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_1b_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batchnormalization_1b_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9²
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batchnormalization_1b_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10½
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batchnormalization_1b_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batchnormalization_1b_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ª
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv_2b_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¨
AssignVariableOp_15AssignVariableOp assignvariableop_15_conv_2b_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16·
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batchnormalization_2b_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¶
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batchnormalization_2b_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18½
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batchnormalization_2b_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Á
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batchnormalization_2b_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20©
AssignVariableOp_20AssignVariableOp!assignvariableop_20_conv_3_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21§
AssignVariableOp_21AssignVariableOpassignvariableop_21_conv_3_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¶
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batchnormalization_3_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23µ
AssignVariableOp_23AssignVariableOp-assignvariableop_23_batchnormalization_3_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¼
AssignVariableOp_24AssignVariableOp4assignvariableop_24_batchnormalization_3_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25À
AssignVariableOp_25AssignVariableOp8assignvariableop_25_batchnormalization_3_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26©
AssignVariableOp_26AssignVariableOp!assignvariableop_26_conv_4_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27§
AssignVariableOp_27AssignVariableOpassignvariableop_27_conv_4_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¶
AssignVariableOp_28AssignVariableOp.assignvariableop_28_batchnormalization_4_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29µ
AssignVariableOp_29AssignVariableOp-assignvariableop_29_batchnormalization_4_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¼
AssignVariableOp_30AssignVariableOp4assignvariableop_30_batchnormalization_4_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31À
AssignVariableOp_31AssignVariableOp8assignvariableop_31_batchnormalization_4_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¶
AssignVariableOp_32AssignVariableOp.assignvariableop_32_batchnormalization_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33µ
AssignVariableOp_33AssignVariableOp-assignvariableop_33_batchnormalization_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¼
AssignVariableOp_34AssignVariableOp4assignvariableop_34_batchnormalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35À
AssignVariableOp_35AssignVariableOp8assignvariableop_35_batchnormalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36®
AssignVariableOp_36AssignVariableOp&assignvariableop_36_dense_final_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¬
AssignVariableOp_37AssignVariableOp$assignvariableop_37_dense_final_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_38§
AssignVariableOp_38AssignVariableOpassignvariableop_38_adamax_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39©
AssignVariableOp_39AssignVariableOp!assignvariableop_39_adamax_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40©
AssignVariableOp_40AssignVariableOp!assignvariableop_40_adamax_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¨
AssignVariableOp_41AssignVariableOp assignvariableop_41_adamax_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42°
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adamax_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¡
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¡
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45£
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46£
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47²
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adamax_conv_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48°
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adamax_conv_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¿
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adamax_batchnormalization_1_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50¾
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adamax_batchnormalization_1_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51³
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adamax_conv_1b_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52±
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adamax_conv_1b_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53À
AssignVariableOp_53AssignVariableOp8assignvariableop_53_adamax_batchnormalization_1b_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¿
AssignVariableOp_54AssignVariableOp7assignvariableop_54_adamax_batchnormalization_1b_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55²
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adamax_conv_2_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56°
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adamax_conv_2_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57³
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adamax_conv_2b_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58±
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adamax_conv_2b_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59À
AssignVariableOp_59AssignVariableOp8assignvariableop_59_adamax_batchnormalization_2b_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60¿
AssignVariableOp_60AssignVariableOp7assignvariableop_60_adamax_batchnormalization_2b_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61²
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adamax_conv_3_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62°
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adamax_conv_3_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¿
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adamax_batchnormalization_3_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64¾
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adamax_batchnormalization_3_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65²
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adamax_conv_4_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66°
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adamax_conv_4_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67¿
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adamax_batchnormalization_4_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68¾
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adamax_batchnormalization_4_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69¿
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adamax_batchnormalization_5_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70¾
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adamax_batchnormalization_5_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71·
AssignVariableOp_71AssignVariableOp/assignvariableop_71_adamax_dense_final_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72µ
AssignVariableOp_72AssignVariableOp-assignvariableop_72_adamax_dense_final_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73²
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adamax_conv_1_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74°
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adamax_conv_1_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75¿
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adamax_batchnormalization_1_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76¾
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adamax_batchnormalization_1_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77³
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adamax_conv_1b_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78±
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adamax_conv_1b_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79À
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adamax_batchnormalization_1b_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80¿
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adamax_batchnormalization_1b_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81²
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adamax_conv_2_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82°
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adamax_conv_2_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83³
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adamax_conv_2b_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84±
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adamax_conv_2b_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85À
AssignVariableOp_85AssignVariableOp8assignvariableop_85_adamax_batchnormalization_2b_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86¿
AssignVariableOp_86AssignVariableOp7assignvariableop_86_adamax_batchnormalization_2b_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87²
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adamax_conv_3_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88°
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adamax_conv_3_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89¿
AssignVariableOp_89AssignVariableOp7assignvariableop_89_adamax_batchnormalization_3_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90¾
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adamax_batchnormalization_3_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91²
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adamax_conv_4_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92°
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adamax_conv_4_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93¿
AssignVariableOp_93AssignVariableOp7assignvariableop_93_adamax_batchnormalization_4_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94¾
AssignVariableOp_94AssignVariableOp6assignvariableop_94_adamax_batchnormalization_4_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95¿
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adamax_batchnormalization_5_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96¾
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adamax_batchnormalization_5_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97·
AssignVariableOp_97AssignVariableOp/assignvariableop_97_adamax_dense_final_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98µ
AssignVariableOp_98AssignVariableOp-assignvariableop_98_adamax_dense_final_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_989
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpà
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_99h
Identity_100IdentityIdentity_99:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_100È
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_100Identity_100:output:0*Ý
_input_shapesË
È: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Å	
Ó
4__inference_batchNormalization_3_layer_call_fn_52148

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_489472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
E
)__inference_dropout_6_layer_call_fn_52487

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_496682
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
F
*__inference_maxPool_2b_layer_call_fn_51881

inputs
identityé
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_487202
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
I
-__inference_globAvgPool_5_layer_call_fn_52385

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_496522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
	
,__inference_sequential_5_layer_call_fn_49789
conv_1_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallconv_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: *H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_497092
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameconv_1_input
Å	
Ó
4__inference_batchNormalization_4_layer_call_fn_52337

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_490952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
Ý(
 __inference__wrapped_model_48415
conv_1_inputL
2sequential_5_conv_1_conv2d_readvariableop_resource:A
3sequential_5_conv_1_biasadd_readvariableop_resource:G
9sequential_5_batchnormalization_1_readvariableop_resource:I
;sequential_5_batchnormalization_1_readvariableop_1_resource:X
Jsequential_5_batchnormalization_1_fusedbatchnormv3_readvariableop_resource:Z
Lsequential_5_batchnormalization_1_fusedbatchnormv3_readvariableop_1_resource:M
3sequential_5_conv_1b_conv2d_readvariableop_resource: B
4sequential_5_conv_1b_biasadd_readvariableop_resource: H
:sequential_5_batchnormalization_1b_readvariableop_resource: J
<sequential_5_batchnormalization_1b_readvariableop_1_resource: Y
Ksequential_5_batchnormalization_1b_fusedbatchnormv3_readvariableop_resource: [
Msequential_5_batchnormalization_1b_fusedbatchnormv3_readvariableop_1_resource: L
2sequential_5_conv_2_conv2d_readvariableop_resource: @A
3sequential_5_conv_2_biasadd_readvariableop_resource:@M
3sequential_5_conv_2b_conv2d_readvariableop_resource:@@B
4sequential_5_conv_2b_biasadd_readvariableop_resource:@H
:sequential_5_batchnormalization_2b_readvariableop_resource:@J
<sequential_5_batchnormalization_2b_readvariableop_1_resource:@Y
Ksequential_5_batchnormalization_2b_fusedbatchnormv3_readvariableop_resource:@[
Msequential_5_batchnormalization_2b_fusedbatchnormv3_readvariableop_1_resource:@M
2sequential_5_conv_3_conv2d_readvariableop_resource:@B
3sequential_5_conv_3_biasadd_readvariableop_resource:	H
9sequential_5_batchnormalization_3_readvariableop_resource:	J
;sequential_5_batchnormalization_3_readvariableop_1_resource:	Y
Jsequential_5_batchnormalization_3_fusedbatchnormv3_readvariableop_resource:	[
Lsequential_5_batchnormalization_3_fusedbatchnormv3_readvariableop_1_resource:	N
2sequential_5_conv_4_conv2d_readvariableop_resource:B
3sequential_5_conv_4_biasadd_readvariableop_resource:	H
9sequential_5_batchnormalization_4_readvariableop_resource:	J
;sequential_5_batchnormalization_4_readvariableop_1_resource:	Y
Jsequential_5_batchnormalization_4_fusedbatchnormv3_readvariableop_resource:	[
Lsequential_5_batchnormalization_4_fusedbatchnormv3_readvariableop_1_resource:	R
Csequential_5_batchnormalization_5_batchnorm_readvariableop_resource:	V
Gsequential_5_batchnormalization_5_batchnorm_mul_readvariableop_resource:	T
Esequential_5_batchnormalization_5_batchnorm_readvariableop_1_resource:	T
Esequential_5_batchnormalization_5_batchnorm_readvariableop_2_resource:	J
7sequential_5_dense_final_matmul_readvariableop_resource:	F
8sequential_5_dense_final_biasadd_readvariableop_resource:
identity¢Asequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp¢Csequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1¢0sequential_5/batchNormalization_1/ReadVariableOp¢2sequential_5/batchNormalization_1/ReadVariableOp_1¢Bsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp¢Dsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1¢1sequential_5/batchNormalization_1b/ReadVariableOp¢3sequential_5/batchNormalization_1b/ReadVariableOp_1¢Bsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp¢Dsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1¢1sequential_5/batchNormalization_2b/ReadVariableOp¢3sequential_5/batchNormalization_2b/ReadVariableOp_1¢Asequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp¢Csequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1¢0sequential_5/batchNormalization_3/ReadVariableOp¢2sequential_5/batchNormalization_3/ReadVariableOp_1¢Asequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp¢Csequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1¢0sequential_5/batchNormalization_4/ReadVariableOp¢2sequential_5/batchNormalization_4/ReadVariableOp_1¢:sequential_5/batchNormalization_5/batchnorm/ReadVariableOp¢<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_1¢<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_2¢>sequential_5/batchNormalization_5/batchnorm/mul/ReadVariableOp¢*sequential_5/conv_1/BiasAdd/ReadVariableOp¢)sequential_5/conv_1/Conv2D/ReadVariableOp¢+sequential_5/conv_1b/BiasAdd/ReadVariableOp¢*sequential_5/conv_1b/Conv2D/ReadVariableOp¢*sequential_5/conv_2/BiasAdd/ReadVariableOp¢)sequential_5/conv_2/Conv2D/ReadVariableOp¢+sequential_5/conv_2b/BiasAdd/ReadVariableOp¢*sequential_5/conv_2b/Conv2D/ReadVariableOp¢*sequential_5/conv_3/BiasAdd/ReadVariableOp¢)sequential_5/conv_3/Conv2D/ReadVariableOp¢*sequential_5/conv_4/BiasAdd/ReadVariableOp¢)sequential_5/conv_4/Conv2D/ReadVariableOp¢/sequential_5/dense_final/BiasAdd/ReadVariableOp¢.sequential_5/dense_final/MatMul/ReadVariableOpÑ
)sequential_5/conv_1/Conv2D/ReadVariableOpReadVariableOp2sequential_5_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential_5/conv_1/Conv2D/ReadVariableOpå
sequential_5/conv_1/Conv2DConv2Dconv_1_input1sequential_5/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2
sequential_5/conv_1/Conv2DÈ
*sequential_5/conv_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_5_conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential_5/conv_1/BiasAdd/ReadVariableOpØ
sequential_5/conv_1/BiasAddBiasAdd#sequential_5/conv_1/Conv2D:output:02sequential_5/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
sequential_5/conv_1/BiasAdd
sequential_5/conv_1/ReluRelu$sequential_5/conv_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
sequential_5/conv_1/ReluÚ
0sequential_5/batchNormalization_1/ReadVariableOpReadVariableOp9sequential_5_batchnormalization_1_readvariableop_resource*
_output_shapes
:*
dtype022
0sequential_5/batchNormalization_1/ReadVariableOpà
2sequential_5/batchNormalization_1/ReadVariableOp_1ReadVariableOp;sequential_5_batchnormalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype024
2sequential_5/batchNormalization_1/ReadVariableOp_1
Asequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJsequential_5_batchnormalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02C
Asequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp
Csequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLsequential_5_batchnormalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02E
Csequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1¶
2sequential_5/batchNormalization_1/FusedBatchNormV3FusedBatchNormV3&sequential_5/conv_1/Relu:activations:08sequential_5/batchNormalization_1/ReadVariableOp:value:0:sequential_5/batchNormalization_1/ReadVariableOp_1:value:0Isequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp:value:0Ksequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 24
2sequential_5/batchNormalization_1/FusedBatchNormV3Ô
*sequential_5/conv_1b/Conv2D/ReadVariableOpReadVariableOp3sequential_5_conv_1b_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*sequential_5/conv_1b/Conv2D/ReadVariableOp
sequential_5/conv_1b/Conv2DConv2D6sequential_5/batchNormalization_1/FusedBatchNormV3:y:02sequential_5/conv_1b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingSAME*
strides
2
sequential_5/conv_1b/Conv2DË
+sequential_5/conv_1b/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_conv_1b_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_5/conv_1b/BiasAdd/ReadVariableOpÜ
sequential_5/conv_1b/BiasAddBiasAdd$sequential_5/conv_1b/Conv2D:output:03sequential_5/conv_1b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
sequential_5/conv_1b/BiasAdd
sequential_5/conv_1b/ReluRelu%sequential_5/conv_1b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
sequential_5/conv_1b/Reluã
sequential_5/maxPool_1b/MaxPoolMaxPool'sequential_5/conv_1b/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *
ksize
*
paddingVALID*
strides
2!
sequential_5/maxPool_1b/MaxPoolÝ
1sequential_5/batchNormalization_1b/ReadVariableOpReadVariableOp:sequential_5_batchnormalization_1b_readvariableop_resource*
_output_shapes
: *
dtype023
1sequential_5/batchNormalization_1b/ReadVariableOpã
3sequential_5/batchNormalization_1b/ReadVariableOp_1ReadVariableOp<sequential_5_batchnormalization_1b_readvariableop_1_resource*
_output_shapes
: *
dtype025
3sequential_5/batchNormalization_1b/ReadVariableOp_1
Bsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batchnormalization_1b_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp
Dsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batchnormalization_1b_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1¾
3sequential_5/batchNormalization_1b/FusedBatchNormV3FusedBatchNormV3(sequential_5/maxPool_1b/MaxPool:output:09sequential_5/batchNormalization_1b/ReadVariableOp:value:0;sequential_5/batchNormalization_1b/ReadVariableOp_1:value:0Jsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
is_training( 25
3sequential_5/batchNormalization_1b/FusedBatchNormV3Ñ
)sequential_5/conv_2/Conv2D/ReadVariableOpReadVariableOp2sequential_5_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential_5/conv_2/Conv2D/ReadVariableOp
sequential_5/conv_2/Conv2DConv2D7sequential_5/batchNormalization_1b/FusedBatchNormV3:y:01sequential_5/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
sequential_5/conv_2/Conv2DÈ
*sequential_5/conv_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_5_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential_5/conv_2/BiasAdd/ReadVariableOpØ
sequential_5/conv_2/BiasAddBiasAdd#sequential_5/conv_2/Conv2D:output:02sequential_5/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
sequential_5/conv_2/BiasAdd
sequential_5/conv_2/ReluRelu$sequential_5/conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
sequential_5/conv_2/Reluà
sequential_5/maxPool_2/MaxPoolMaxPool&sequential_5/conv_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2 
sequential_5/maxPool_2/MaxPool±
sequential_5/dropout_2/IdentityIdentity'sequential_5/maxPool_2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_5/dropout_2/IdentityÔ
*sequential_5/conv_2b/Conv2D/ReadVariableOpReadVariableOp3sequential_5_conv_2b_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*sequential_5/conv_2b/Conv2D/ReadVariableOp
sequential_5/conv_2b/Conv2DConv2D(sequential_5/dropout_2/Identity:output:02sequential_5/conv_2b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
sequential_5/conv_2b/Conv2DË
+sequential_5/conv_2b/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_conv_2b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_5/conv_2b/BiasAdd/ReadVariableOpÜ
sequential_5/conv_2b/BiasAddBiasAdd$sequential_5/conv_2b/Conv2D:output:03sequential_5/conv_2b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_5/conv_2b/BiasAdd
sequential_5/conv_2b/ReluRelu%sequential_5/conv_2b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_5/conv_2b/Reluã
sequential_5/maxPool_2b/MaxPoolMaxPool'sequential_5/conv_2b/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2!
sequential_5/maxPool_2b/MaxPoolÝ
1sequential_5/batchNormalization_2b/ReadVariableOpReadVariableOp:sequential_5_batchnormalization_2b_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_5/batchNormalization_2b/ReadVariableOpã
3sequential_5/batchNormalization_2b/ReadVariableOp_1ReadVariableOp<sequential_5_batchnormalization_2b_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3sequential_5/batchNormalization_2b/ReadVariableOp_1
Bsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batchnormalization_2b_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp
Dsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batchnormalization_2b_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1¾
3sequential_5/batchNormalization_2b/FusedBatchNormV3FusedBatchNormV3(sequential_5/maxPool_2b/MaxPool:output:09sequential_5/batchNormalization_2b/ReadVariableOp:value:0;sequential_5/batchNormalization_2b/ReadVariableOp_1:value:0Jsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 25
3sequential_5/batchNormalization_2b/FusedBatchNormV3Ò
)sequential_5/conv_3/Conv2D/ReadVariableOpReadVariableOp2sequential_5_conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02+
)sequential_5/conv_3/Conv2D/ReadVariableOp
sequential_5/conv_3/Conv2DConv2D7sequential_5/batchNormalization_2b/FusedBatchNormV3:y:01sequential_5/conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
sequential_5/conv_3/Conv2DÉ
*sequential_5/conv_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_5_conv_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*sequential_5/conv_3/BiasAdd/ReadVariableOpÙ
sequential_5/conv_3/BiasAddBiasAdd#sequential_5/conv_3/Conv2D:output:02sequential_5/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/conv_3/BiasAdd
sequential_5/conv_3/ReluRelu$sequential_5/conv_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/conv_3/Reluá
sequential_5/maxPool_3/MaxPoolMaxPool&sequential_5/conv_3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2 
sequential_5/maxPool_3/MaxPoolÛ
0sequential_5/batchNormalization_3/ReadVariableOpReadVariableOp9sequential_5_batchnormalization_3_readvariableop_resource*
_output_shapes	
:*
dtype022
0sequential_5/batchNormalization_3/ReadVariableOpá
2sequential_5/batchNormalization_3/ReadVariableOp_1ReadVariableOp;sequential_5_batchnormalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2sequential_5/batchNormalization_3/ReadVariableOp_1
Asequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJsequential_5_batchnormalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
Asequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp
Csequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLsequential_5_batchnormalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
Csequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1¼
2sequential_5/batchNormalization_3/FusedBatchNormV3FusedBatchNormV3'sequential_5/maxPool_3/MaxPool:output:08sequential_5/batchNormalization_3/ReadVariableOp:value:0:sequential_5/batchNormalization_3/ReadVariableOp_1:value:0Isequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp:value:0Ksequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 24
2sequential_5/batchNormalization_3/FusedBatchNormV3Ó
)sequential_5/conv_4/Conv2D/ReadVariableOpReadVariableOp2sequential_5_conv_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)sequential_5/conv_4/Conv2D/ReadVariableOp
sequential_5/conv_4/Conv2DConv2D6sequential_5/batchNormalization_3/FusedBatchNormV3:y:01sequential_5/conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
sequential_5/conv_4/Conv2DÉ
*sequential_5/conv_4/BiasAdd/ReadVariableOpReadVariableOp3sequential_5_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*sequential_5/conv_4/BiasAdd/ReadVariableOpÙ
sequential_5/conv_4/BiasAddBiasAdd#sequential_5/conv_4/Conv2D:output:02sequential_5/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/conv_4/BiasAdd
sequential_5/conv_4/ReluRelu$sequential_5/conv_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/conv_4/Reluá
sequential_5/maxPool_4/MaxPoolMaxPool&sequential_5/conv_4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2 
sequential_5/maxPool_4/MaxPoolÛ
0sequential_5/batchNormalization_4/ReadVariableOpReadVariableOp9sequential_5_batchnormalization_4_readvariableop_resource*
_output_shapes	
:*
dtype022
0sequential_5/batchNormalization_4/ReadVariableOpá
2sequential_5/batchNormalization_4/ReadVariableOp_1ReadVariableOp;sequential_5_batchnormalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype024
2sequential_5/batchNormalization_4/ReadVariableOp_1
Asequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJsequential_5_batchnormalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02C
Asequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp
Csequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLsequential_5_batchnormalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02E
Csequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1¼
2sequential_5/batchNormalization_4/FusedBatchNormV3FusedBatchNormV3'sequential_5/maxPool_4/MaxPool:output:08sequential_5/batchNormalization_4/ReadVariableOp:value:0:sequential_5/batchNormalization_4/ReadVariableOp_1:value:0Isequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp:value:0Ksequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 24
2sequential_5/batchNormalization_4/FusedBatchNormV3·
1sequential_5/globAvgPool_5/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1sequential_5/globAvgPool_5/Mean/reduction_indicesñ
sequential_5/globAvgPool_5/MeanMean6sequential_5/batchNormalization_4/FusedBatchNormV3:y:0:sequential_5/globAvgPool_5/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_5/globAvgPool_5/Meanù
:sequential_5/batchNormalization_5/batchnorm/ReadVariableOpReadVariableOpCsequential_5_batchnormalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02<
:sequential_5/batchNormalization_5/batchnorm/ReadVariableOp«
1sequential_5/batchNormalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:23
1sequential_5/batchNormalization_5/batchnorm/add/y
/sequential_5/batchNormalization_5/batchnorm/addAddV2Bsequential_5/batchNormalization_5/batchnorm/ReadVariableOp:value:0:sequential_5/batchNormalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:21
/sequential_5/batchNormalization_5/batchnorm/addÊ
1sequential_5/batchNormalization_5/batchnorm/RsqrtRsqrt3sequential_5/batchNormalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:23
1sequential_5/batchNormalization_5/batchnorm/Rsqrt
>sequential_5/batchNormalization_5/batchnorm/mul/ReadVariableOpReadVariableOpGsequential_5_batchnormalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02@
>sequential_5/batchNormalization_5/batchnorm/mul/ReadVariableOp
/sequential_5/batchNormalization_5/batchnorm/mulMul5sequential_5/batchNormalization_5/batchnorm/Rsqrt:y:0Fsequential_5/batchNormalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:21
/sequential_5/batchNormalization_5/batchnorm/mulÿ
1sequential_5/batchNormalization_5/batchnorm/mul_1Mul(sequential_5/globAvgPool_5/Mean:output:03sequential_5/batchNormalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_5/batchNormalization_5/batchnorm/mul_1ÿ
<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_1ReadVariableOpEsequential_5_batchnormalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02>
<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_1
1sequential_5/batchNormalization_5/batchnorm/mul_2MulDsequential_5/batchNormalization_5/batchnorm/ReadVariableOp_1:value:03sequential_5/batchNormalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:23
1sequential_5/batchNormalization_5/batchnorm/mul_2ÿ
<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_2ReadVariableOpEsequential_5_batchnormalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02>
<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_2
/sequential_5/batchNormalization_5/batchnorm/subSubDsequential_5/batchNormalization_5/batchnorm/ReadVariableOp_2:value:05sequential_5/batchNormalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:21
/sequential_5/batchNormalization_5/batchnorm/sub
1sequential_5/batchNormalization_5/batchnorm/add_1AddV25sequential_5/batchNormalization_5/batchnorm/mul_1:z:03sequential_5/batchNormalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_5/batchNormalization_5/batchnorm/add_1¸
sequential_5/dropout_6/IdentityIdentity5sequential_5/batchNormalization_5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_5/dropout_6/IdentityÙ
.sequential_5/dense_final/MatMul/ReadVariableOpReadVariableOp7sequential_5_dense_final_matmul_readvariableop_resource*
_output_shapes
:	*
dtype020
.sequential_5/dense_final/MatMul/ReadVariableOpà
sequential_5/dense_final/MatMulMatMul(sequential_5/dropout_6/Identity:output:06sequential_5/dense_final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_5/dense_final/MatMul×
/sequential_5/dense_final/BiasAdd/ReadVariableOpReadVariableOp8sequential_5_dense_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_5/dense_final/BiasAdd/ReadVariableOpå
 sequential_5/dense_final/BiasAddBiasAdd)sequential_5/dense_final/MatMul:product:07sequential_5/dense_final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_5/dense_final/BiasAdd¬
 sequential_5/dense_final/SoftmaxSoftmax)sequential_5/dense_final/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_5/dense_final/SoftmaxÒ
3sequential_5/dense_final/ActivityRegularizer/SquareSquare*sequential_5/dense_final/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3sequential_5/dense_final/ActivityRegularizer/Square¹
2sequential_5/dense_final/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2sequential_5/dense_final/ActivityRegularizer/Const
0sequential_5/dense_final/ActivityRegularizer/SumSum7sequential_5/dense_final/ActivityRegularizer/Square:y:0;sequential_5/dense_final/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_5/dense_final/ActivityRegularizer/Sum­
2sequential_5/dense_final/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£<24
2sequential_5/dense_final/ActivityRegularizer/mul/x
0sequential_5/dense_final/ActivityRegularizer/mulMul;sequential_5/dense_final/ActivityRegularizer/mul/x:output:09sequential_5/dense_final/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_5/dense_final/ActivityRegularizer/mulÂ
2sequential_5/dense_final/ActivityRegularizer/ShapeShape*sequential_5/dense_final/Softmax:softmax:0*
T0*
_output_shapes
:24
2sequential_5/dense_final/ActivityRegularizer/ShapeÎ
@sequential_5/dense_final/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_5/dense_final/ActivityRegularizer/strided_slice/stackÒ
Bsequential_5/dense_final/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_5/dense_final/ActivityRegularizer/strided_slice/stack_1Ò
Bsequential_5/dense_final/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_5/dense_final/ActivityRegularizer/strided_slice/stack_2ð
:sequential_5/dense_final/ActivityRegularizer/strided_sliceStridedSlice;sequential_5/dense_final/ActivityRegularizer/Shape:output:0Isequential_5/dense_final/ActivityRegularizer/strided_slice/stack:output:0Ksequential_5/dense_final/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_5/dense_final/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_5/dense_final/ActivityRegularizer/strided_sliceã
1sequential_5/dense_final/ActivityRegularizer/CastCastCsequential_5/dense_final/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_5/dense_final/ActivityRegularizer/Cast
4sequential_5/dense_final/ActivityRegularizer/truedivRealDiv4sequential_5/dense_final/ActivityRegularizer/mul:z:05sequential_5/dense_final/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_5/dense_final/ActivityRegularizer/truediv
IdentityIdentity*sequential_5/dense_final/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOpB^sequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOpD^sequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp_11^sequential_5/batchNormalization_1/ReadVariableOp3^sequential_5/batchNormalization_1/ReadVariableOp_1C^sequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOpE^sequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batchNormalization_1b/ReadVariableOp4^sequential_5/batchNormalization_1b/ReadVariableOp_1C^sequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOpE^sequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_12^sequential_5/batchNormalization_2b/ReadVariableOp4^sequential_5/batchNormalization_2b/ReadVariableOp_1B^sequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOpD^sequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp_11^sequential_5/batchNormalization_3/ReadVariableOp3^sequential_5/batchNormalization_3/ReadVariableOp_1B^sequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOpD^sequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp_11^sequential_5/batchNormalization_4/ReadVariableOp3^sequential_5/batchNormalization_4/ReadVariableOp_1;^sequential_5/batchNormalization_5/batchnorm/ReadVariableOp=^sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_1=^sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_2?^sequential_5/batchNormalization_5/batchnorm/mul/ReadVariableOp+^sequential_5/conv_1/BiasAdd/ReadVariableOp*^sequential_5/conv_1/Conv2D/ReadVariableOp,^sequential_5/conv_1b/BiasAdd/ReadVariableOp+^sequential_5/conv_1b/Conv2D/ReadVariableOp+^sequential_5/conv_2/BiasAdd/ReadVariableOp*^sequential_5/conv_2/Conv2D/ReadVariableOp,^sequential_5/conv_2b/BiasAdd/ReadVariableOp+^sequential_5/conv_2b/Conv2D/ReadVariableOp+^sequential_5/conv_3/BiasAdd/ReadVariableOp*^sequential_5/conv_3/Conv2D/ReadVariableOp+^sequential_5/conv_4/BiasAdd/ReadVariableOp*^sequential_5/conv_4/Conv2D/ReadVariableOp0^sequential_5/dense_final/BiasAdd/ReadVariableOp/^sequential_5/dense_final/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Asequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOpAsequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp2
Csequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1Csequential_5/batchNormalization_1/FusedBatchNormV3/ReadVariableOp_12d
0sequential_5/batchNormalization_1/ReadVariableOp0sequential_5/batchNormalization_1/ReadVariableOp2h
2sequential_5/batchNormalization_1/ReadVariableOp_12sequential_5/batchNormalization_1/ReadVariableOp_12
Bsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOpBsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp2
Dsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batchNormalization_1b/ReadVariableOp1sequential_5/batchNormalization_1b/ReadVariableOp2j
3sequential_5/batchNormalization_1b/ReadVariableOp_13sequential_5/batchNormalization_1b/ReadVariableOp_12
Bsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOpBsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp2
Dsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1Dsequential_5/batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_12f
1sequential_5/batchNormalization_2b/ReadVariableOp1sequential_5/batchNormalization_2b/ReadVariableOp2j
3sequential_5/batchNormalization_2b/ReadVariableOp_13sequential_5/batchNormalization_2b/ReadVariableOp_12
Asequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOpAsequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp2
Csequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1Csequential_5/batchNormalization_3/FusedBatchNormV3/ReadVariableOp_12d
0sequential_5/batchNormalization_3/ReadVariableOp0sequential_5/batchNormalization_3/ReadVariableOp2h
2sequential_5/batchNormalization_3/ReadVariableOp_12sequential_5/batchNormalization_3/ReadVariableOp_12
Asequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOpAsequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp2
Csequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1Csequential_5/batchNormalization_4/FusedBatchNormV3/ReadVariableOp_12d
0sequential_5/batchNormalization_4/ReadVariableOp0sequential_5/batchNormalization_4/ReadVariableOp2h
2sequential_5/batchNormalization_4/ReadVariableOp_12sequential_5/batchNormalization_4/ReadVariableOp_12x
:sequential_5/batchNormalization_5/batchnorm/ReadVariableOp:sequential_5/batchNormalization_5/batchnorm/ReadVariableOp2|
<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_1<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_12|
<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_2<sequential_5/batchNormalization_5/batchnorm/ReadVariableOp_22
>sequential_5/batchNormalization_5/batchnorm/mul/ReadVariableOp>sequential_5/batchNormalization_5/batchnorm/mul/ReadVariableOp2X
*sequential_5/conv_1/BiasAdd/ReadVariableOp*sequential_5/conv_1/BiasAdd/ReadVariableOp2V
)sequential_5/conv_1/Conv2D/ReadVariableOp)sequential_5/conv_1/Conv2D/ReadVariableOp2Z
+sequential_5/conv_1b/BiasAdd/ReadVariableOp+sequential_5/conv_1b/BiasAdd/ReadVariableOp2X
*sequential_5/conv_1b/Conv2D/ReadVariableOp*sequential_5/conv_1b/Conv2D/ReadVariableOp2X
*sequential_5/conv_2/BiasAdd/ReadVariableOp*sequential_5/conv_2/BiasAdd/ReadVariableOp2V
)sequential_5/conv_2/Conv2D/ReadVariableOp)sequential_5/conv_2/Conv2D/ReadVariableOp2Z
+sequential_5/conv_2b/BiasAdd/ReadVariableOp+sequential_5/conv_2b/BiasAdd/ReadVariableOp2X
*sequential_5/conv_2b/Conv2D/ReadVariableOp*sequential_5/conv_2b/Conv2D/ReadVariableOp2X
*sequential_5/conv_3/BiasAdd/ReadVariableOp*sequential_5/conv_3/BiasAdd/ReadVariableOp2V
)sequential_5/conv_3/Conv2D/ReadVariableOp)sequential_5/conv_3/Conv2D/ReadVariableOp2X
*sequential_5/conv_4/BiasAdd/ReadVariableOp*sequential_5/conv_4/BiasAdd/ReadVariableOp2V
)sequential_5/conv_4/Conv2D/ReadVariableOp)sequential_5/conv_4/Conv2D/ReadVariableOp2b
/sequential_5/dense_final/BiasAdd/ReadVariableOp/sequential_5/dense_final/BiasAdd/ReadVariableOp2`
.sequential_5/dense_final/MatMul/ReadVariableOp.sequential_5/dense_final/MatMul/ReadVariableOp:] Y
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
&
_user_specified_nameconv_1_input
¥
a
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_51615

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
a
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_51876

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51684

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00 : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs


P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_49445

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00 : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs
Ç	
Ó
4__inference_batchNormalization_4_layer_call_fn_52324

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_490512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
`
D__inference_maxPool_2_layer_call_and_return_conditional_losses_48698

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¿
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51958

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ì
E
)__inference_maxPool_3_layer_call_fn_52045

inputs
identityè
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxPool_3_layer_call_and_return_conditional_losses_488682
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

a
B__inference_noise_3_layer_call_and_return_conditional_losses_52189

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *>2
random_normal/stddevØ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed¡¶*
seed2ÆÛÙ2$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_normali
addAddV2inputsrandom_normal:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


'__inference_conv_1b_layer_call_fn_51610

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_1b_layer_call_and_return_conditional_losses_494162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ``: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ç
ú
A__inference_conv_1_layer_call_and_return_conditional_losses_49372

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ``: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
Û
C
'__inference_noise_1_layer_call_fn_51841

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_noise_1_layer_call_and_return_conditional_losses_494892
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â

P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_48585

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


&__inference_conv_3_layer_call_fn_52030

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_495522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
õ
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_49668

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
Ð
5__inference_batchNormalization_1b_layer_call_fn_51741

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_494452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00 : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs


'__inference_conv_2b_layer_call_fn_51866

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv_2b_layer_call_and_return_conditional_losses_495022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
û

+__inference_dense_final_layer_call_fn_52518

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_final_layer_call_and_return_conditional_losses_496872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
I
2__inference_dense_final_activity_regularizer_49354
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£<2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
¨

O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52293

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
`
D__inference_maxPool_3_layer_call_and_return_conditional_losses_52035

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51520

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ï
ü
A__inference_conv_3_layer_call_and_return_conditional_losses_49552

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ
Ó
4__inference_batchNormalization_3_layer_call_fn_52161

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_495812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ*
ì
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_49263

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay¥
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOp
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/sub
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg/mul¿
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay«
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp¡
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/sub
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:2
AssignMovingAvg_1/mulÉ
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
íÄ
¿-
__inference__traced_save_52866
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop9
5savev2_batchnormalization_1_gamma_read_readvariableop8
4savev2_batchnormalization_1_beta_read_readvariableop?
;savev2_batchnormalization_1_moving_mean_read_readvariableopC
?savev2_batchnormalization_1_moving_variance_read_readvariableop-
)savev2_conv_1b_kernel_read_readvariableop+
'savev2_conv_1b_bias_read_readvariableop:
6savev2_batchnormalization_1b_gamma_read_readvariableop9
5savev2_batchnormalization_1b_beta_read_readvariableop@
<savev2_batchnormalization_1b_moving_mean_read_readvariableopD
@savev2_batchnormalization_1b_moving_variance_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop-
)savev2_conv_2b_kernel_read_readvariableop+
'savev2_conv_2b_bias_read_readvariableop:
6savev2_batchnormalization_2b_gamma_read_readvariableop9
5savev2_batchnormalization_2b_beta_read_readvariableop@
<savev2_batchnormalization_2b_moving_mean_read_readvariableopD
@savev2_batchnormalization_2b_moving_variance_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop9
5savev2_batchnormalization_3_gamma_read_readvariableop8
4savev2_batchnormalization_3_beta_read_readvariableop?
;savev2_batchnormalization_3_moving_mean_read_readvariableopC
?savev2_batchnormalization_3_moving_variance_read_readvariableop,
(savev2_conv_4_kernel_read_readvariableop*
&savev2_conv_4_bias_read_readvariableop9
5savev2_batchnormalization_4_gamma_read_readvariableop8
4savev2_batchnormalization_4_beta_read_readvariableop?
;savev2_batchnormalization_4_moving_mean_read_readvariableopC
?savev2_batchnormalization_4_moving_variance_read_readvariableop9
5savev2_batchnormalization_5_gamma_read_readvariableop8
4savev2_batchnormalization_5_beta_read_readvariableop?
;savev2_batchnormalization_5_moving_mean_read_readvariableopC
?savev2_batchnormalization_5_moving_variance_read_readvariableop1
-savev2_dense_final_kernel_read_readvariableop/
+savev2_dense_final_bias_read_readvariableop*
&savev2_adamax_iter_read_readvariableop	,
(savev2_adamax_beta_1_read_readvariableop,
(savev2_adamax_beta_2_read_readvariableop+
'savev2_adamax_decay_read_readvariableop3
/savev2_adamax_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adamax_conv_1_kernel_m_read_readvariableop3
/savev2_adamax_conv_1_bias_m_read_readvariableopB
>savev2_adamax_batchnormalization_1_gamma_m_read_readvariableopA
=savev2_adamax_batchnormalization_1_beta_m_read_readvariableop6
2savev2_adamax_conv_1b_kernel_m_read_readvariableop4
0savev2_adamax_conv_1b_bias_m_read_readvariableopC
?savev2_adamax_batchnormalization_1b_gamma_m_read_readvariableopB
>savev2_adamax_batchnormalization_1b_beta_m_read_readvariableop5
1savev2_adamax_conv_2_kernel_m_read_readvariableop3
/savev2_adamax_conv_2_bias_m_read_readvariableop6
2savev2_adamax_conv_2b_kernel_m_read_readvariableop4
0savev2_adamax_conv_2b_bias_m_read_readvariableopC
?savev2_adamax_batchnormalization_2b_gamma_m_read_readvariableopB
>savev2_adamax_batchnormalization_2b_beta_m_read_readvariableop5
1savev2_adamax_conv_3_kernel_m_read_readvariableop3
/savev2_adamax_conv_3_bias_m_read_readvariableopB
>savev2_adamax_batchnormalization_3_gamma_m_read_readvariableopA
=savev2_adamax_batchnormalization_3_beta_m_read_readvariableop5
1savev2_adamax_conv_4_kernel_m_read_readvariableop3
/savev2_adamax_conv_4_bias_m_read_readvariableopB
>savev2_adamax_batchnormalization_4_gamma_m_read_readvariableopA
=savev2_adamax_batchnormalization_4_beta_m_read_readvariableopB
>savev2_adamax_batchnormalization_5_gamma_m_read_readvariableopA
=savev2_adamax_batchnormalization_5_beta_m_read_readvariableop:
6savev2_adamax_dense_final_kernel_m_read_readvariableop8
4savev2_adamax_dense_final_bias_m_read_readvariableop5
1savev2_adamax_conv_1_kernel_v_read_readvariableop3
/savev2_adamax_conv_1_bias_v_read_readvariableopB
>savev2_adamax_batchnormalization_1_gamma_v_read_readvariableopA
=savev2_adamax_batchnormalization_1_beta_v_read_readvariableop6
2savev2_adamax_conv_1b_kernel_v_read_readvariableop4
0savev2_adamax_conv_1b_bias_v_read_readvariableopC
?savev2_adamax_batchnormalization_1b_gamma_v_read_readvariableopB
>savev2_adamax_batchnormalization_1b_beta_v_read_readvariableop5
1savev2_adamax_conv_2_kernel_v_read_readvariableop3
/savev2_adamax_conv_2_bias_v_read_readvariableop6
2savev2_adamax_conv_2b_kernel_v_read_readvariableop4
0savev2_adamax_conv_2b_bias_v_read_readvariableopC
?savev2_adamax_batchnormalization_2b_gamma_v_read_readvariableopB
>savev2_adamax_batchnormalization_2b_beta_v_read_readvariableop5
1savev2_adamax_conv_3_kernel_v_read_readvariableop3
/savev2_adamax_conv_3_bias_v_read_readvariableopB
>savev2_adamax_batchnormalization_3_gamma_v_read_readvariableopA
=savev2_adamax_batchnormalization_3_beta_v_read_readvariableop5
1savev2_adamax_conv_4_kernel_v_read_readvariableop3
/savev2_adamax_conv_4_bias_v_read_readvariableopB
>savev2_adamax_batchnormalization_4_gamma_v_read_readvariableopA
=savev2_adamax_batchnormalization_4_beta_v_read_readvariableopB
>savev2_adamax_batchnormalization_5_gamma_v_read_readvariableopA
=savev2_adamax_batchnormalization_5_beta_v_read_readvariableop:
6savev2_adamax_dense_final_kernel_v_read_readvariableop8
4savev2_adamax_dense_final_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÈ7
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ú6
valueÐ6BÍ6dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÓ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ý
valueÓBÐdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_sliceså+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop5savev2_batchnormalization_1_gamma_read_readvariableop4savev2_batchnormalization_1_beta_read_readvariableop;savev2_batchnormalization_1_moving_mean_read_readvariableop?savev2_batchnormalization_1_moving_variance_read_readvariableop)savev2_conv_1b_kernel_read_readvariableop'savev2_conv_1b_bias_read_readvariableop6savev2_batchnormalization_1b_gamma_read_readvariableop5savev2_batchnormalization_1b_beta_read_readvariableop<savev2_batchnormalization_1b_moving_mean_read_readvariableop@savev2_batchnormalization_1b_moving_variance_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop)savev2_conv_2b_kernel_read_readvariableop'savev2_conv_2b_bias_read_readvariableop6savev2_batchnormalization_2b_gamma_read_readvariableop5savev2_batchnormalization_2b_beta_read_readvariableop<savev2_batchnormalization_2b_moving_mean_read_readvariableop@savev2_batchnormalization_2b_moving_variance_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop5savev2_batchnormalization_3_gamma_read_readvariableop4savev2_batchnormalization_3_beta_read_readvariableop;savev2_batchnormalization_3_moving_mean_read_readvariableop?savev2_batchnormalization_3_moving_variance_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop5savev2_batchnormalization_4_gamma_read_readvariableop4savev2_batchnormalization_4_beta_read_readvariableop;savev2_batchnormalization_4_moving_mean_read_readvariableop?savev2_batchnormalization_4_moving_variance_read_readvariableop5savev2_batchnormalization_5_gamma_read_readvariableop4savev2_batchnormalization_5_beta_read_readvariableop;savev2_batchnormalization_5_moving_mean_read_readvariableop?savev2_batchnormalization_5_moving_variance_read_readvariableop-savev2_dense_final_kernel_read_readvariableop+savev2_dense_final_bias_read_readvariableop&savev2_adamax_iter_read_readvariableop(savev2_adamax_beta_1_read_readvariableop(savev2_adamax_beta_2_read_readvariableop'savev2_adamax_decay_read_readvariableop/savev2_adamax_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adamax_conv_1_kernel_m_read_readvariableop/savev2_adamax_conv_1_bias_m_read_readvariableop>savev2_adamax_batchnormalization_1_gamma_m_read_readvariableop=savev2_adamax_batchnormalization_1_beta_m_read_readvariableop2savev2_adamax_conv_1b_kernel_m_read_readvariableop0savev2_adamax_conv_1b_bias_m_read_readvariableop?savev2_adamax_batchnormalization_1b_gamma_m_read_readvariableop>savev2_adamax_batchnormalization_1b_beta_m_read_readvariableop1savev2_adamax_conv_2_kernel_m_read_readvariableop/savev2_adamax_conv_2_bias_m_read_readvariableop2savev2_adamax_conv_2b_kernel_m_read_readvariableop0savev2_adamax_conv_2b_bias_m_read_readvariableop?savev2_adamax_batchnormalization_2b_gamma_m_read_readvariableop>savev2_adamax_batchnormalization_2b_beta_m_read_readvariableop1savev2_adamax_conv_3_kernel_m_read_readvariableop/savev2_adamax_conv_3_bias_m_read_readvariableop>savev2_adamax_batchnormalization_3_gamma_m_read_readvariableop=savev2_adamax_batchnormalization_3_beta_m_read_readvariableop1savev2_adamax_conv_4_kernel_m_read_readvariableop/savev2_adamax_conv_4_bias_m_read_readvariableop>savev2_adamax_batchnormalization_4_gamma_m_read_readvariableop=savev2_adamax_batchnormalization_4_beta_m_read_readvariableop>savev2_adamax_batchnormalization_5_gamma_m_read_readvariableop=savev2_adamax_batchnormalization_5_beta_m_read_readvariableop6savev2_adamax_dense_final_kernel_m_read_readvariableop4savev2_adamax_dense_final_bias_m_read_readvariableop1savev2_adamax_conv_1_kernel_v_read_readvariableop/savev2_adamax_conv_1_bias_v_read_readvariableop>savev2_adamax_batchnormalization_1_gamma_v_read_readvariableop=savev2_adamax_batchnormalization_1_beta_v_read_readvariableop2savev2_adamax_conv_1b_kernel_v_read_readvariableop0savev2_adamax_conv_1b_bias_v_read_readvariableop?savev2_adamax_batchnormalization_1b_gamma_v_read_readvariableop>savev2_adamax_batchnormalization_1b_beta_v_read_readvariableop1savev2_adamax_conv_2_kernel_v_read_readvariableop/savev2_adamax_conv_2_bias_v_read_readvariableop2savev2_adamax_conv_2b_kernel_v_read_readvariableop0savev2_adamax_conv_2b_bias_v_read_readvariableop?savev2_adamax_batchnormalization_2b_gamma_v_read_readvariableop>savev2_adamax_batchnormalization_2b_beta_v_read_readvariableop1savev2_adamax_conv_3_kernel_v_read_readvariableop/savev2_adamax_conv_3_bias_v_read_readvariableop>savev2_adamax_batchnormalization_3_gamma_v_read_readvariableop=savev2_adamax_batchnormalization_3_beta_v_read_readvariableop1savev2_adamax_conv_4_kernel_v_read_readvariableop/savev2_adamax_conv_4_bias_v_read_readvariableop>savev2_adamax_batchnormalization_4_gamma_v_read_readvariableop=savev2_adamax_batchnormalization_4_beta_v_read_readvariableop>savev2_adamax_batchnormalization_5_gamma_v_read_readvariableop=savev2_adamax_batchnormalization_5_beta_v_read_readvariableop6savev2_adamax_dense_final_kernel_v_read_readvariableop4savev2_adamax_dense_final_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Õ
_input_shapesÃ
À: ::::::: : : : : : : @:@:@@:@:@:@:@:@:@::::::::::::::::	:: : : : : : : : : ::::: : : : : @:@:@@:@:@:@:@::::::::::	:::::: : : : : @:@:@@:@:@:@:@::::::::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::!$

_output_shapes	
::%%!

_output_shapes
:	: &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :,0(
&
_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
: @: 9

_output_shapes
:@:,:(
&
_output_shapes
:@@: ;

_output_shapes
:@: <

_output_shapes
:@: =

_output_shapes
:@:->)
'
_output_shapes
:@:!?

_output_shapes	
::!@

_output_shapes	
::!A

_output_shapes	
::.B*
(
_output_shapes
::!C

_output_shapes	
::!D

_output_shapes	
::!E

_output_shapes	
::!F

_output_shapes	
::!G

_output_shapes	
::%H!

_output_shapes
:	: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
:: L

_output_shapes
:: M

_output_shapes
::,N(
&
_output_shapes
: : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: :,R(
&
_output_shapes
: @: S

_output_shapes
:@:,T(
&
_output_shapes
:@@: U

_output_shapes
:@: V

_output_shapes
:@: W

_output_shapes
:@:-X)
'
_output_shapes
:@:!Y

_output_shapes	
::!Z

_output_shapes	
::![

_output_shapes	
::.\*
(
_output_shapes
::!]

_output_shapes	
::!^

_output_shapes	
::!_

_output_shapes	
::!`

_output_shapes	
::!a

_output_shapes	
::%b!

_output_shapes
:	: c

_output_shapes
::d

_output_shapes
: 
Ô

a
B__inference_noise_1_layer_call_and_return_conditional_losses_50057

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *>2
random_normal/stddev×
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*
seed¡¶*
seed2¼Æ2$
"random_normal/RandomStandardNormal³
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
random_normal/mul
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
random_normalh
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ý
Ó
4__inference_batchNormalization_3_layer_call_fn_52174

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_499532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á	
Ð
5__inference_batchNormalization_1b_layer_call_fn_51715

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_485852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ì
é!
G__inference_sequential_5_layer_call_and_return_conditional_losses_51073

inputs?
%conv_1_conv2d_readvariableop_resource:4
&conv_1_biasadd_readvariableop_resource::
,batchnormalization_1_readvariableop_resource:<
.batchnormalization_1_readvariableop_1_resource:K
=batchnormalization_1_fusedbatchnormv3_readvariableop_resource:M
?batchnormalization_1_fusedbatchnormv3_readvariableop_1_resource:@
&conv_1b_conv2d_readvariableop_resource: 5
'conv_1b_biasadd_readvariableop_resource: ;
-batchnormalization_1b_readvariableop_resource: =
/batchnormalization_1b_readvariableop_1_resource: L
>batchnormalization_1b_fusedbatchnormv3_readvariableop_resource: N
@batchnormalization_1b_fusedbatchnormv3_readvariableop_1_resource: ?
%conv_2_conv2d_readvariableop_resource: @4
&conv_2_biasadd_readvariableop_resource:@@
&conv_2b_conv2d_readvariableop_resource:@@5
'conv_2b_biasadd_readvariableop_resource:@;
-batchnormalization_2b_readvariableop_resource:@=
/batchnormalization_2b_readvariableop_1_resource:@L
>batchnormalization_2b_fusedbatchnormv3_readvariableop_resource:@N
@batchnormalization_2b_fusedbatchnormv3_readvariableop_1_resource:@@
%conv_3_conv2d_readvariableop_resource:@5
&conv_3_biasadd_readvariableop_resource:	;
,batchnormalization_3_readvariableop_resource:	=
.batchnormalization_3_readvariableop_1_resource:	L
=batchnormalization_3_fusedbatchnormv3_readvariableop_resource:	N
?batchnormalization_3_fusedbatchnormv3_readvariableop_1_resource:	A
%conv_4_conv2d_readvariableop_resource:5
&conv_4_biasadd_readvariableop_resource:	;
,batchnormalization_4_readvariableop_resource:	=
.batchnormalization_4_readvariableop_1_resource:	L
=batchnormalization_4_fusedbatchnormv3_readvariableop_resource:	N
?batchnormalization_4_fusedbatchnormv3_readvariableop_1_resource:	E
6batchnormalization_5_batchnorm_readvariableop_resource:	I
:batchnormalization_5_batchnorm_mul_readvariableop_resource:	G
8batchnormalization_5_batchnorm_readvariableop_1_resource:	G
8batchnormalization_5_batchnorm_readvariableop_2_resource:	=
*dense_final_matmul_readvariableop_resource:	9
+dense_final_biasadd_readvariableop_resource:
identity

identity_1¢4batchNormalization_1/FusedBatchNormV3/ReadVariableOp¢6batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1¢#batchNormalization_1/ReadVariableOp¢%batchNormalization_1/ReadVariableOp_1¢5batchNormalization_1b/FusedBatchNormV3/ReadVariableOp¢7batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1¢$batchNormalization_1b/ReadVariableOp¢&batchNormalization_1b/ReadVariableOp_1¢5batchNormalization_2b/FusedBatchNormV3/ReadVariableOp¢7batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1¢$batchNormalization_2b/ReadVariableOp¢&batchNormalization_2b/ReadVariableOp_1¢4batchNormalization_3/FusedBatchNormV3/ReadVariableOp¢6batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1¢#batchNormalization_3/ReadVariableOp¢%batchNormalization_3/ReadVariableOp_1¢4batchNormalization_4/FusedBatchNormV3/ReadVariableOp¢6batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1¢#batchNormalization_4/ReadVariableOp¢%batchNormalization_4/ReadVariableOp_1¢-batchNormalization_5/batchnorm/ReadVariableOp¢/batchNormalization_5/batchnorm/ReadVariableOp_1¢/batchNormalization_5/batchnorm/ReadVariableOp_2¢1batchNormalization_5/batchnorm/mul/ReadVariableOp¢conv_1/BiasAdd/ReadVariableOp¢conv_1/Conv2D/ReadVariableOp¢conv_1b/BiasAdd/ReadVariableOp¢conv_1b/Conv2D/ReadVariableOp¢conv_2/BiasAdd/ReadVariableOp¢conv_2/Conv2D/ReadVariableOp¢conv_2b/BiasAdd/ReadVariableOp¢conv_2b/Conv2D/ReadVariableOp¢conv_3/BiasAdd/ReadVariableOp¢conv_3/Conv2D/ReadVariableOp¢conv_4/BiasAdd/ReadVariableOp¢conv_4/Conv2D/ReadVariableOp¢"dense_final/BiasAdd/ReadVariableOp¢!dense_final/MatMul/ReadVariableOp¢4dense_final/kernel/Regularizer/Square/ReadVariableOpª
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_1/Conv2D/ReadVariableOp¸
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*
paddingSAME*
strides
2
conv_1/Conv2D¡
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp¤
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
conv_1/BiasAddu
conv_1/ReluReluconv_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2
conv_1/Relu³
#batchNormalization_1/ReadVariableOpReadVariableOp,batchnormalization_1_readvariableop_resource*
_output_shapes
:*
dtype02%
#batchNormalization_1/ReadVariableOp¹
%batchNormalization_1/ReadVariableOp_1ReadVariableOp.batchnormalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02'
%batchNormalization_1/ReadVariableOp_1æ
4batchNormalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp=batchnormalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype026
4batchNormalization_1/FusedBatchNormV3/ReadVariableOpì
6batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?batchnormalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype028
6batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1Û
%batchNormalization_1/FusedBatchNormV3FusedBatchNormV3conv_1/Relu:activations:0+batchNormalization_1/ReadVariableOp:value:0-batchNormalization_1/ReadVariableOp_1:value:0<batchNormalization_1/FusedBatchNormV3/ReadVariableOp:value:0>batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
is_training( 2'
%batchNormalization_1/FusedBatchNormV3­
conv_1b/Conv2D/ReadVariableOpReadVariableOp&conv_1b_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_1b/Conv2D/ReadVariableOpÞ
conv_1b/Conv2DConv2D)batchNormalization_1/FusedBatchNormV3:y:0%conv_1b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingSAME*
strides
2
conv_1b/Conv2D¤
conv_1b/BiasAdd/ReadVariableOpReadVariableOp'conv_1b_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
conv_1b/BiasAdd/ReadVariableOp¨
conv_1b/BiasAddBiasAddconv_1b/Conv2D:output:0&conv_1b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
conv_1b/BiasAddx
conv_1b/ReluReluconv_1b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
conv_1b/Relu¼
maxPool_1b/MaxPoolMaxPoolconv_1b/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *
ksize
*
paddingVALID*
strides
2
maxPool_1b/MaxPool¶
$batchNormalization_1b/ReadVariableOpReadVariableOp-batchnormalization_1b_readvariableop_resource*
_output_shapes
: *
dtype02&
$batchNormalization_1b/ReadVariableOp¼
&batchNormalization_1b/ReadVariableOp_1ReadVariableOp/batchnormalization_1b_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batchNormalization_1b/ReadVariableOp_1é
5batchNormalization_1b/FusedBatchNormV3/ReadVariableOpReadVariableOp>batchnormalization_1b_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batchNormalization_1b/FusedBatchNormV3/ReadVariableOpï
7batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batchnormalization_1b_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1ã
&batchNormalization_1b/FusedBatchNormV3FusedBatchNormV3maxPool_1b/MaxPool:output:0,batchNormalization_1b/ReadVariableOp:value:0.batchNormalization_1b/ReadVariableOp_1:value:0=batchNormalization_1b/FusedBatchNormV3/ReadVariableOp:value:0?batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
is_training( 2(
&batchNormalization_1b/FusedBatchNormV3ª
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_2/Conv2D/ReadVariableOpÜ
conv_2/Conv2DConv2D*batchNormalization_1b/FusedBatchNormV3:y:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
conv_2/Conv2D¡
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_2/BiasAdd/ReadVariableOp¤
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv_2/BiasAddu
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv_2/Relu¹
maxPool_2/MaxPoolMaxPoolconv_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
maxPool_2/MaxPool
dropout_2/IdentityIdentitymaxPool_2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_2/Identity­
conv_2b/Conv2D/ReadVariableOpReadVariableOp&conv_2b_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv_2b/Conv2D/ReadVariableOpÐ
conv_2b/Conv2DConv2Ddropout_2/Identity:output:0%conv_2b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv_2b/Conv2D¤
conv_2b/BiasAdd/ReadVariableOpReadVariableOp'conv_2b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
conv_2b/BiasAdd/ReadVariableOp¨
conv_2b/BiasAddBiasAddconv_2b/Conv2D:output:0&conv_2b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_2b/BiasAddx
conv_2b/ReluReluconv_2b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv_2b/Relu¼
maxPool_2b/MaxPoolMaxPoolconv_2b/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
maxPool_2b/MaxPool¶
$batchNormalization_2b/ReadVariableOpReadVariableOp-batchnormalization_2b_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batchNormalization_2b/ReadVariableOp¼
&batchNormalization_2b/ReadVariableOp_1ReadVariableOp/batchnormalization_2b_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batchNormalization_2b/ReadVariableOp_1é
5batchNormalization_2b/FusedBatchNormV3/ReadVariableOpReadVariableOp>batchnormalization_2b_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batchNormalization_2b/FusedBatchNormV3/ReadVariableOpï
7batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batchnormalization_2b_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1ã
&batchNormalization_2b/FusedBatchNormV3FusedBatchNormV3maxPool_2b/MaxPool:output:0,batchNormalization_2b/ReadVariableOp:value:0.batchNormalization_2b/ReadVariableOp_1:value:0=batchNormalization_2b/FusedBatchNormV3/ReadVariableOp:value:0?batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batchNormalization_2b/FusedBatchNormV3«
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_3/Conv2D/ReadVariableOpÝ
conv_3/Conv2DConv2D*batchNormalization_2b/FusedBatchNormV3:y:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_3/Conv2D¢
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_3/BiasAdd/ReadVariableOp¥
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_3/BiasAddv
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_3/Reluº
maxPool_3/MaxPoolMaxPoolconv_3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
maxPool_3/MaxPool´
#batchNormalization_3/ReadVariableOpReadVariableOp,batchnormalization_3_readvariableop_resource*
_output_shapes	
:*
dtype02%
#batchNormalization_3/ReadVariableOpº
%batchNormalization_3/ReadVariableOp_1ReadVariableOp.batchnormalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02'
%batchNormalization_3/ReadVariableOp_1ç
4batchNormalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp=batchnormalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype026
4batchNormalization_3/FusedBatchNormV3/ReadVariableOpí
6batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?batchnormalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1á
%batchNormalization_3/FusedBatchNormV3FusedBatchNormV3maxPool_3/MaxPool:output:0+batchNormalization_3/ReadVariableOp:value:0-batchNormalization_3/ReadVariableOp_1:value:0<batchNormalization_3/FusedBatchNormV3/ReadVariableOp:value:0>batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2'
%batchNormalization_3/FusedBatchNormV3¬
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOpÜ
conv_4/Conv2DConv2D)batchNormalization_3/FusedBatchNormV3:y:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_4/Conv2D¢
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_4/BiasAdd/ReadVariableOp¥
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_4/BiasAddv
conv_4/ReluReluconv_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv_4/Reluº
maxPool_4/MaxPoolMaxPoolconv_4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
maxPool_4/MaxPool´
#batchNormalization_4/ReadVariableOpReadVariableOp,batchnormalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02%
#batchNormalization_4/ReadVariableOpº
%batchNormalization_4/ReadVariableOp_1ReadVariableOp.batchnormalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02'
%batchNormalization_4/ReadVariableOp_1ç
4batchNormalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp=batchnormalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype026
4batchNormalization_4/FusedBatchNormV3/ReadVariableOpí
6batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?batchnormalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype028
6batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1á
%batchNormalization_4/FusedBatchNormV3FusedBatchNormV3maxPool_4/MaxPool:output:0+batchNormalization_4/ReadVariableOp:value:0-batchNormalization_4/ReadVariableOp_1:value:0<batchNormalization_4/FusedBatchNormV3/ReadVariableOp:value:0>batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2'
%batchNormalization_4/FusedBatchNormV3
$globAvgPool_5/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2&
$globAvgPool_5/Mean/reduction_indices½
globAvgPool_5/MeanMean)batchNormalization_4/FusedBatchNormV3:y:0-globAvgPool_5/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
globAvgPool_5/MeanÒ
-batchNormalization_5/batchnorm/ReadVariableOpReadVariableOp6batchnormalization_5_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02/
-batchNormalization_5/batchnorm/ReadVariableOp
$batchNormalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$batchNormalization_5/batchnorm/add/yÝ
"batchNormalization_5/batchnorm/addAddV25batchNormalization_5/batchnorm/ReadVariableOp:value:0-batchNormalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2$
"batchNormalization_5/batchnorm/add£
$batchNormalization_5/batchnorm/RsqrtRsqrt&batchNormalization_5/batchnorm/add:z:0*
T0*
_output_shapes	
:2&
$batchNormalization_5/batchnorm/RsqrtÞ
1batchNormalization_5/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype023
1batchNormalization_5/batchnorm/mul/ReadVariableOpÚ
"batchNormalization_5/batchnorm/mulMul(batchNormalization_5/batchnorm/Rsqrt:y:09batchNormalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2$
"batchNormalization_5/batchnorm/mulË
$batchNormalization_5/batchnorm/mul_1MulglobAvgPool_5/Mean:output:0&batchNormalization_5/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$batchNormalization_5/batchnorm/mul_1Ø
/batchNormalization_5/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormalization_5_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype021
/batchNormalization_5/batchnorm/ReadVariableOp_1Ú
$batchNormalization_5/batchnorm/mul_2Mul7batchNormalization_5/batchnorm/ReadVariableOp_1:value:0&batchNormalization_5/batchnorm/mul:z:0*
T0*
_output_shapes	
:2&
$batchNormalization_5/batchnorm/mul_2Ø
/batchNormalization_5/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormalization_5_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype021
/batchNormalization_5/batchnorm/ReadVariableOp_2Ø
"batchNormalization_5/batchnorm/subSub7batchNormalization_5/batchnorm/ReadVariableOp_2:value:0(batchNormalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2$
"batchNormalization_5/batchnorm/subÚ
$batchNormalization_5/batchnorm/add_1AddV2(batchNormalization_5/batchnorm/mul_1:z:0&batchNormalization_5/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$batchNormalization_5/batchnorm/add_1
dropout_6/IdentityIdentity(batchNormalization_5/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_6/Identity²
!dense_final/MatMul/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02#
!dense_final/MatMul/ReadVariableOp¬
dense_final/MatMulMatMuldropout_6/Identity:output:0)dense_final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_final/MatMul°
"dense_final/BiasAdd/ReadVariableOpReadVariableOp+dense_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dense_final/BiasAdd/ReadVariableOp±
dense_final/BiasAddBiasAdddense_final/MatMul:product:0*dense_final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_final/BiasAdd
dense_final/SoftmaxSoftmaxdense_final/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_final/Softmax«
&dense_final/ActivityRegularizer/SquareSquaredense_final/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&dense_final/ActivityRegularizer/Square
%dense_final/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%dense_final/ActivityRegularizer/ConstÎ
#dense_final/ActivityRegularizer/SumSum*dense_final/ActivityRegularizer/Square:y:0.dense_final/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2%
#dense_final/ActivityRegularizer/Sum
%dense_final/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×£<2'
%dense_final/ActivityRegularizer/mul/xÐ
#dense_final/ActivityRegularizer/mulMul.dense_final/ActivityRegularizer/mul/x:output:0,dense_final/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_final/ActivityRegularizer/mul
%dense_final/ActivityRegularizer/ShapeShapedense_final/Softmax:softmax:0*
T0*
_output_shapes
:2'
%dense_final/ActivityRegularizer/Shape´
3dense_final/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3dense_final/ActivityRegularizer/strided_slice/stack¸
5dense_final/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_1¸
5dense_final/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5dense_final/ActivityRegularizer/strided_slice/stack_2¢
-dense_final/ActivityRegularizer/strided_sliceStridedSlice.dense_final/ActivityRegularizer/Shape:output:0<dense_final/ActivityRegularizer/strided_slice/stack:output:0>dense_final/ActivityRegularizer/strided_slice/stack_1:output:0>dense_final/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-dense_final/ActivityRegularizer/strided_slice¼
$dense_final/ActivityRegularizer/CastCast6dense_final/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2&
$dense_final/ActivityRegularizer/CastÑ
'dense_final/ActivityRegularizer/truedivRealDiv'dense_final/ActivityRegularizer/mul:z:0(dense_final/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_final/ActivityRegularizer/truedivØ
4dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_final_matmul_readvariableop_resource*
_output_shapes
:	*
dtype026
4dense_final/kernel/Regularizer/Square/ReadVariableOpÀ
%dense_final/kernel/Regularizer/SquareSquare<dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%dense_final/kernel/Regularizer/Square
$dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$dense_final/kernel/Regularizer/ConstÊ
"dense_final/kernel/Regularizer/SumSum)dense_final/kernel/Regularizer/Square:y:0-dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/Sum
$dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *-²=2&
$dense_final/kernel/Regularizer/mul/xÌ
"dense_final/kernel/Regularizer/mulMul-dense_final/kernel/Regularizer/mul/x:output:0+dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/mulx
IdentityIdentitydense_final/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityy

Identity_1Identity+dense_final/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 2

Identity_1Ò
NoOpNoOp5^batchNormalization_1/FusedBatchNormV3/ReadVariableOp7^batchNormalization_1/FusedBatchNormV3/ReadVariableOp_1$^batchNormalization_1/ReadVariableOp&^batchNormalization_1/ReadVariableOp_16^batchNormalization_1b/FusedBatchNormV3/ReadVariableOp8^batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_1%^batchNormalization_1b/ReadVariableOp'^batchNormalization_1b/ReadVariableOp_16^batchNormalization_2b/FusedBatchNormV3/ReadVariableOp8^batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_1%^batchNormalization_2b/ReadVariableOp'^batchNormalization_2b/ReadVariableOp_15^batchNormalization_3/FusedBatchNormV3/ReadVariableOp7^batchNormalization_3/FusedBatchNormV3/ReadVariableOp_1$^batchNormalization_3/ReadVariableOp&^batchNormalization_3/ReadVariableOp_15^batchNormalization_4/FusedBatchNormV3/ReadVariableOp7^batchNormalization_4/FusedBatchNormV3/ReadVariableOp_1$^batchNormalization_4/ReadVariableOp&^batchNormalization_4/ReadVariableOp_1.^batchNormalization_5/batchnorm/ReadVariableOp0^batchNormalization_5/batchnorm/ReadVariableOp_10^batchNormalization_5/batchnorm/ReadVariableOp_22^batchNormalization_5/batchnorm/mul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_1b/BiasAdd/ReadVariableOp^conv_1b/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_2b/BiasAdd/ReadVariableOp^conv_2b/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp#^dense_final/BiasAdd/ReadVariableOp"^dense_final/MatMul/ReadVariableOp5^dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2l
4batchNormalization_1/FusedBatchNormV3/ReadVariableOp4batchNormalization_1/FusedBatchNormV3/ReadVariableOp2p
6batchNormalization_1/FusedBatchNormV3/ReadVariableOp_16batchNormalization_1/FusedBatchNormV3/ReadVariableOp_12J
#batchNormalization_1/ReadVariableOp#batchNormalization_1/ReadVariableOp2N
%batchNormalization_1/ReadVariableOp_1%batchNormalization_1/ReadVariableOp_12n
5batchNormalization_1b/FusedBatchNormV3/ReadVariableOp5batchNormalization_1b/FusedBatchNormV3/ReadVariableOp2r
7batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_17batchNormalization_1b/FusedBatchNormV3/ReadVariableOp_12L
$batchNormalization_1b/ReadVariableOp$batchNormalization_1b/ReadVariableOp2P
&batchNormalization_1b/ReadVariableOp_1&batchNormalization_1b/ReadVariableOp_12n
5batchNormalization_2b/FusedBatchNormV3/ReadVariableOp5batchNormalization_2b/FusedBatchNormV3/ReadVariableOp2r
7batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_17batchNormalization_2b/FusedBatchNormV3/ReadVariableOp_12L
$batchNormalization_2b/ReadVariableOp$batchNormalization_2b/ReadVariableOp2P
&batchNormalization_2b/ReadVariableOp_1&batchNormalization_2b/ReadVariableOp_12l
4batchNormalization_3/FusedBatchNormV3/ReadVariableOp4batchNormalization_3/FusedBatchNormV3/ReadVariableOp2p
6batchNormalization_3/FusedBatchNormV3/ReadVariableOp_16batchNormalization_3/FusedBatchNormV3/ReadVariableOp_12J
#batchNormalization_3/ReadVariableOp#batchNormalization_3/ReadVariableOp2N
%batchNormalization_3/ReadVariableOp_1%batchNormalization_3/ReadVariableOp_12l
4batchNormalization_4/FusedBatchNormV3/ReadVariableOp4batchNormalization_4/FusedBatchNormV3/ReadVariableOp2p
6batchNormalization_4/FusedBatchNormV3/ReadVariableOp_16batchNormalization_4/FusedBatchNormV3/ReadVariableOp_12J
#batchNormalization_4/ReadVariableOp#batchNormalization_4/ReadVariableOp2N
%batchNormalization_4/ReadVariableOp_1%batchNormalization_4/ReadVariableOp_12^
-batchNormalization_5/batchnorm/ReadVariableOp-batchNormalization_5/batchnorm/ReadVariableOp2b
/batchNormalization_5/batchnorm/ReadVariableOp_1/batchNormalization_5/batchnorm/ReadVariableOp_12b
/batchNormalization_5/batchnorm/ReadVariableOp_2/batchNormalization_5/batchnorm/ReadVariableOp_22f
1batchNormalization_5/batchnorm/mul/ReadVariableOp1batchNormalization_5/batchnorm/mul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2@
conv_1b/BiasAdd/ReadVariableOpconv_1b/BiasAdd/ReadVariableOp2>
conv_1b/Conv2D/ReadVariableOpconv_1b/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2@
conv_2b/BiasAdd/ReadVariableOpconv_2b/BiasAdd/ReadVariableOp2>
conv_2b/Conv2D/ReadVariableOpconv_2b/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
conv_4/BiasAdd/ReadVariableOpconv_4/BiasAdd/ReadVariableOp2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2H
"dense_final/BiasAdd/ReadVariableOp"dense_final/BiasAdd/ReadVariableOp2F
!dense_final/MatMul/ReadVariableOp!dense_final/MatMul/ReadVariableOp2l
4dense_final/kernel/Regularizer/Square/ReadVariableOp4dense_final/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
Ì
¾
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51538

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
Ï
¯
F__inference_dense_final_layer_call_and_return_conditional_losses_52546

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢4dense_final/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
SoftmaxÌ
4dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype026
4dense_final/kernel/Regularizer/Square/ReadVariableOpÀ
%dense_final/kernel/Regularizer/SquareSquare<dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%dense_final/kernel/Regularizer/Square
$dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$dense_final/kernel/Regularizer/ConstÊ
"dense_final/kernel/Regularizer/SumSum)dense_final/kernel/Regularizer/Square:y:0-dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/Sum
$dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *-²=2&
$dense_final/kernel/Regularizer/mul/xÌ
"dense_final/kernel/Regularizer/mulMul-dense_final/kernel/Regularizer/mul/x:output:0+dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/mull
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¶
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4dense_final/kernel/Regularizer/Square/ReadVariableOp4dense_final/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
ú
A__inference_conv_2_layer_call_and_return_conditional_losses_51765

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ00 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs

²
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_52405

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÂ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¿
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_50012

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¿
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_48799

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

b
D__inference_dropout_2_layer_call_and_return_conditional_losses_49483

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸
a
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_49512

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¾
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51502

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
Â
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52086

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
Â
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52311

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
E
)__inference_dropout_2_layer_call_fn_51816

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_494832
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸
a
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_51620

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 
 
_user_specified_nameinputs
»
`
D__inference_maxPool_4_layer_call_and_return_conditional_losses_52229

inputs
identity
MaxPoolMaxPoolinputs*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
c
D__inference_dropout_6_layer_call_and_return_conditional_losses_52482

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÅ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed¡¶2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

I
-__inference_globAvgPool_5_layer_call_fn_52380

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_491652
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
	
,__inference_sequential_5_layer_call_fn_51364

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@%

unknown_19:@

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:	

unknown_34:	

unknown_35:	

unknown_36:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: *H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_497092
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:ÿÿÿÿÿÿÿÿÿ``: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
¤
`
D__inference_maxPool_3_layer_call_and_return_conditional_losses_48868

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
û
B__inference_conv_1b_layer_call_and_return_conditional_losses_49416

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ``: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

^
B__inference_noise_3_layer_call_and_return_conditional_losses_49595

inputs
identityc
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_49581

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


&__inference_conv_1_layer_call_fn_51466

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_493722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ``: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs

·
__inference_loss_fn_0_52529P
=dense_final_kernel_regularizer_square_readvariableop_resource:	
identity¢4dense_final/kernel/Regularizer/Square/ReadVariableOpë
4dense_final/kernel/Regularizer/Square/ReadVariableOpReadVariableOp=dense_final_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype026
4dense_final/kernel/Regularizer/Square/ReadVariableOpÀ
%dense_final/kernel/Regularizer/SquareSquare<dense_final/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	2'
%dense_final/kernel/Regularizer/Square
$dense_final/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2&
$dense_final/kernel/Regularizer/ConstÊ
"dense_final/kernel/Regularizer/SumSum)dense_final/kernel/Regularizer/Square:y:0-dense_final/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/Sum
$dense_final/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *-²=2&
$dense_final/kernel/Regularizer/mul/xÌ
"dense_final/kernel/Regularizer/mulMul-dense_final/kernel/Regularizer/mul/x:output:0+dense_final/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_final/kernel/Regularizer/mulp
IdentityIdentity&dense_final/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp5^dense_final/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4dense_final/kernel/Regularizer/Square/ReadVariableOp4dense_final/kernel/Regularizer/Square/ReadVariableOp
ù
Ð
5__inference_batchNormalization_2b_layer_call_fn_51997

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_495312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
·
`
D__inference_maxPool_2_layer_call_and_return_conditional_losses_51784

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ00@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
Ì
¾
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_50190

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ``:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ``: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ``
 
_user_specified_nameinputs
ñ

O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52257

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
F
*__inference_maxPool_1b_layer_call_fn_51630

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_494262
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 
 
_user_specified_nameinputs
¥
b
)__inference_dropout_6_layer_call_fn_52492

inputs
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_498312
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
a
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_49426

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`` 
 
_user_specified_nameinputs
¥
Â
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52275

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
¿
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_50131

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3Â
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueÎ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 2

IdentityÜ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00 : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00 
 
_user_specified_nameinputs

d
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_49652

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
Ó
4__inference_batchNormalization_4_layer_call_fn_52350

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_496372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_52470

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*À
serving_default¬
M
conv_1_input=
serving_default_conv_1_input:0ÿÿÿÿÿÿÿÿÿ``?
dense_final0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¾ì
ö
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
layer-21
layer_with_weights-12
layer-22
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
â_default_save_signature
+ã&call_and_return_all_conditional_losses
ä__call__"
_tf_keras_sequential
½

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
+å&call_and_return_all_conditional_losses
æ__call__"
_tf_keras_layer
ì
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+ç&call_and_return_all_conditional_losses
è__call__"
_tf_keras_layer
½

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+é&call_and_return_all_conditional_losses
ê__call__"
_tf_keras_layer
§
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+ë&call_and_return_all_conditional_losses
ì__call__"
_tf_keras_layer
ì
7axis
	8gamma
9beta
:moving_mean
;moving_variance
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+í&call_and_return_all_conditional_losses
î__call__"
_tf_keras_layer
½

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+ï&call_and_return_all_conditional_losses
ð__call__"
_tf_keras_layer
§
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+ñ&call_and_return_all_conditional_losses
ò__call__"
_tf_keras_layer
§
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+ó&call_and_return_all_conditional_losses
ô__call__"
_tf_keras_layer
§
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+õ&call_and_return_all_conditional_losses
ö__call__"
_tf_keras_layer
½

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+÷&call_and_return_all_conditional_losses
ø__call__"
_tf_keras_layer
§
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+ù&call_and_return_all_conditional_losses
ú__call__"
_tf_keras_layer
ì
\axis
	]gamma
^beta
_moving_mean
`moving_variance
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+û&call_and_return_all_conditional_losses
ü__call__"
_tf_keras_layer
½

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
+ý&call_and_return_all_conditional_losses
þ__call__"
_tf_keras_layer
§
kregularization_losses
l	variables
mtrainable_variables
n	keras_api
+ÿ&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
tregularization_losses
u	variables
vtrainable_variables
w	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
§
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
¿

|kernel
}bias
~regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
«
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
«
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
«
regularization_losses
	variables
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Ã
 kernel
	¡bias
¢regularization_losses
£	variables
¤trainable_variables
¥	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ì
	¦iter
§beta_1
¨beta_2

©decay
ªlearning_ratem®m¯%m°&m±-m².m³8m´9mµ@m¶Am·Rm¸Sm¹]mº^m»em¼fm½pm¾qm¿|mÀ}mÁ	mÂ	mÃ	mÄ	mÅ	 mÆ	¡mÇvÈvÉ%vÊ&vË-vÌ.vÍ8vÎ9vÏ@vÐAvÑRvÒSvÓ]vÔ^vÕevÖfv×pvØqvÙ|vÚ}vÛ	vÜ	vÝ	vÞ	vß	 và	¡vá"
	optimizer
(
0"
trackable_list_wrapper
Ð
0
1
%2
&3
'4
(5
-6
.7
88
99
:10
;11
@12
A13
R14
S15
]16
^17
_18
`19
e20
f21
p22
q23
r24
s25
|26
}27
28
29
30
31
32
33
34
35
 36
¡37"
trackable_list_wrapper
ì
0
1
%2
&3
-4
.5
86
97
@8
A9
R10
S11
]12
^13
e14
f15
p16
q17
|18
}19
20
21
22
23
 24
¡25"
trackable_list_wrapper
Ó
«non_trainable_variables
regularization_losses
 ¬layer_regularization_losses
­layer_metrics
®metrics
¯layers
	variables
trainable_variables
ä__call__
â_default_save_signature
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
':%2conv_1/kernel
:2conv_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
µ
°non_trainable_variables
 regularization_losses
 ±layer_regularization_losses
²layer_metrics
³metrics
´layers
!	variables
"trainable_variables
æ__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&2batchNormalization_1/gamma
':%2batchNormalization_1/beta
0:. (2 batchNormalization_1/moving_mean
4:2 (2$batchNormalization_1/moving_variance
 "
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
µ
µnon_trainable_variables
)regularization_losses
 ¶layer_regularization_losses
·layer_metrics
¸metrics
¹layers
*	variables
+trainable_variables
è__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
(:& 2conv_1b/kernel
: 2conv_1b/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
µ
ºnon_trainable_variables
/regularization_losses
 »layer_regularization_losses
¼layer_metrics
½metrics
¾layers
0	variables
1trainable_variables
ê__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¿non_trainable_variables
3regularization_losses
 Àlayer_regularization_losses
Álayer_metrics
Âmetrics
Ãlayers
4	variables
5trainable_variables
ì__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batchNormalization_1b/gamma
(:& 2batchNormalization_1b/beta
1:/  (2!batchNormalization_1b/moving_mean
5:3  (2%batchNormalization_1b/moving_variance
 "
trackable_list_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
µ
Änon_trainable_variables
<regularization_losses
 Ålayer_regularization_losses
Ælayer_metrics
Çmetrics
Èlayers
=	variables
>trainable_variables
î__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
':% @2conv_2/kernel
:@2conv_2/bias
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
µ
Énon_trainable_variables
Bregularization_losses
 Êlayer_regularization_losses
Ëlayer_metrics
Ìmetrics
Ílayers
C	variables
Dtrainable_variables
ð__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Înon_trainable_variables
Fregularization_losses
 Ïlayer_regularization_losses
Ðlayer_metrics
Ñmetrics
Òlayers
G	variables
Htrainable_variables
ò__call__
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ónon_trainable_variables
Jregularization_losses
 Ôlayer_regularization_losses
Õlayer_metrics
Ömetrics
×layers
K	variables
Ltrainable_variables
ô__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ønon_trainable_variables
Nregularization_losses
 Ùlayer_regularization_losses
Úlayer_metrics
Ûmetrics
Ülayers
O	variables
Ptrainable_variables
ö__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
(:&@@2conv_2b/kernel
:@2conv_2b/bias
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
µ
Ýnon_trainable_variables
Tregularization_losses
 Þlayer_regularization_losses
ßlayer_metrics
àmetrics
álayers
U	variables
Vtrainable_variables
ø__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ânon_trainable_variables
Xregularization_losses
 ãlayer_regularization_losses
älayer_metrics
åmetrics
ælayers
Y	variables
Ztrainable_variables
ú__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batchNormalization_2b/gamma
(:&@2batchNormalization_2b/beta
1:/@ (2!batchNormalization_2b/moving_mean
5:3@ (2%batchNormalization_2b/moving_variance
 "
trackable_list_wrapper
<
]0
^1
_2
`3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
µ
çnon_trainable_variables
aregularization_losses
 èlayer_regularization_losses
élayer_metrics
êmetrics
ëlayers
b	variables
ctrainable_variables
ü__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
(:&@2conv_3/kernel
:2conv_3/bias
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
µ
ìnon_trainable_variables
gregularization_losses
 ílayer_regularization_losses
îlayer_metrics
ïmetrics
ðlayers
h	variables
itrainable_variables
þ__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ñnon_trainable_variables
kregularization_losses
 òlayer_regularization_losses
ólayer_metrics
ômetrics
õlayers
l	variables
mtrainable_variables
__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batchNormalization_3/gamma
(:&2batchNormalization_3/beta
1:/ (2 batchNormalization_3/moving_mean
5:3 (2$batchNormalization_3/moving_variance
 "
trackable_list_wrapper
<
p0
q1
r2
s3"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
µ
önon_trainable_variables
tregularization_losses
 ÷layer_regularization_losses
ølayer_metrics
ùmetrics
úlayers
u	variables
vtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ûnon_trainable_variables
xregularization_losses
 ülayer_regularization_losses
ýlayer_metrics
þmetrics
ÿlayers
y	variables
ztrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'2conv_4/kernel
:2conv_4/bias
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
¶
non_trainable_variables
~regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batchNormalization_4/gamma
(:&2batchNormalization_4/beta
1:/ (2 batchNormalization_4/moving_mean
5:3 (2$batchNormalization_4/moving_variance
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batchNormalization_5/gamma
(:&2batchNormalization_5/beta
1:/ (2 batchNormalization_5/moving_mean
5:3 (2$batchNormalization_5/moving_variance
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
regularization_losses
 layer_regularization_losses
layer_metrics
metrics
layers
	variables
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#	2dense_final/kernel
:2dense_final/bias
(
0"
trackable_list_wrapper
0
 0
¡1"
trackable_list_wrapper
0
 0
¡1"
trackable_list_wrapper
Ö
non_trainable_variables
¢regularization_losses
 layer_regularization_losses
 layer_metrics
¡metrics
¢layers
£	variables
¤trainable_variables
__call__
activity_regularizer_fn
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adamax/iter
: (2Adamax/beta_1
: (2Adamax/beta_2
: (2Adamax/decay
: (2Adamax/learning_rate
z
'0
(1
:2
;3
_4
`5
r6
s7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
£0
¤1"
trackable_list_wrapper
Î
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

¥total

¦count
§	variables
¨	keras_api"
_tf_keras_metric
c

©total

ªcount
«
_fn_kwargs
¬	variables
­	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
¥0
¦1"
trackable_list_wrapper
.
§	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
©0
ª1"
trackable_list_wrapper
.
¬	variables"
_generic_user_object
.:,2Adamax/conv_1/kernel/m
 :2Adamax/conv_1/bias/m
/:-2#Adamax/batchNormalization_1/gamma/m
.:,2"Adamax/batchNormalization_1/beta/m
/:- 2Adamax/conv_1b/kernel/m
!: 2Adamax/conv_1b/bias/m
0:. 2$Adamax/batchNormalization_1b/gamma/m
/:- 2#Adamax/batchNormalization_1b/beta/m
.:, @2Adamax/conv_2/kernel/m
 :@2Adamax/conv_2/bias/m
/:-@@2Adamax/conv_2b/kernel/m
!:@2Adamax/conv_2b/bias/m
0:.@2$Adamax/batchNormalization_2b/gamma/m
/:-@2#Adamax/batchNormalization_2b/beta/m
/:-@2Adamax/conv_3/kernel/m
!:2Adamax/conv_3/bias/m
0:.2#Adamax/batchNormalization_3/gamma/m
/:-2"Adamax/batchNormalization_3/beta/m
0:.2Adamax/conv_4/kernel/m
!:2Adamax/conv_4/bias/m
0:.2#Adamax/batchNormalization_4/gamma/m
/:-2"Adamax/batchNormalization_4/beta/m
0:.2#Adamax/batchNormalization_5/gamma/m
/:-2"Adamax/batchNormalization_5/beta/m
,:*	2Adamax/dense_final/kernel/m
%:#2Adamax/dense_final/bias/m
.:,2Adamax/conv_1/kernel/v
 :2Adamax/conv_1/bias/v
/:-2#Adamax/batchNormalization_1/gamma/v
.:,2"Adamax/batchNormalization_1/beta/v
/:- 2Adamax/conv_1b/kernel/v
!: 2Adamax/conv_1b/bias/v
0:. 2$Adamax/batchNormalization_1b/gamma/v
/:- 2#Adamax/batchNormalization_1b/beta/v
.:, @2Adamax/conv_2/kernel/v
 :@2Adamax/conv_2/bias/v
/:-@@2Adamax/conv_2b/kernel/v
!:@2Adamax/conv_2b/bias/v
0:.@2$Adamax/batchNormalization_2b/gamma/v
/:-@2#Adamax/batchNormalization_2b/beta/v
/:-@2Adamax/conv_3/kernel/v
!:2Adamax/conv_3/bias/v
0:.2#Adamax/batchNormalization_3/gamma/v
/:-2"Adamax/batchNormalization_3/beta/v
0:.2Adamax/conv_4/kernel/v
!:2Adamax/conv_4/bias/v
0:.2#Adamax/batchNormalization_4/gamma/v
/:-2"Adamax/batchNormalization_4/beta/v
0:.2#Adamax/batchNormalization_5/gamma/v
/:-2"Adamax/batchNormalization_5/beta/v
,:*	2Adamax/dense_final/kernel/v
%:#2Adamax/dense_final/bias/v
ÐBÍ
 __inference__wrapped_model_48415conv_1_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
G__inference_sequential_5_layer_call_and_return_conditional_losses_51073
G__inference_sequential_5_layer_call_and_return_conditional_losses_51282
G__inference_sequential_5_layer_call_and_return_conditional_losses_50693
G__inference_sequential_5_layer_call_and_return_conditional_losses_50811À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
,__inference_sequential_5_layer_call_fn_49789
,__inference_sequential_5_layer_call_fn_51364
,__inference_sequential_5_layer_call_fn_51446
,__inference_sequential_5_layer_call_fn_50575À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_conv_1_layer_call_and_return_conditional_losses_51457¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_conv_1_layer_call_fn_51466¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51484
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51502
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51520
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51538´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
4__inference_batchNormalization_1_layer_call_fn_51551
4__inference_batchNormalization_1_layer_call_fn_51564
4__inference_batchNormalization_1_layer_call_fn_51577
4__inference_batchNormalization_1_layer_call_fn_51590´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
B__inference_conv_1b_layer_call_and_return_conditional_losses_51601¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_conv_1b_layer_call_fn_51610¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¶2³
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_51615
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_51620¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ý
*__inference_maxPool_1b_layer_call_fn_51625
*__inference_maxPool_1b_layer_call_fn_51630¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿ
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51648
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51666
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51684
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51702´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
5__inference_batchNormalization_1b_layer_call_fn_51715
5__inference_batchNormalization_1b_layer_call_fn_51728
5__inference_batchNormalization_1b_layer_call_fn_51741
5__inference_batchNormalization_1b_layer_call_fn_51754´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_conv_2_layer_call_and_return_conditional_losses_51765¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_conv_2_layer_call_fn_51774¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±
D__inference_maxPool_2_layer_call_and_return_conditional_losses_51779
D__inference_maxPool_2_layer_call_and_return_conditional_losses_51784¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
)__inference_maxPool_2_layer_call_fn_51789
)__inference_maxPool_2_layer_call_fn_51794¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Æ2Ã
D__inference_dropout_2_layer_call_and_return_conditional_losses_51799
D__inference_dropout_2_layer_call_and_return_conditional_losses_51811´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_dropout_2_layer_call_fn_51816
)__inference_dropout_2_layer_call_fn_51821´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Â2¿
B__inference_noise_1_layer_call_and_return_conditional_losses_51825
B__inference_noise_1_layer_call_and_return_conditional_losses_51836´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
'__inference_noise_1_layer_call_fn_51841
'__inference_noise_1_layer_call_fn_51846´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
B__inference_conv_2b_layer_call_and_return_conditional_losses_51857¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_conv_2b_layer_call_fn_51866¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¶2³
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_51871
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_51876¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ý
*__inference_maxPool_2b_layer_call_fn_51881
*__inference_maxPool_2b_layer_call_fn_51886¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿ
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51904
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51922
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51940
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51958´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
5__inference_batchNormalization_2b_layer_call_fn_51971
5__inference_batchNormalization_2b_layer_call_fn_51984
5__inference_batchNormalization_2b_layer_call_fn_51997
5__inference_batchNormalization_2b_layer_call_fn_52010´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_conv_3_layer_call_and_return_conditional_losses_52021¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_conv_3_layer_call_fn_52030¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±
D__inference_maxPool_3_layer_call_and_return_conditional_losses_52035
D__inference_maxPool_3_layer_call_and_return_conditional_losses_52040¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
)__inference_maxPool_3_layer_call_fn_52045
)__inference_maxPool_3_layer_call_fn_52050¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52068
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52086
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52104
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52122´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
4__inference_batchNormalization_3_layer_call_fn_52135
4__inference_batchNormalization_3_layer_call_fn_52148
4__inference_batchNormalization_3_layer_call_fn_52161
4__inference_batchNormalization_3_layer_call_fn_52174´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Â2¿
B__inference_noise_3_layer_call_and_return_conditional_losses_52178
B__inference_noise_3_layer_call_and_return_conditional_losses_52189´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
'__inference_noise_3_layer_call_fn_52194
'__inference_noise_3_layer_call_fn_52199´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_conv_4_layer_call_and_return_conditional_losses_52210¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_conv_4_layer_call_fn_52219¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±
D__inference_maxPool_4_layer_call_and_return_conditional_losses_52224
D__inference_maxPool_4_layer_call_and_return_conditional_losses_52229¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
)__inference_maxPool_4_layer_call_fn_52234
)__inference_maxPool_4_layer_call_fn_52239¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52257
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52275
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52293
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52311´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
4__inference_batchNormalization_4_layer_call_fn_52324
4__inference_batchNormalization_4_layer_call_fn_52337
4__inference_batchNormalization_4_layer_call_fn_52350
4__inference_batchNormalization_4_layer_call_fn_52363´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¼2¹
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_52369
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_52375¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
-__inference_globAvgPool_5_layer_call_fn_52380
-__inference_globAvgPool_5_layer_call_fn_52385¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_52405
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_52439´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦2£
4__inference_batchNormalization_5_layer_call_fn_52452
4__inference_batchNormalization_5_layer_call_fn_52465´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_6_layer_call_and_return_conditional_losses_52470
D__inference_dropout_6_layer_call_and_return_conditional_losses_52482´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_dropout_6_layer_call_fn_52487
)__inference_dropout_6_layer_call_fn_52492´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ô2ñ
J__inference_dense_final_layer_call_and_return_all_conditional_losses_52509¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_final_layer_call_fn_52518¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
²2¯
__inference_loss_fn_0_52529
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
ÏBÌ
#__inference_signature_wrapper_50906conv_1_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ã2à
2__inference_dense_final_activity_regularizer_49354©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ð2í
F__inference_dense_final_layer_call_and_return_conditional_losses_52546¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ñ
 __inference__wrapped_model_48415¬0%&'(-.89:;@ARS]^_`efpqrs|} ¡=¢:
3¢0
.+
conv_1_inputÿÿÿÿÿÿÿÿÿ``
ª "9ª6
4
dense_final%"
dense_finalÿÿÿÿÿÿÿÿÿê
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51484%&'(M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ê
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51502%&'(M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51520r%&'(;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ``
 Å
O__inference_batchNormalization_1_layer_call_and_return_conditional_losses_51538r%&'(;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ``
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ``
 Â
4__inference_batchNormalization_1_layer_call_fn_51551%&'(M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
4__inference_batchNormalization_1_layer_call_fn_51564%&'(M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4__inference_batchNormalization_1_layer_call_fn_51577e%&'(;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 
ª " ÿÿÿÿÿÿÿÿÿ``
4__inference_batchNormalization_1_layer_call_fn_51590e%&'(;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ``
p
ª " ÿÿÿÿÿÿÿÿÿ``ë
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_5164889:;M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ë
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_5166689:;M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Æ
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51684r89:;;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00 
 Æ
P__inference_batchNormalization_1b_layer_call_and_return_conditional_losses_51702r89:;;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00 
 Ã
5__inference_batchNormalization_1b_layer_call_fn_5171589:;M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
5__inference_batchNormalization_1b_layer_call_fn_5172889:;M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
5__inference_batchNormalization_1b_layer_call_fn_51741e89:;;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00 
p 
ª " ÿÿÿÿÿÿÿÿÿ00 
5__inference_batchNormalization_1b_layer_call_fn_51754e89:;;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00 
p
ª " ÿÿÿÿÿÿÿÿÿ00 ë
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51904]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ë
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51922]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Æ
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51940r]^_`;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Æ
P__inference_batchNormalization_2b_layer_call_and_return_conditional_losses_51958r]^_`;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ã
5__inference_batchNormalization_2b_layer_call_fn_51971]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
5__inference_batchNormalization_2b_layer_call_fn_51984]^_`M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
5__inference_batchNormalization_2b_layer_call_fn_51997e]^_`;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@
5__inference_batchNormalization_2b_layer_call_fn_52010e]^_`;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@ì
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52068pqrsN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ì
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52086pqrsN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52104tpqrs<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ç
O__inference_batchNormalization_3_layer_call_and_return_conditional_losses_52122tpqrs<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ä
4__inference_batchNormalization_3_layer_call_fn_52135pqrsN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
4__inference_batchNormalization_3_layer_call_fn_52148pqrsN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4__inference_batchNormalization_3_layer_call_fn_52161gpqrs<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ
4__inference_batchNormalization_3_layer_call_fn_52174gpqrs<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿð
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52257N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52275N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ë
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52293x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ë
O__inference_batchNormalization_4_layer_call_and_return_conditional_losses_52311x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 È
4__inference_batchNormalization_4_layer_call_fn_52324N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
4__inference_batchNormalization_4_layer_call_fn_52337N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
4__inference_batchNormalization_4_layer_call_fn_52350k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ£
4__inference_batchNormalization_4_layer_call_fn_52363k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ»
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_52405h4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 »
O__inference_batchNormalization_5_layer_call_and_return_conditional_losses_52439h4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
4__inference_batchNormalization_5_layer_call_fn_52452[4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
4__inference_batchNormalization_5_layer_call_fn_52465[4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ±
A__inference_conv_1_layer_call_and_return_conditional_losses_51457l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ``
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ``
 
&__inference_conv_1_layer_call_fn_51466_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ``
ª " ÿÿÿÿÿÿÿÿÿ``²
B__inference_conv_1b_layer_call_and_return_conditional_losses_51601l-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ``
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ`` 
 
'__inference_conv_1b_layer_call_fn_51610_-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ``
ª " ÿÿÿÿÿÿÿÿÿ`` ±
A__inference_conv_2_layer_call_and_return_conditional_losses_51765l@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00@
 
&__inference_conv_2_layer_call_fn_51774_@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00 
ª " ÿÿÿÿÿÿÿÿÿ00@²
B__inference_conv_2b_layer_call_and_return_conditional_losses_51857lRS7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
'__inference_conv_2b_layer_call_fn_51866_RS7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@²
A__inference_conv_3_layer_call_and_return_conditional_losses_52021mef7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
&__inference_conv_3_layer_call_fn_52030`ef7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ³
A__inference_conv_4_layer_call_and_return_conditional_losses_52210n|}8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
&__inference_conv_4_layer_call_fn_52219a|}8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ\
2__inference_dense_final_activity_regularizer_49354&¢
¢
	
x
ª " »
J__inference_dense_final_layer_call_and_return_all_conditional_losses_52509m ¡0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 ©
F__inference_dense_final_layer_call_and_return_conditional_losses_52546_ ¡0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_final_layer_call_fn_52518R ¡0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ´
D__inference_dropout_2_layer_call_and_return_conditional_losses_51799l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ´
D__inference_dropout_2_layer_call_and_return_conditional_losses_51811l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_dropout_2_layer_call_fn_51816_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@
)__inference_dropout_2_layer_call_fn_51821_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@¦
D__inference_dropout_6_layer_call_and_return_conditional_losses_52470^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¦
D__inference_dropout_6_layer_call_and_return_conditional_losses_52482^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dropout_6_layer_call_fn_52487Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ~
)__inference_dropout_6_layer_call_fn_52492Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÑ
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_52369R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ®
H__inference_globAvgPool_5_layer_call_and_return_conditional_losses_52375b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
-__inference_globAvgPool_5_layer_call_fn_52380wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-__inference_globAvgPool_5_layer_call_fn_52385U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ;
__inference_loss_fn_0_52529 ¢

¢ 
ª " è
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_51615R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ±
E__inference_maxPool_1b_layer_call_and_return_conditional_losses_51620h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ`` 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00 
 À
*__inference_maxPool_1b_layer_call_fn_51625R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*__inference_maxPool_1b_layer_call_fn_51630[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ`` 
ª " ÿÿÿÿÿÿÿÿÿ00 ç
D__inference_maxPool_2_layer_call_and_return_conditional_losses_51779R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
D__inference_maxPool_2_layer_call_and_return_conditional_losses_51784h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ¿
)__inference_maxPool_2_layer_call_fn_51789R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
)__inference_maxPool_2_layer_call_fn_51794[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00@
ª " ÿÿÿÿÿÿÿÿÿ@è
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_51871R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ±
E__inference_maxPool_2b_layer_call_and_return_conditional_losses_51876h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 À
*__inference_maxPool_2b_layer_call_fn_51881R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*__inference_maxPool_2b_layer_call_fn_51886[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ç
D__inference_maxPool_3_layer_call_and_return_conditional_losses_52035R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ²
D__inference_maxPool_3_layer_call_and_return_conditional_losses_52040j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¿
)__inference_maxPool_3_layer_call_fn_52045R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
)__inference_maxPool_3_layer_call_fn_52050]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿç
D__inference_maxPool_4_layer_call_and_return_conditional_losses_52224R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ²
D__inference_maxPool_4_layer_call_and_return_conditional_losses_52229j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¿
)__inference_maxPool_4_layer_call_fn_52234R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
)__inference_maxPool_4_layer_call_fn_52239]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ²
B__inference_noise_1_layer_call_and_return_conditional_losses_51825l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ²
B__inference_noise_1_layer_call_and_return_conditional_losses_51836l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
'__inference_noise_1_layer_call_fn_51841_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@
'__inference_noise_1_layer_call_fn_51846_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@´
B__inference_noise_3_layer_call_and_return_conditional_losses_52178n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ´
B__inference_noise_3_layer_call_and_return_conditional_losses_52189n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
'__inference_noise_3_layer_call_fn_52194a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ
'__inference_noise_3_layer_call_fn_52199a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿú
G__inference_sequential_5_layer_call_and_return_conditional_losses_50693®0%&'(-.89:;@ARS]^_`efpqrs|} ¡E¢B
;¢8
.+
conv_1_inputÿÿÿÿÿÿÿÿÿ``
p 

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 ú
G__inference_sequential_5_layer_call_and_return_conditional_losses_50811®0%&'(-.89:;@ARS]^_`efpqrs|} ¡E¢B
;¢8
.+
conv_1_inputÿÿÿÿÿÿÿÿÿ``
p

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 ô
G__inference_sequential_5_layer_call_and_return_conditional_losses_51073¨0%&'(-.89:;@ARS]^_`efpqrs|} ¡?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 ô
G__inference_sequential_5_layer_call_and_return_conditional_losses_51282¨0%&'(-.89:;@ARS]^_`efpqrs|} ¡?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 Ä
,__inference_sequential_5_layer_call_fn_497890%&'(-.89:;@ARS]^_`efpqrs|} ¡E¢B
;¢8
.+
conv_1_inputÿÿÿÿÿÿÿÿÿ``
p 

 
ª "ÿÿÿÿÿÿÿÿÿÄ
,__inference_sequential_5_layer_call_fn_505750%&'(-.89:;@ARS]^_`efpqrs|} ¡E¢B
;¢8
.+
conv_1_inputÿÿÿÿÿÿÿÿÿ``
p

 
ª "ÿÿÿÿÿÿÿÿÿ¾
,__inference_sequential_5_layer_call_fn_513640%&'(-.89:;@ARS]^_`efpqrs|} ¡?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¾
,__inference_sequential_5_layer_call_fn_514460%&'(-.89:;@ARS]^_`efpqrs|} ¡?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ``
p

 
ª "ÿÿÿÿÿÿÿÿÿä
#__inference_signature_wrapper_50906¼0%&'(-.89:;@ARS]^_`efpqrs|} ¡M¢J
¢ 
Cª@
>
conv_1_input.+
conv_1_inputÿÿÿÿÿÿÿÿÿ``"9ª6
4
dense_final%"
dense_finalÿÿÿÿÿÿÿÿÿ