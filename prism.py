#感谢我自己和ai

from Crypto.Util.number import long_to_bytes,bytes_to_long,isPrime
import base64
from itertools import product,combinations
# from gmpy2 import gcdext

#自定义类使得对列表的操作变成对内容的操作
class cal(list):
    def __add__(self, other):
        if isinstance(other, cal):
            min_len = min(len(self), len(other))
            result = cal(a + b for a, b in zip(self[:min_len], other[:min_len]))
            if len(self) > min_len:
                result.extend(self[min_len:])
            elif len(other) > min_len:
                result.extend(other[min_len:])
            return result
        elif isinstance(other, int):
            return cal(a + other for a in self)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, cal):
            min_len = min(len(self), len(other))
            result = cal(a - b for a, b in zip(self[:min_len], other[:min_len]))
            if len(self) > min_len:
                result.extend(self[min_len:])
            elif len(other) > min_len:
                result.extend(other[min_len:])
            return result
        elif isinstance(other, int):
            return cal(a - other for a in self)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, cal):
            min_len = min(len(self), len(other))
            result = cal(a * b for a, b in zip(self[:min_len], other[:min_len]))
            if len(self) > min_len:
                result.extend(self[min_len:])
            elif len(other) > min_len:
                result.extend(other[min_len:])
            return result
        elif isinstance(other, int):
            return cal(a * other for a in self)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, cal):
            min_len = min(len(self), len(other))
            result = cal(a / b for a, b in zip(self[:min_len], other[:min_len]))
            if len(self) > min_len:
                result.extend(self[min_len:])
            elif len(other) > min_len:
                result.extend(other[min_len:])
            return result
        elif isinstance(other, int):
            return cal(a / other for a in self)
        else:
            return NotImplemented

    def __mod__(self, other):
        if isinstance(other, cal):
            min_len = min(len(self), len(other))
            result = cal(a % b for a, b in zip(self[:min_len], other[:min_len]))
            if len(self) > min_len:
                result.extend(self[min_len:])
            elif len(other) > min_len:
                result.extend(other[min_len:])
            return result
        elif isinstance(other, int):
            return cal(a % other for a in self)
        else:
            return NotImplemented

    def __xor__(self, other):
        if isinstance(other, cal):
            min_len = min(len(self), len(other))
            result = cal(a ^ b for a, b in zip(self[:min_len], other[:min_len]))
            if len(self) > min_len:
                result.extend(self[min_len:])
            elif len(other) > min_len:
                result.extend(other[min_len:])
            return result
        elif isinstance(other, int):
            return cal(a ^ other for a in self)
        else:
            return NotImplemented
    def __rxor__(self, other):
        if isinstance(other, int):
            return self.__xor__(other)
    def __getitem__(self, index):#省略%len
        if index < 0:
            index = len(self) + index
        if index >= len(self):
            return super().__getitem__(index % len(self))
        else:
            return super().__getitem__(index)

class Encflag:
    def __init__(self, flag):
        self.encflag = flag
        self.encflag_len = len(flag)
        self.flag_bytes = bytes(flag, encoding='utf-8')
        self.flag_bytes_len = len(self.flag_bytes)
    def pl(self):
        return pl(self.encflag)

class Hooker:
    def __init__(self, obj, origin_func):
        self.obj = obj
        self.origin_func = origin_func
        self.cont = 0
        self.original_func_name = ""

    def setHook(self, hook_func=None):
        self.original_func_name = f"original_{self.origin_func.__name__}_{self.cont}"
        self.cont += 1
        setattr(self.obj, self.original_func_name, getattr(self.obj, self.origin_func.__name__))
        if hook_func is None:
            hook_func = self.hookEntry
        setattr(self.obj, self.origin_func.__name__, hook_func.__get__(self.obj, self.obj.__class__))

    def endHook(self):
        setattr(self.obj, self.origin_func.__name__, getattr(self.obj, self.original_func_name))

    def hookEntry(self, *args, **kwargs):
        def modeficed_start(*args, **kwargs):
            print("Arguments:", args)
            print("Keyword Arguments:", kwargs)
        modeficed_start(*args, **kwargs)
        ret = self.origin_func(*args, **kwargs)
        print(ret)
        return ret
# 实现对数的自动模运算 
class TeaDec:
    def __init__(self, value, mask=0xffffffff):
        if not isinstance(value, int):
            raise ValueError("Value must be an integer")
        self.value = value & mask
        self.mask = mask

    def __add__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        return TeaDec((self.value + other.value) & self.mask, self.mask)

    def __iadd__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        self.value = (self.value + other.value) & self.mask
        return self

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        return TeaDec((self.value - other.value) & self.mask, self.mask)

    def __isub__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        self.value = (self.value - other.value) & self.mask
        return self

    def __rsub__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        return TeaDec((other.value - self.value) & self.mask, self.mask)

    def __mul__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        return TeaDec((self.value * other.value) & self.mask, self.mask)

    def __imul__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        self.value = (self.value * other.value) & self.mask
        return self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        if other.value == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return TeaDec(self.value // other.value, self.mask)

    def __itruediv__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        if other.value == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        self.value = self.value // other.value
        return self

    def __rtruediv__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        if self.value == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return TeaDec(other.value // self.value, self.mask)

    def __mod__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        if other.value == 0:
            raise ZeroDivisionError("Cannot modulo by zero")
        return TeaDec(self.value % other.value, self.mask)

    def __imod__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        if other.value == 0:
            raise ZeroDivisionError("Cannot modulo by zero")
        self.value = self.value % other.value
        return self

    def __rmod__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        if self.value == 0:
            raise ZeroDivisionError("Cannot modulo by zero")
        return TeaDec(other.value % self.value, self.mask)

    def __and__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        return TeaDec(self.value & other.value, self.mask)

    def __or__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        return TeaDec(self.value | other.value, self.mask)

    def __xor__(self, other):
        if isinstance(other, int):
            other = TeaDec(other, self.mask)
        elif not isinstance(other, TeaDec):
            raise TypeError("Operand must be a TeaDec instance or an integer")
        return TeaDec(self.value ^ other.value, self.mask)

    def __rand__(self, other):
        return self & other

    def __ror__(self, other):
        return self | other

    def __rxor__(self, other):
        return self ^ other

    def __lshift__(self, other):
        if not isinstance(other, int):
            raise TypeError("Shift amount must be an integer")
        return TeaDec((self.value << other) & self.mask, self.mask)

    def __ilshift__(self, other):
        if not isinstance(other, int):
            raise TypeError("Shift amount must be an integer")
        self.value = (self.value << other) & self.mask
        return self

    def __rlshift__(self, other):
        if not isinstance(other, int):
            raise TypeError("Shift amount must be an integer")
        return TeaDec((other << self.value) & self.mask, self.mask)

    def __rshift__(self, other):
        if not isinstance(other, int):
            raise TypeError("Shift amount must be an integer")
        return TeaDec(self.value >> other, self.mask)

    def __irshift__(self, other):
        if not isinstance(other, int):
            raise TypeError("Shift amount must be an integer")
        self.value = self.value >> other
        return self

    def __rrshift__(self, other):
        if not isinstance(other, int):
            raise TypeError("Shift amount must be an integer")
        return TeaDec(other >> self.value, self.mask)

    def __str__(self):
        return hex(self.value)
    def setMask(self, mask):
        self.mask = mask
class Tealist:
    def __init__(self, *args):
        if isinstance(args[0], list):
            args = args[0]
        self.list = [TeaDec(arg) for arg in args]
    
    def __getitem__(self, index):
        return self.list[index]
    
    def __setitem__(self, index, value):
        if not isinstance(value, TeaDec):
            raise TypeError("Value must be a TeaDec instance")
        self.list[index] = value
    
    def __len__(self):
        return len(self.list)
    
    def __iter__(self):
        return iter(self.list)
    
    def __str__(self) -> str:
        return " ".join(str(x) for x in self.list)
    def setMask(self, mask):
        for i in range(len(self.list)):
            self.list[i].setMask(mask)
# 定义一个类

'''
常用工具
'''

import string
import random

def generate(char_set, length, prefix=None):
    # 预定义字符集
    predefined_sets = {
        "_NUM": string.digits,
        "_BIG": string.ascii_uppercase,
        "_SMALL": string.ascii_lowercase,
        "_OTHER": string.punctuation.replace("{", "").replace("}", "")
    }

    # 解析字符集参数
    chars = []
    for s in char_set.split("+"):
        if s.startswith("_"):
            chars.extend(list(predefined_sets[s]))
        else:
            chars.extend(list(s))
    chars = list(set(chars))  # 去重

    # 生成随机字符串
    result = []
    if prefix:
        result.append(prefix + "{")
        length -= len(prefix) + 2  # 考虑到前缀和 "{" "}"
    while len(result) < min(length, len(chars)):
        random.shuffle(chars)
        result.extend(c for c in chars if c not in result)
    while len(result) < length:
        result.append(random.choice(chars))
    result = result[:length]  # 如果超过长度，进行截断

    # 添加后缀
    if prefix:
        result.append("}")

    # 返回结果
    result_str = "".join(result)
    print(result_str)
    return result_str



'''
爆破flag。传入最后检查列表，传入检查函数，传入一次爆破字节数(默认为1)
传出flag字符串。
check_function要求第一个参数是检查列表，第二个是本次爆破的位数i，第三个是本次爆破的字符串
返回检查是否成功，成功返回True，否则返回False
'''
def burst(final_list, check_function,num_bytes=1):
    def generate_byte_combinations(num_bytes):
        for byte_values in product(range(32, 128), repeat=num_bytes):
            yield bytes(byte_values).decode('ascii')

    flag = []
    for i in range(0,len(final_list),num_bytes):
        for byte_combination in generate_byte_combinations(num_bytes):
            if check_function(final_list,i, byte_combination):
                flag.append(byte_combination)
    return ''.join(flag)






#把字符串或普通列表或字节字符串转为自定义列表类
def tolist(x):
    if isinstance(x, str):
        return cal([ord(char) for char in x])
    elif isinstance(x, (list, bytes)):
        return cal(x)

#从hex数组求flag字符串
'''
传入任意长度每一位可以被解释为字符的字符串，输出这个字符串，如果b有flag前缀，添加后会添加flag前缀
'''
def pl(a, b=None):
    if all(0 <= x <= 255 for x in a):
        result = ''.join(chr(a[i]) for i in range(len(a)))
    else:
        result = (''.join([long_to_bytes(a[i]).decode() for i in range(len(a)-1,-1,-1)]))[::-1]

    if b is None:
        print(result)
    else:
        if result.startswith(b):
            print(result)
        elif '{' in result and '}' in result:
            print(f"{b}{result}")
        else:
            print(f"{b}{{{result}}}")

def phex(input_list):
    res = ' '.join(f'0x{x:02x},' for x in input_list)[:-1]
    print('['+res+ ']')

#xor解题
'''
传入需要异或的列表，后面传入任意个key，然后对这个列表异或，最后输出结果列表，打印字符串
如果没有key，会把列表和i位异或
key可以是数或字符串，如果是数，则是对每一位都异或这个值
如果是字符串，会对每一位异或对应位的key，如果key长度小于传入的列表，会重复

'''
def pxor(nums, *keys):
    if isinstance(nums, str):
        nums = [ord(char) for char in nums]
    result = cal([])

    if not keys:
        for i in range(len(nums)):
            result.append(nums[i] ^ i)

    for key in keys:
        if isinstance(key, int):
            for num in nums:
                result.append(num ^ key)
        elif isinstance(key, str) or isinstance(key, bytes):
            for i, num in enumerate(nums):
                xor_char = key[i % len(key)]
                result.append(num ^ ord(xor_char))
        elif isinstance(key, list):
            for i in range(len(nums)):
                result.append(nums[i] ^ key[i%len(key)])
    try:            
        pl(result)
    except:
        pass
    return list(result)


#简化多重循环
'''
示例：
for i in range(0,10):
    for j in range(0,20):
        for k in range(0,7):
等价于
for i,j,k in plop(10,20,7):
必须从0开始

'''
def plop(*args):
    return product(*(range(x) for x in args))



#取字符串/列表元素在第二个列表的位置
'''
会先转换为数字然后比较
'''
def geti(input_data, reference_list):
    if isinstance(input_data, str):
        input_data = [ord(char) for char in input_data]
    if isinstance(reference_list, str):
        reference_list = [ord(char) for char in reference_list]
    try:
        return [reference_list.index(item) for item in input_data]
    except ValueError:
        raise ValueError("不存在匹配")






#该函数将一个大的hex分割成小的值，返回整个列表
'''
输入hex数字列表,返回变成byte的列表并打印
remove_zero表示是否移除所有的0值

'''
def hextbyte(hex_list, reverse=False,remove_zero = False,is_print = False):
    byte_list = []
    for idx, hex_num in enumerate(hex_list):
        bit_length = hex_num.bit_length()
        byte_length = (bit_length + 7) // 8
        bytes = [(hex_num >> (i * 8)) & 0xFF for i in range(byte_length)]
        if reverse:
            bytes = bytes[::-1]
            if remove_zero and idx == len(hex_list) - 1 and bit_length % 8 != 0:
                bytes = list(filter(lambda x: x != 0, bytes))
        byte_list.extend(bytes)
        if is_print:
            print(f'{[hex(b) for b in bytes]}')
    return byte_list


#将一个字符串或者列表分为奇和偶返回
def oe(inpu):
    odd_numbers = cal([])
    even_numbers = cal([])
    if isinstance(inpu, str):
        inpu = [ord(char) for char in inpu]
    for num in inpu:
        if num % 2 == 0:
            even_numbers.append(num)
        else:
            odd_numbers.append(num)
    return odd_numbers, even_numbers



'''
以下为z3简写方法

'''

#z3初始化flag和out
#如果遇到mod，好像只写8有问题，不理解
def zini(length):
    from z3 import BitVec
    flag = [BitVec('flag[%d]' % i, 9) for i in range(length)]
    out = cal(flag)  
    return flag, out
#z3检查终值
def zcheck(f,flag):
    from z3 import SolverObj,Solver,sat,Or
    print(f.check())
    while(f.check()==sat):
        try:
            condition = []
            m = f.model()
            p=""
            for i in range(len(flag)):
                p+=chr(int("%s" % (m[flag[i]])))
                condition.append(flag[i]!=int("%s" % (m[flag[i]])))
            print(p)
            f.add(Or(condition))
        except:
            pass
#规定字符范围，可以提高速度
def isflag(f,flag):
    from z3 import And
    from z3 import SolverObj,Solver
    for i in range(len(flag)):
        f.add(And(flag[i]>31,flag[i]<129))

#为某一位添加约束
def addchar(s,flag,i,char):
    from z3 import SolverObj,Solver
    s.add(flag[i]==ord(char))


'''
pyc代码/code对象文件转字节码

'''

def scode(inputfile,outputfile=None):
    from marshal import marshal,loads
    from dis import dis
    fp = open(inputfile,"rb")
    data = fp.read()
    try:
        Pyobj = marshal.loads(data[16:])
    except:
        Pyobj = marshal.loads(data)
    out = dis.dis(Pyobj)
    if(outputfile!=None):
        with open(outputfile, 'w') as f:
            f.write(out)


'''
传入迷宫列表和行列，输出迷宫图

'''
def pmaze(maze,row,col):
    length=len(maze)
    block = row*col
    for k in range(0,length,block):
        for i in range(row): 
            for j in range(col):
                print(str(maze[k+i*col+j]),end="")
            print()
        print()




'''
以下是其它工具

'''

# 递归遍历
class Maze_cracker:
    class Map:
        def __init__(self,map_data,goway) -> None:
            self.map_data = map_data
            self.goway = goway
            
    def __init__(self,map,max_len) -> None:
        self.map = map
        self.max_len = max_len



#自定义base64解码
def de64(edata, ctable=None):
    if isinstance(edata, list):
        edata = ''.join(edata)
    elif isinstance(edata, bytes):
        edata = edata.decode('utf-8')

    stable = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    
    if ctable is not None:
        ttable = str.maketrans(ctable, stable)
        edata = edata.translate(ttable)
    
    decoded_bytes = base64.b64decode(edata)
    
    try:
        decoded_str = decoded_bytes.decode('utf-8')
        print(decoded_str)
    except UnicodeDecodeError:
        print(list(decoded_bytes))
    return cal(decoded_bytes)

#取md5字符串,还用解释吗
#如果参数2是1，求逆序str的md5

def pmd5(input_string, reverse_flag=0):
    import hashlib
    if reverse_flag == 1:
        input_string = input_string[::-1]
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    print(md5_hash.hexdigest())
    return md5_hash.hexdigest()




#偏移密码爆破
def pcs(enc,min,max):
    if isinstance(enc, str):
        enc = tolist(enc)
    for num in enc:
        if (num < min or num > max):
            print("越界")
            return 1
    
    output = [0]*len(enc)
    for i in range(min,max):
        for j in range(len(enc)):
            output[j] = ((enc[j]-min)+i)%(max-min)+min
        pl(output)


#凯撒密码/变异凯撒密码
def caser(encoded_text:str,offset:int,adder:int=0):
    upper_case = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    lower_case = list("abcdefghijklmnopqrstuvwxyz")
    res = ""

    for i in range(len(encoded_text)):
        if encoded_text[i] in upper_case:
            res += upper_case[(upper_case.index(encoded_text[i]) +offset+ adder*i) % 26]
        elif encoded_text[i] in lower_case:
            res += lower_case[(lower_case.index(encoded_text[i]) +offset+ adder*i) % 26]
        else:
            res += encoded_text[i]

    return res




def exgcd(a, b):
    if b == 0:
        return a, 1, 0
    else:
        gcd, x, y = exgcd(b, a % b)
        return gcd, y, x - (a // b) * y
    


#-----------------------------------
#pwn



def pwnini64(file_name, lib_name):
    from pwn import ELF,context,remote,process
    context(os='linux', arch='amd64', log_level='debug')
    e = ELF(file_name)
    
    if ":" in file_name:
        ip, port = file_name.split(":")
        p = remote(ip, int(port))
    else:
        p = process(file_name)
    
    libc = ELF(lib_name)
    return p,libc,e
def pwnini32(file_name, lib_name):
    from pwn import ELF,context,remote,process
    context(os='linux', arch='i386', log_level='debug')
    e = ELF(file_name)
    
    if ":" in file_name:
        ip, port = file_name.split(":")
        p = remote(ip, int(port))
    else:
        p = process(file_name)
    
    libc = ELF(lib_name)
    return p,libc,e
# sd = lambda s : p.send(s)
# sl = lambda s : p.sendline(s)
# sa = lambda n,s : p.sendafter(n,s)
# sla = lambda n,s : p.sendlineafter(n,s)
# rc = lambda n : p.recv(n)
# rl = lambda : p.recvline()
# ru = lambda s : p.recvuntil(s)
# ra = lambda : p.recvall()
# it = lambda : p.interactive()
# uu32 = lambda data : u32(data.ljust(4, b'\x00'))
# uu64 = lambda data : u64(data.ljust(8, b'\x00'))




'''
cry

'''
def lb(n):
    return long_to_bytes(n)
def bl(n):
    return bytes_to_long(n)
def pflag(n):
    flag = lb(n)
    print(flag.decode())
    return flag
def Srsa(p,q,e,c):
    phi = (p-1)*(q-1)
    d = pow(e,-1,phi)
    m = pow(c,d,p*q)
    m_ = long_to_bytes(m)
    print(m_)
    return m
    
    
# 已知和和差，求p,q 
def Ssumdiff(sum,dif):
    return (sum-dif)//2,(sum+dif)//2


#已知p^q和pq求p和q
def Sxor_factor(n, p_xor_q):
    def xor_factor(n, p_xor_q):
        import math
        tracked = set([(p, q) for p in [0, 1] for q in [0, 1]
                    if check_cong(1, p, q, n, p_xor_q)])

        PRIME_BITS = int(math.ceil(math.log(n, 2)/2))

        maxtracked = len(tracked)
        for k in range(2, PRIME_BITS+1):
            newset = set()
            for tp, tq in tracked:
                for newp_ in extend(k, tp):
                    for newq_ in extend(k, tq):
                        # Remove symmetry
                        newp, newq = sorted([newp_, newq_])
                        if check_cong(k, newp, newq, n, p_xor_q):
                            newset.add((newp, newq))

            tracked = newset
            if len(tracked) > maxtracked:
                maxtracked = len(tracked)
        print('Tracked set size: {} (max={})'.format(len(tracked), maxtracked))

        # go through the tracked set and pick the correct (p, q)
        for p, q in tracked:
            if p != 1 and p*q == n:
                return p, q
            
    def check_cong(k, p, q, n, xored=None):
        kmask = (1 << k) - 1
        p &= kmask
        q &= kmask
        n &= kmask
        pqm = (p*q) & kmask
        return pqm == n and (xored is None or (p^q) == (xored & kmask))

    def extend(k, a):
        kbit = 1 << (k-1)
        assert a < kbit
        yield a
        yield a | kbit
    return xor_factor(n, p_xor_q)

# 获取一个列表的元素所有可能的乘积组合(用于找正确的因数)
# 输入一个数的列表和一个检查函数(可选),检查函数需要在检查通过时返回真，否则返回假
# 返回那个数的列表的所有可能乘积


def get_comb(nums, check_fn=None):
    from functools import reduce
    from operator import mul
    result = []
    for r in range(2, len(nums) + 1):
        for combo in combinations(nums, r):
            product = reduce(mul, combo)
            result.append(product)
            if check_fn and check_fn(product):
                return result
    return result

# 通过http://www.factordb.com/ 获取已解析的数的因数，需要联网
# 输入一个数
# 输出那个数的因数列表
def fetch_factors(number):
    import requests
    def parse_factors(factor_string):
        factor_parts = factor_string.split(' ')
        if factor_parts[len(factor_parts)-1] == '':
            factor_parts.pop()
        result_factors = []
        for part in factor_parts:
            part = part.strip()
            if '^' in part:
                base, exp = part.split('^')
                result_factors.extend([int(base)] * int(exp))
            else:
                result_factors.append(int(part))
        
        return result_factors
    url = f"http://www.factordb.com/index.php?query={number}"
    response = requests.get(url)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    result = soup.find_all("font", color="#000000")
    if not result:
        return "No factors found."
    factor_string = ''
    for tag in result:
        factor_string += tag.get_text()+' '
    return parse_factors(factor_string)

# 一些密码可能需要的导入
def iroot(num,fac):
    import gmpy2
    return gmpy2.iroot(num,fac)
