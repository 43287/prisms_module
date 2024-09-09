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
    from z3 import SolverObj,Solver
    print(f.check())
    while(f.check()==sat):
        condition = []
        m = f.model()
        p=""
        for i in range(len(flag)):
            p+=chr(int("%s" % (m[flag[i]])))
            condition.append(flag[i]!=int("%s" % (m[flag[i]])))
        print(p)
        f.add(Or(condition))
#规定字符范围，可以提高速度
def isflag(f,flag):
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
    from marshal import loads
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
以下是其它工具

'''


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
    print(flag)
    return flag
def Srsa(p,q,e,c):
    phi = (p-1)*(q-1)
    d = pow(e,-1,phi)
    m = pow(c,d,p*q)
    m_ = long_to_bytes(m)
    print(m_)
    
# 已知和和差，求p,q 
def Ssumdiff(sum,dif):
    return (sum-dif)//2,(sum+dif)//2


#已知p^q和pq求p和q
def Sxor_factor(n, p_xor_q):
    def xor_factor(n, p_xor_q):
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
