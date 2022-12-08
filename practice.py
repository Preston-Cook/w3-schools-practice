from __future__ import print_function

# Question 1
print('Twinkle, twinkle, little star,\n\tHow I wonder what you are!\n\t\tUp above the world so high,\n\t\tLike a diamond in the sky.\nTwinkle, twinkle, little star,\n\tHow I wonder what you are')

# Question 2
import subprocess

result = subprocess.run(['python3.11', '--version'], stdout=subprocess.PIPE)

print(result.stdout.decode('utf-8'))

# Alternative solution
import sys

print(sys.version)

# Question 3
from datetime import datetime

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Question 4
from math import pi

r = float(input('Enter a radius: '))
print(f'Area = {pi * r ** 2}')

# Question 5
first, last = input('Enter your first name: '), input('Enter your last name: ')

print(f'Hello, {last} {first}')

# Question 6
user_input = input('Enter a list of numbers separated by commas: ')

lst = user_input.split(',')
tup = tuple(lst)

print(f'''
List: {lst}
Tuple: {tup}''')

# Question 7
import re 

f = input('Enter a file name: ')

match = re.search(r'(?<=.)[0-9a-z]+$', f)

print(match.group(0))

# Question 8
color_list = ["Red","Green","White" ,"Black"]

print(color_list[0], color_list[-1])

# Question 9

exam_st_date = tuple(input('Enter a date separated by commas: ').split(','))
print(' / '.join(exam_st_date))

# Question 10
n = input('Enter an integer: ')

total = 0
for i in range(1, 4):
    total += int(n * i)

print(total)

# Question 11
func_name = input('Enter a function name: ')
func = eval(func_name)
print(func.__doc__)

# Question 12
import calendar
from datetime import datetime

dt = datetime.now()
m, y = dt.month, dt.year

print(calendar.month(y, m))

# Question 13
print('''
a string you don't have to escape
This 
is a ...... multi-line
heredoc string -------> example
''')

# Question 14
from datetime import datetime

dates = input('Enter a starting date: '), input('Enter an ending date: ')
d1, d2 = [tuple(item.split(',')) for item in dates]

start = datetime.strptime(f'{d1[0]}-{d1[1]}-{d1[2]}', '%Y-%m-%d')
end = datetime.strptime(f'{d2[0]}-{d2[1]}-{d2[2]}', '%Y-%m-%d')

diff = end - start
print(f'Difference in Days: {diff.days}')

# Question 15
from math import pi

def spherical_volume(r):
    return (4/3) * pi * r ** 3

r = float(input('Enter a radius of a sphere: '))

print(f'The volume is {spherical_volume(r):.2f}')

# Question 16

def seventeen_dif(num):
    dif = abs(17 - num)
    if dif > 17:
        return dif * 2
    return dif

# Question 17
def inRange(num):
    return num in range(900, 1101) or num in range(1900, 2101)

# Question 18
def threeSum(n1, n2, n3):
    total = sum(n1, n2, n3)
    if n1 == n2 and n2 == n3:
        return total * 3
    return total

# Question 19
def lsFormat(s1):
    if s1.startswith('ls'):
        return s1
    return 'ls' + s1

# Question 20
def copy_str(s1, n):
    return s1 * n

# Question 21
def oddCheck(n):
    if n % 2:
        return 'The number is odd'
    return 'The number is even'
oddCheck(3)

# Question 22
def fourCount(lst):
    return lst.count(4)

# Question 23
def str_copy(s1, n):
    if len(s1) < 2:
        return s1 * n
    return s1[0:2] * n

# Question 24
def vowelCheck(char):
    return char.lower() in 'aeiou'

# Question 25
def inList(lst, n):
    return n in lst

# Question 26
import matplotlib.pyplot as plt

def generateHistogram(lst):
    plt.hist(lst)
    plt.show()
    return 

# Question 27

data = [1.7,1.8,2.0,2.2,2.2,2.3,2.4,2.5,2.5,2.5,2.6,2.6,2.8,
        2.9,3.0,3.1,3.1,3.2,3.3,3.5,3.6,3.7,4.1,4.1,4.2,4.3]

def concatLst(lst):
    return ''.join(map(str, lst))

# Question 28
def evenLst(lst):
    for num in lst:
        if num % 2 == 0 and num < 237:
            print(num)


# Question 29
color_list_1 = set(["White", "Black", "Red"])
color_list_2 = set(["Red", "Green"])
color_list_1.difference(color_list_2)

# Question 30
def area(b, h):
    return 0.5 * b * h

# Question 31

def gcd(n1, n2):
    small_num = min(n1, n2)
    factor_lst = []
    for i in range(1, small_num + 1):
        if n1 % i == 0 and n2 % i == 0:
            factor_lst.append(i)
    return max(factor_lst)

# Question 32

def lcm(x, y):
    min_num, max_num = min(x, y), max(x, y)
    i = 1

    while True:
        prod = min_num * i
        if prod % max_num == 0:
            return prod
        i += 1

# Question 33
def threeSum(n1, n2, n3):
    if n1 == n2 or n2 == n3 or n1 == n3:
        return 0
    return sum(n1, n2, n3)

# Question 34
def twoSum(x, y):
    return 20 if x + y in range(15, 21) else x + y

# Question 35

# Question 36
def integerAdd(x: int, y: int) -> int:
    assert isinstance(x, int) and isinstance(y, int)
    return x + y

# Quesiton 37
class Person:
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address
    
    def __str__(self):
        return f'{self.name}\n{self.age}\n{self.address}'

# Question 38
def squareThis(x, y):
    return (x + y) ** 2

# Question 39
def calculatePrincipal(r, t, amt):
    return round(amt * (1 + r / 100) ** t, 2)

calculatePrincipal(3.5, 7, 10000)

# Question 40
from math import sqrt

def computeDistance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return round(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)

computeDistance((3, 2), (5, 6))


# Question 41
def fileExists(fname):
    try:
        open(fname)
        return True
    
    except FileNotFoundError:
        return False

# Question 42
import sys

if sys.maxsize > 2 ** 32:
    print('64 Bit')
else:
    print('32 Bit')

# Question 43
import platform, os

print(os.name, platform.system(), platform.release())

# Question 44
import site

site.getsitepackages()

# Question 45
from subprocess import call
call(['pwd'])

# Question 46
import sys

sys.argv[0].split('/')[-1]

# Question 47
import multiprocessing

multiprocessing.cpu_count()


# Quesiton 48
print(float(input('Enter a number: ')))

# Question 49
import os

os.listdir(input('Enter a path to a directory: '))

# Question 50
print(input('Enter a sample string: '), end='')

# Question 51
import cProfile

def prod(x, y):
    return x * y

cProfile.run('prod(4, 3)')

# Question 52
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

eprint("abc", "efg", "xyz", sep="--")

# Question 53
import os
os.environ

# Question 54
import getpass

getpass.getuser()

# Question 55
import socket

hostname = socket.gethostname()
socket.gethostbyname(hostname)

# Question 56

def terminal_size():
    import fcntl, termios, struct
    th, tw, hp, wp = struct.unpack('HHHH',
        fcntl.ioctl(0, termios.TIOCGWINSZ,
        struct.pack('HHHH', 0, 0, 0, 0)))
    return tw, th

print('Number of columns and Rows: ',terminal_size())


# Question 57
from datetime import datetime

start = datetime.now()
print('hello, world')
print(datetime.now() - start)


# Question 58
def sumFirstN(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    
    return total

# Question 59
def convertImperial(feet, inches):
    return (inches + feet * 12) * 2.54

# Question 60
from math import sqrt

def pythag(a, b):
    return sqrt(a ** 2 + b ** 2)

# Question 61
def convertFeet(feet):
    print(f'''
Inches: {feet * 12}
Yards: {feet / 3}
Miles: {feet / 5280}''')

convertFeet(6)

# Question 62
from datetime import datetime

#get current date
dt = datetime.today()  
seconds = dt.timestamp()
print(seconds)


# Question 63
import os

sys.argv[0]

# Question 64
import os.path, time
print("Last modified: %s" % time.ctime(os.path.getmtime("w3-practice.ipynb")))
print("Created: %s" % time.ctime(os.path.getctime("w3-practice.ipynb")))

# Question 65
time = float(input("Input time in seconds: "))
day = time // (24 * 3600)
time = time % (24 * 3600)
hour = time // 3600
time %= 3600
minutes = time // 60
time %= 60
seconds = time
print("d:h:m:s-> %d:%d:%d:%d" % (day, hour, minutes, seconds))


# Question 66
def bmi(height, weight):
    return weight / height ** 2

# Question 67
def pressureConvert(kp):
    return (kp / 6.895, kp / 7.501, kp / 101.3)

# Question 68
def sumDigits(num):
    total = 0
    for char in str(num):
        total += int(char)
    return total


# Question 69
x = int(input("Input first number: "))
y = int(input("Input second number: "))
z = int(input("Input third number: "))

a1 = min(x, y, z)
a3 = max(x, y, z)
a2 = (x + y + z) - a1 - a3
print("Numbers in sorted order: ", a1, a2, a3)

# Question 70
import os
files = input('Enter a series of files separated by commas: ').split(',')
sorted(files, key=os.path.getmtime)

# Question 71
import os

files = os.listdir('.')
sorted(files, key=os.path.getmtime)

# Question 72
import math
dir(math)

# Qustion 73
def getMidpoint(p1, p2):
    x1, y1, = p1
    x2, y2 = p2
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# Question 74
import hashlib

user_input = input('Enter a string to be hashed').encode('utf-8')
print(hashlib.sha256(user_input).hexdigest())


# Question 75
import sys
print(sys.copyright)

# Question 76
import sys
sys.argv

# Question 77
import sys
sys.byteorder

# Question 78
import sys

sys.builtin_module_names

# Question 79
import sys

sys.getsizeof(input())

# Question 80
import sys
sys.getrecursionlimit()

# Question 81
def concat_str(*args):
    return ''.join(args)

# Question 82
def sum_of_items(item):
    return sum(item)

# Question 83
def lst_test(lst, num):
    return all(item > num for item in lst)

# Question 84
def char_count(s1, char):
    return s1.count(char)

# Question 85
import os

path = input('Enter a path to a file or directory: ')

if os.path.isfile(path):
    print('It is a path to a file')
elif os.path.isdir(path):
    print('It is the path to a directory')

# Question 86
ord(input('Enter a character: '))

# Question 87
import os

path = input('Input the path to a file: ')
os.path.getsize(path)

# Question 88
def str_addition(x, y):
    return f'{x} + {y} = {x + y}'

# Question 89
if 1 == 1:
    print('hello, world')

# Question 90
with open(sys.argv[0].split('/')[-1]) as f:
    print(f.read())

# Question 91
x, y = input(), input()
x, y = y, x

# Question 92
print()
print("\#{'}${\"}@/")
print("\#{'}${"'"'"}@/")
print(r"""\#{'}${"}@/""")
print('\#{\'}${"}@/')
print('\#{'"'"'}${"}@/')
print(r'''\#{'}${"}@/''')
print()

# Question 93
val = input()

val, type(val), id(val)


# Question 94
bytes = b'asdf'
list(bytes)

# Question 95
input().isdigit()

# Question 96
import traceback
print()
def f1():
    return abc()
def abc():
    traceback.print_stack()
f1()
print()

# Question 97
s_var_names = sorted((set(globals().keys()) | set(__builtins__.__dict__.keys())) - set('_ names i'.split()))
print()
print( '\n'.join(' '.join(s_var_names[i:i+8]) for i in range(0, len(s_var_names), 8)) )
print()

# Question 98
from datetime import datetime

str(datetime.now())

# Question 99
os.system('clear')

# Question 100
import socket
socket.gethostname()

# Question 101
import requests

requests.get(input('Enter a URL')).text


# Question 102
import subprocess

process = subprocess.Popen(['ls', '-l'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

stdout, _ = process.communicate()

print(str(stdout.decode()))


# Question 103
import sys

abs_path = input('Enter an absolute path: ')
pieces = abs_path.split('/')
print(pieces[-1])

# Question 104

import os
print("\nEffective group id: ",os.getegid())
print("Effective user id: ",os.geteuid())
print("Real group id: ",os.getgid())
print("List of supplemental group ids: ",os.getgroups())
print()

# Question 105
import os

print(os.environ)

# Question 106
abs_path = input('Enter an absolute path: ')
pieces = abs_path.split('/')
pieces[-1].rsplit('.', 1)


# Question 107
import os.path
import time

print('File         :', __file__)
print('Access time  :', time.ctime(os.path.getatime(__file__)))
print('Modified time:', time.ctime(os.path.getmtime(__file__)))
print('Change time  :', time.ctime(os.path.getctime(__file__)))
print('Size         :', os.path.getsize(__file__))


# Question 108
import os

path = input('Enter a path to a file or directory: ')

if os.path.isfile(path):
    print('It is a path to a file')
elif os.path.isdir(path):
    print('It is the path to a directory')


# Question 109
user_input = float(input('Enter a number: '))

if user_input < 0:
    print('Negative')
elif user_input > 0:
    print('Positive')
else:
    print('The number is zero')

# Question 110
lst = [1,2,3,50,60,75,150]

list(filter(lambda x: x % 15 == 0, lst))

# Question 111
import glob

glob.glob('*.*')

# Question 112
lst = [1,2,3]
del lst[0]
lst

# Question 113

try:
    float(input('Enter a number: '))
except:
    print('You did not enter a number')

# Question 114
lst = [1,2,3,-43.43,-9,5,3,1,123]

list(filter(lambda x: x > 0, lst))

# Question 115
from functools import reduce

lst = [1,2,3,-43.43,-9,5,3,1,123]

reduce(lambda x, y: x * y, lst)

# Question 116
def str_to_unicode(s: str):
    return s.encode("unicode_escape").decode()
print(str_to_unicode('Python is Great '))

# Question 117
s1 = "Python"
s2 = "Python"

hex(id(s1)) == hex(id(s2))

# Question 118
arr = [num for num in range(0, 256)]
vals = bytearray(arr)

for val in vals:
    print(val)

# Question 119
from math import pi

places = input('Enter a number of places past the decimal: ')

float(format(pi, f'.{places}f'))

# Question 120
str_num = "1234567890"
print("Original string:",str_num)
print('%.6s' % str_num)
print('%.9s' % str_num)
print('%.10s' % str_num)

# Question 121
try:
    x
except NameError:
    print('x is not defined')


# Question 122
n = 20
d = {"x":200}
l = [1,3,5]
t= (5,7,8)
print(type(n)())
print(type(d)())
print(type(l)())
print(type(t)())

# Question 123
import sys
print("Float value information: ",sys.float_info)
print("\nInteger value information: ",sys.int_info)
print("\nMaximum size of an integer: ",sys.maxsize) 

# Question 124

x, y, z = 1,2,3

tup = (x, y, z)

if all(num == x for num in tup):
    print('All same')
else:
    print('Not all same')

# Question 125
from collections import Counter

s1 = Counter('hello, world')

sum(s1.values())


# Question 126
from inspect import getmodule
from math import sqrt
print(getmodule(sqrt))

# Question 127
num = int(input('Enter a number: '))

num in range(-(2 ** 32), 2 ** 32 + 1)

# Question 128
from string import ascii_lowercase

s1 = input('Enter a string: ')

any(char in ascii_lowercase for char in s1)

# Question 129
input().rjust(4, '0')

# Question 130
lst = [1, 2, 3, 'asdf']

for item in lst:
    if isinstance(item, str):
        print(f'"{item}"')


# Question 131
var_list = ['a', 'b', 'c']
x,y,z=var_list
print(x,y,z)

# Question 132
import os.path
print(os.path.expanduser('~'))

# Question 133
from datetime import datetime

start = datetime.now()
print('Hello, world!')
dif = datetime.now() - start

print(dif)


# Question 134
x, y = map(int, (input(), input()))

print(f'{x}, {y}')

# Question 135
val = input()
print(f'Value of x is "{val}"')

# Question 136
import os
rel = input('Enter an absolute path to a directory: ')
for file in os.listdir():
    if not os.path.isdir(rel + file):
        print(file)


# Question 137
d = {'Red': 'Green'}
(c1, c2), = d.items()
print(c1)
print(c2)

# Question 138
int(True)
int(False)

# Question 139
import re

pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'

print(bool(re.match(pattern, input())))

# Question 140
x = 12

bin(12)[2:].rjust(8, '0')

# Question 141
hex(255)[2:]

# Question 142

def consec_zero_test(s1):
    zero_count = 0

    for char in s1:
        if char != '0':
            break
        zero_count += 1

    if len(s1) % zero_count != 0 or len(set(s1)) != 2:
        return False
    
    for i in range(len(s1) // zero_count):
        slice = s1[i * zero_count: (i + 1) * zero_count]
        if i % 2 == 0 and slice.count('0') != zero_count:
            return False
        elif i % 2 == 1 and slice.count('1') != zero_count:
            return False
    return True

consec_zero_test(s1)

# Question 143
import struct
print(struct.calcsize("P") * 8)

# Question 144
x = '2'

isinstance(x, int) or isinstance(x, str)

# Question 145
x = set()

isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set)

# Question 146
import imp
print(imp.find_module('os'))

# Question 147
def isDivisible(x, y):
    return x % y == 0

# Question 148
max_num, min_num = float('-inf'), float('inf')

lst = [1,2,3,4,4,34,1123,43,-123,43,21]
for num in lst:
    max_num = num if num > max_num else max_num
    min_num = num if num < min_num else min_num

print(max_num, min_num)


# Question 149
total = 0
num = int(input('Enter a positive number :'))

for i in range(1, num):
    total += i ** 3

print(total)

# Question 150
lst = [1,2,3,4,4,34,1123,43,-123,43,21]

def distinct_odd_prod(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] != nums[j] and (nums[i] * nums[j]) % 2 == 1:
                return True
    return False

distinct_odd_prod([1, 6, 4, 7, 8])