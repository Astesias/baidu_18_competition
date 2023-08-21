from pysl import cmd
from pprint import pprint
import time

type_msg={
        'b':'[@1/]',
        't':'[@2/]',
        'f':'[@3/]',
        'u':'[@4/]',
        
        's':'[@5/]',
        'v':'[@6/]',
        
        'bb':'[$1/]',
        'bc':'[$2/]',
        'bg':'[$3/]',
        
        'sf':'[*70/]',
        'sb':'[*-70/]',
        
        'tf':'[&70:70/]',
        'tb':'[&-70:-70/]',
    }

while 1:
    s=input('')
    if s=='h':
        pprint(type_msg)
        continue
    else:
        if s.isdigit():
            s=int(s)
            msg = f'[:{s:.0f}/]'
        else:
            msg = type_msg[s]
    print(msg)
    for _ in range(3):
        cmd(f'echo {msg} > MSG.txt')
        time.sleep(0.2)