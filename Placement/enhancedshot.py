helth=[3,4,2]
n=7
shot=1
helth.sort()
#print(helth
a=max(helth)
while(a>=1):
    if sum(helth)==3 and shot%n==0:
        print('t')
    if shot%n==0:
        for j in helth:
            print('shot',helth.index(j),":new helth",j)
            j-=1
    else:
        a-=1
        print('shott',a+1,)
    shot+=1
if sum(helth)==3 and shot%n==0:
    print('t')
else:
    print('f')
        
