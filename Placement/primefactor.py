from math import sqrt
def prime(m):
    flag=0
    for i in range(2,m//2):
        if m%i==0:
            flag=1
            break
    if flag==1:
        return False
    return True
            

n=36
fact=[]
sum=0
for i in range(2,n):
    if n%i==0:
        fact.append(i)
print(fact)       
for j in fact:
    if prime(j):
        sum+=j
    print(j)
print("Num=",n)
print("Sum=",sum)
