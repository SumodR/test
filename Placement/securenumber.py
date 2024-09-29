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

#to find operands of sum...
def operandof(a):
    ab=[]
    for i in range(6,int(a/2)+1):
        ab.append([i,a-i])
    return ab

#to find prime factors..
def  factors(sumof):
    for k in sumof:
            for n in k:
                fact=[]
                sum=0
                print('set-',n)
                for i in range(2,n):
                    if n%i==0:
                        if prime(i)and prime(int(n/i)):
                            if i!=int(n/i):
                                return True
            return False

                    
a=41
sumof=operandof(a)
print(sumof)
factors(sumof[1])





