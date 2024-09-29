#print left rotn.n steps.
a=[1,2,3,4,5]
n=12
for j in range(n):
    temp=a[0]
    for i in range(len(a)-1):
        a[i]=a[i+1]
    a[-1]=temp
print(a)

#orrr

a=[1,2,3,4,5]
n=2
n=n%len(a) #insted of 12 loops, do only 2 loops.
print(a[n:]+a[:n])
