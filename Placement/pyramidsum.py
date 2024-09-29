def pyramidsum(li):
    s=[]
    for i in range(len(li)-1):
        s.append(li[i]+li[i+1])
        #print(s)
    if len(s)!=1:
        return pyramidsum(s)
    else:
        return s

li=[1,2,3]
result=pyramidsum(li)
print(result)
