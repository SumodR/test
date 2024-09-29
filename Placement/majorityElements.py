li=[2,4,2,4,2,2,2,2,2,3,3,3,3,3]
major=[]
for i in li:
    if li.count(i)>=len(li)//2:
        major.append(i)
print("Majority elements are:-")
print(set(major))

#--OR----
'''
a=array
b=list(set(a))
for i in b:
    if a.count(i)>=len(a)//2;
        print(i)

