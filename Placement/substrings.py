'''string="jagga"
substr=[]
for i in range(len(string)):
    for j in range(i+1,len(string)+1):
        substr.append(string[i:j])
print(substr)

count=0
for j in substr:
    if len(j)>1 and j==j[::-1]:
        print(j)
        count+=1
print(count)
'''
#-----OR-Use function for Palindrme-----

'''
a=[2,4,7,1,6,3]
n=3
for i in range(len(a)):
    for j in range(i+1,len(a)+1):
        if len(a[i:j])==n:
            print(max(a[i:j]))
#--or--'''
'''FOr i in range(len(a)-n+1):
        print(max(a[i:i+n]))'''
    

#subbarays with given sum;
a=[3,4,-7,1,3,3,1,-4]
k=7
for i in range(len(a)):
    for j in range(i+1,len(a)+1):
        if sum(a[i:j])==k:
            print(a[i:j])
