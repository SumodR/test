#print nonrepetin.pair for n-sum
a=[1,2,3,4,5,6]
n=5
for i in range(len(a)):
    if n-a[i] in a[i+1:]:
        print([a[i],n-a[i]])
