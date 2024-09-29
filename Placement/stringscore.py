string="asdfg htth jklm rrtrr qwerty"
splitstr=string.split()
#for testing--print(sp)
score=0
for i in splitstr:
    if i==i[::-1] and len(i)==4:
        score+=5
    if i==i[::-1] and len(i)==5:
        score+=10
print(score)
