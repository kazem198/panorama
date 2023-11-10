myarray = ["r", "t", "y"]
c = []

for i in range(len(myarray)):
    if i == 0:
        x = myarray[i]
        y = myarray[i+1]
        c.append(x+y)
    else:
        c.append(c[i-1] + myarray[i])
print(c)
