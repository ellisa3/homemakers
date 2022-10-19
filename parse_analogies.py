# to parse analogies copy and pasted from appendix G
# prints the x values of the analogies generated *after* the hard debiasing
with open('analogies.txt', 'r') as f:
    lines = f.readlines()
    j = 0
    for line in lines:
        i = 0
        x = ""
        after_x = []
        while line[i] != ":":
            x += line[i]
            i += 1
        after_x.append(x)
        j += 1
        print(x) #148
f.close()

print("___________________")

# prints the x values of the analogies generated *before* the hard debiasing
with open('analogies.txt', 'r') as f:
    lines = f.readlines()
    j = 0
    for line in lines:
        i = 0
        white_space = 0

        while white_space < 3:
            if line[i] == " ":
                white_space += 1
            i += 1


        x = ""
        before_x = []
        while line[i] != ":":
            x += line[i]
            i += 1
        before_x.append(x)
        j += 1
        print(x)
    print(j)
f.close()
    

#     actress
# nurse
# homemaker
# adorable
# aunt
# aunts
# babe
# beautiful
# beauty
# homemaker

