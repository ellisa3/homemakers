import json

diffs = ['Dad', 'Mama', 'Uncle', 'Brother', 'Son', 'Viagra', 'She', 'Ladies', 'Ma', 'Sons', 'Prince', 'Female', 'Lions', 'Spokesman', 'King', 'PA', 'Mom', 'Sisters', 'Boys', 'MAN', 'Girls', 'Mothers', 'HE', 'Mother', 'Lady', 'Lion', 'Guy', 'Male', 'Brotherhood', 'Sir', 'Women', 'Princess', 'Father', 'Councilwoman', 'Colts', 'Men', 'Daddy', 'His', 'Brothers', 'Deer', 'Beard', 'Man', 'Bull', 'Girl', 'Husband', 'Wife', 'Queen', 'Queens', 'Bachelor', 'Councilman', 'Statesman', 'Grandma', 'Bulls', 'Sister', 'Actress', 'Chairman', 'Congressman', 'MA', 'Him', 'Daughter', 'Woman', 'Boy', 'He', 'Monk', 'Her', 'Kings']
with open('/content/homemakers/data/gender_seed.json', "r") as f:
    gender_seed_words = json.load(f)
gender_seed_words = set(gender_seed_words + diffs)
gendered_ours = set(line.strip() for line in open('./data/Ogs_predict2.txt'))

count = 0
for word in gender_seed_words:
  if word not in gendered_ours:
    

print(len(gendered_ours) + count)