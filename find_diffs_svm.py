# find diffs between their gender neutral words and our gender neutral words

# load gender_specific_full and fins which ones overlap

import json

class FindDiffs:
  def __init__(self):
    self.missing_words = set()
    self.commonalities = set()
    self.false_words = set()

  def find_diffs(self):
    gendered_theirs = set()

    diffs = ['Dad', 'Mama', 'Uncle', 'Brother', 'Son', 'Viagra', 'She', 'Ladies', 'Ma', 'Sons', 'Prince', 'Female', 'Lions', 'Spokesman', 'King', 'PA', 'Mom', 'Sisters', 'Boys', 'MAN', 'Girls', 'Mothers', 'HE', 'Mother', 'Lady', 'Lion', 'Guy', 'Male', 'Brotherhood', 'Sir', 'Women', 'Princess', 'Father', 'Councilwoman', 'Colts', 'Men', 'Daddy', 'His', 'Brothers', 'Deer', 'Beard', 'Man', 'Bull', 'Girl', 'Husband', 'Wife', 'Queen', 'Queens', 'Bachelor', 'Councilman', 'Statesman', 'Grandma', 'Bulls', 'Sister', 'Actress', 'Chairman', 'Congressman', 'MA', 'Him', 'Daughter', 'Woman', 'Boy', 'He', 'Monk', 'Her', 'Kings']
    gendered_ours = list(set(line.strip() for line in open('./data/Ogs_predict.txt')))
    gendered_theirs = set(line.strip() for line in open('./data/Tgs_predict.txt'))
    gendered_ours = set(gendered_ours + diffs)

    print("length of gendered theirs", len(gendered_theirs))
    print("length of gendered ours", len(gendered_ours))

    for word in gendered_theirs:
      if word not in gendered_ours:
        self.missing_words.add(word)
      else:
        self.commonalities.add(word)

    for word in gendered_ours:
      if word not in gendered_theirs:
        self.false_words.add(word)

def main():
  fd = FindDiffs()
  fd.find_diffs()
  print("Number of words missing in our set: ", len(fd.missing_words))

  print("Number of commonalities: ", len(fd.commonalities))
  print("Number of false words in our set: ", len(fd.false_words))
  print(fd.missing_words)

main()