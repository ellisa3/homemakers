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
    with open('./data/gender_specific_predict2', "r") as f:
        gendered_theirs = json.load(f)

    print("length of gendered_theirs", len(gendered_theirs))

    gendered_ours = set(line.strip() for line in open('./data/gender_specific_predict.txt'))
    print("length of gendered_ours", len(gendered_ours))
    # print(gendered_ours)
    # print("\n")
    # print(gendered_theirs)

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

main()