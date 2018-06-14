#File randomizer
import random
import shutil


x = [i for i in range(1,3001)]

for i in range(600):
    a = random.randint(1, 3000)
    while a not in x:
        a = random.randint(1, 3000)
    x.remove(a)

    src = "asl_alphabet_train/A/A%s.jpg"%a
    dst = "asl_alphabet_test/A"
    shutil.move(src, dst)




















