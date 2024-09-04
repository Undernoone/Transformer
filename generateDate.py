import random

# 英文只有26个为什么用29，因为有BosEos和Pad
vocab_list = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n',
             'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']
char_list = vocab_list[3:]  # Removing BOS and EOS and Padding from the list
bos_token = "[BOS]"  # Beginning of sentence token
eos_token = "[EOS]"  # End of sentence token
pad_token = "[PAD]"  # Padding token

# For an English string ,output string after
# abcfg --> baxwv
source_path = "source.txt"
target_token = "target.txt"
with open(source_path, "w") as f:
    pass

with open(target_token, "w") as f:
    pass

for _ in range(10000):
    source_str = ""
    target_str = ""
    for index in range(random.randint(3, 13)): # random length of source and target string from 3 to 13
        i = random.randint(0, 25)  # Index to access al phabetic characters only
        source_str += char_list[i]
        target_str += char_list[(i + 26 - 5) % 26]  # Use the 5th character of the alphabetic characters as target
    target_str = target_str[::-1]
    with open(source_path, "a") as f:
        f.write(source_str + "\n")


    with open(target_token, "a") as f:
        f.write(target_str + "\n")












    # for _ in range(random.randint(3, 13)):
    #     i = random.randint(3, 28)  # Index to access alphabetic characters only
    #     source_str += char_list[i]
    #     target_str += char_list[(i + 26 - 5) % 26 + 3]  # Use the
    # target_str = bos_token + target_str[::-1] + eos_token
    # with open(source_path, "a") as f:
    #     f.write(source_str + "\n")
    #
    # with open(target_token, "a") as f:
    #     f.write(target_str + "\n")


