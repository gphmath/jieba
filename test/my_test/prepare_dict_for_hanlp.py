import re
pattern = re.compile('^(\S+)(\s)(\d+)(\s)([a-z]+)$')
match = pattern.match('王石 12 nr')
print(' '.join(match.group(1, 5, 3)))
with open(file='dict.txt', mode='r', encoding='utf-8') as txt:
    dict_list = txt.read().splitlines()
new_dict = ''
for item in dict_list:
    match = pattern.match(item)
    new_dict += ' '.join(match.group(1, 5, 3))+'\n'
with open('new_dict.txt',mode='w',encoding='utf-8') as new_txt:
    new_txt.write(new_dict)

