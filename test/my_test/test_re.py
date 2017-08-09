import re
re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
s = '“老百姓大药房的药很便宜”，很适合老百姓去买。\r\n'
blocks = re_han_internal.split(s)
print(blocks)
