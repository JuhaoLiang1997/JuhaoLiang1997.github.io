#!/usr/bin/env python3

import os
import re

root = 'https://juhaoliang1997.github.io/'
# header
header = "# [**JuhaoLiang Blog**]({})".format(root)

# files
file_path = '_posts'
dir_list = [f for f in os.listdir(file_path) if f.endswith('.md')]
dir_list = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(file_path, x)))

post_root = os.path.join(root, 'posts')
def link_of(title):
    return os.path.join(post_root, title)

filename_pattern = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}-(.+?)\.md')
def title_and_link_of(filename):
    title = filename_pattern.search(filename).group(1)
    link = link_of(title)
    title = " ".join([t.capitalize() for t in title.split('-')])
    return (title, link)

posts = map(title_and_link_of, dir_list)
posts_md = map(lambda post: "- [{}]({})".format(post[0], post[1]), posts)
post_section = "## Posts\n\n" +  "\n".join(posts_md)

# footer
license = "## License\n\nThis work is published under [MIT]({}) License.".format(root + 'blob/main/LICENSE')

whole_md = "\n".join([header, post_section, license])
with open("README.md", 'w') as readme:
    readme.write(whole_md)

os.system("git reset HEAD")
os.system("git add README.md")
os.system('git commit -m "update readme"')
os.system('git push origin main')
os.system('git pull')