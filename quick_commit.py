#!/usr/bin/env python3

import os
import re

def stop_commit():
    print("Interrupt...")
    quit()

os.system("git reset HEAD")
print("git reset HEAD...")
status = os.popen("git status -su").readlines()
files = [s.split()[-1].strip() for s in status]
images = [f for f in files if f.startswith('_images/')]
posts = [f for f in files if f.startswith('_posts/')]

print("\n".join(["{}: {}".format(index, post) for index, post in enumerate(posts)]))
i = int(input("Which one to commit: "))
if i > len(posts):
    stop_commit()

filename = re.search(r'\/(.+?)\.md', posts[i]).group(1)
commit_files = [f for f in files if filename in f]
print("Committing following files: ")
print("\n".join(commit_files))
if not input("Confirm with [y/n]: ").lower() == "y":
    stop_commit()

os.system("git add {}".format(" ".join(commit_files)))
os.system('git commit -m "post: {}"'.format(filename))
os.system('git push origin main')
