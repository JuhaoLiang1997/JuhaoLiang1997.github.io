---
layout: post
title: Text and Typography
date: 2022-03-19 22:54 +0800
category: [Common]
tags: [typography, tools]
typora-root-url: "../_images"
math: true
comments: false
toc: true
pin: false
---
这个 post 主要用于记录 markdown 的写法，以及利用第一篇 post 的发布进行练手，熟悉完善整个利用 github page 写 blog 的 workflow。

# Typography

## Title 2

---

### Basic font

[link](https://juhaoliang1997.github.io), **bold**, *italic*, ~~delete~~, `high`, <kbd>box</kbd>, <https://juhaoliang1997.github.io/>, [^footnote], `inline code`

emoji: :star2: :monkey:

---

### Lists
1. Firstly
2. Secondly
3. Thirdly

---

### Unordered list
- Chapter
	- Section
		- Paragraph

---

### Checklist
- [ ] TODO 1
- [x] TODO 2
  - [ ] TODO 2.1


---

### Block Quote

> This line shows the _block quote_.

---

### Tables

| Company                      | Contact          | Country |
|:-----------------------------|:-----------------|--------:|
| Alfreds Futterkiste          | Maria Anders     | Germany |
| Island Trading               | Helen Bennett    | UK      |
| Magazzini Alimentari Riuniti | Giovanni Rovelli | Italy   |

---

### Images

- default

![screenshot](/2022-03-19-text-and-typography/screenshot-7697724.png){: width="972" height="589" }
_image1_



<br>

- Shadow

![screenshot](/2022-03-19-text-and-typography/screenshot-7697724.png){: .shadow width="1548" height="864" style="max-width: 90%" }
_shadow effect (visible in light mode)_

<br>

- Left aligned

![screenshot](/2022-03-19-text-and-typography/screenshot-7697724.png){: width="972" height="589" style="max-width: 70%" .normal}

<br>

- Float to left

  ![screenshot](/2022-03-19-text-and-typography/screenshot-7697724.png){: width="972" height="589" style="max-width: 200px" .left}
  "A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space."

<br>

- Float to right

  ![screenshot](/2022-03-19-text-and-typography/screenshot-7697724.png){: width="972" height="589" style="max-width: 200px" .right}
  "A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space. A repetitive and meaningless text is used to fill the space."

<br>

---

### [Mathematics](https://zhuanlan.zhihu.com/p/261750408)

$$ \sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6} $$

When $a \ne 0$, there are two solutions to $ax^2 + bx + c = 0$ and they are

$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

$x^2$ 

---

### Code block

#### Common

```
This is a common code snippet, without syntax highlight and line number.
```

#### Specific Languages

#### Console

```console
$ env |grep SHELL
SHELL=/usr/local/bin/bash
PYENV_SHELL=bash
```

#### Shell

```bash
if [ $? -ne 0 ]; then
    echo "The command was not successful.";
    #do the needful / exit
fi;
```

#### Specific filename

```sass
@import
  "colors/light-typography",
  "colors/dark-typography"
```

{: file='_sass/jekyll-theme-chirpy.scss'}

---

### Reverse Footnote

[^footnote]: The footnote source
[^footnote]: note