---
layout: post
title: Gitpage Blog Setup
date: 2022-03-19 23:30 +0800
category: [Common]
tags: [tools, script]
typora-root-url: "../_images"
math: true
comments: false
toc: true
pin: false
---

这一篇记录一下整个建立 github-page blog 的流程，以及针对个人习惯做了怎么样的改动。

**Keywords**: mac m1, [GitHub Pages](https://pages.github.com/), [jekyll-theme-chirpy](https://github.com/cotes2020/jekyll-theme-chirpy), [jekyll-compose](https://github.com/jekyll/jekyll-compose), [Google Analytics](https://analytics.google.com/analytics/web/)

**参考链接**：[GitHub Pages 搭建教程](https://sspai.com/post/54608), [Chirpy Getting Start](https://chirpy.cotes.page/posts/getting-started/), 



## 搭建初衷

最近在忙着做学校的 coursework，以及准备工作面试，无意中在知乎刷到一篇关于搭建个人技术博客的文章，就想着搭一个试试看（种树/十年前/现在）。

## GitHub Pages

### 新建 Github Pages 仓库

- 首先需要找一份自己喜欢的 [**模版**](http://jekyllthemes.org/)，本文使用 [**Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy)，根据 [指导](https://chirpy.cotes.page/posts/getting-started/) 从 [**Chirpy Starter**](https://github.com/cotes2020/chirpy-starter/generate) fork 一个新仓库，取名为`<GH_USERNAME>.github.io`，`GH_USERNAME` 是 Github 用户名

### 本地环境配置

1. 本地安装 `ruby`，检查 **路径** 和 **版本**，安装 `jekyll`，检查版本

   ```console
   $ brew install ruby
   $ which ruby && ruby -v
   $ gem install jekyll bundler
   $ jekyll -v
   ```

1. 把刚刚 fork 的 repo 拉到本地 `git clone xxx.git`，安装依赖

   ``` console
   $ bundle
   ```

1. 配置个人信息，在 **_config.yml** 文件中 配置 title, tagline, 以及其他你想要展示的社交平台信息(twitter, github, email)，还有很多客制化的东西，具体看文件内注释，这一步不影响

1. 本地运行预览，启动服务器后，浏览器打开 *[http://127.0.0.1:4000](http://127.0.0.1:4000/)* 即可看到效果

   ``` console
   $ bundle exec jekyll s
   ```

1. 部署到 github 上，首先检查以下文件是不是都在目录里，完成后就可以 `commit` 和 `push` 上去了

   - **.github/workflows/pages-deploy.yml**，如果没有的话，创建一个新的，把 [sample file](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/.github/workflows/pages-deploy.yml.hook) 拷贝进去，`on.push.branches` 的值要和你仓库默认分支名字相同
   - **tools/deploy.sh**，没有的话拷贝[一份](https://github.com/cotes2020/jekyll-theme-chirpy/tree/master/tools)
   - 建议不要上传 `Gemfile.lock`，在 `.gitignore` 添加一下忽略扫描就好

1. 发布你的网站

   - `push` 之后，等 build 完了，就会出现一个新的分支 `gh-pages` 用于存储网站文件

   - 打开 *Settings* -> *Pages* -> *Source* 选择 `gh-pages/root` 作为发布源，点击保存

     <img src="/2022-03-19-gitpage-blog-setup/publish_source.png" alt="publish_source" style="zoom:50%;" />
     _publishiing source_

   - 就可以点开上方链接浏览你的网站了

   - 如果远程部署有问题的话，可以打开 *Actions* -> *Workflows* 检查部署问题

### 流量监控

- 这里使用的是 [**Google Analytics**](https://analytics.google.com/analytics/web/) 进行流量监控，小博客没有流量，这个也不用在意...

- 创建 **GA** 账号，创建 **Property**，到 *Set up Data Stream* 的位置输入你的网址URL，创建数据流，获得 **Measurement ID** 类似 `G-V6XXXXXXXX`，复制到 `_config.yml` 文件中

  ```yaml
  google_analytics:
    id: 'G-V6XXXXXXX'   # fill in your Google Analytics ID
    # Google Analytics pageviews report settings
    pv:
      proxy_endpoint:   # fill in the Google Analytics superProxy endpoint of Google App Engine
      cache_path:       # the local PV cache data, friendly to visitors from GFW region
  ```

- `commit & push & deploy` 之后就可以在 Google Analytics 中实时查看你的网址流量了

- 如果你想更进一步，使得页面显示浏览量的话，可以参考 [**Chirpy Guide**](https://chirpy.cotes.page/posts/enable-google-pv/)

### 配置网站图标

- 网站图标就是浏览器标签页最靠左那个图标

  <img src="/2022-03-19-gitpage-blog-setup/favicons.png" alt="favicons" style="zoom:50%;" />
  _favicons_

- 到 [**Real Favicon Generator**](https://realfavicongenerator.net/) 打开你选好的图片，滑倒最底下 `Generate your Favicons and HTML code` 生成 favicon

- 下载解压之后，删除 `browserconfig.xml`，`site.webmanifest` 两个文件，把其余的拖到项目目录 `assets/img/favicons/` 下，如果没有这个路径，就新建文件夹

- `commit & push & deploy` 之后就可以在浏览器看到网址的新图标了

## 本地配置

### Jekyll-Compose

- 部署 [**Jekyll-Compose**](https://github.com/jekyll/jekyll-compose) 能让你更方便的写博客，这个是自动添加 [Front Matter](https://jekyllrb.com/docs/front-matter/) 的，使你可以更专注于写文章

- 打开 项目文件夹，在  `Gemfile` 文件中添加 `gem 'jekyll-compose', group: [:jekyll_plugins]`，然后终端执行

  ```console
  $ bundle
  ```

- 命令有 

  ```
  draft      # Creates a new draft post with the given NAME
  post       # Creates a new post with the given NAME
  publish    # Moves a draft into the _posts directory and sets the date
  unpublish  # Moves a post back into the _drafts directory
  page       # Creates a new page with the given NAME
  rename     # Moves a draft to a given NAME and sets the title
  compose    # Creates a new file with the given NAME
  ```
  
- 具体用法可参照 [**Jekyll-Compose**](https://github.com/jekyll/jekyll-compose)
  
### 写文章

#### 快捷键

- **jekyll-compose** 可以在 `_config.yml` 中添加 `auto_open: true`，即可使用 `bundle exec jekyll post newPostName` 直接打开 `newPostName.md`，但在 mac 上用不了，查看源码

  <img src="/2022-03-19-gitpage-blog-setup/jekyll-compose.png" alt="jekyll-compose" style="zoom:50%;" />
  _jekyll-compose source code_

  因为 mac 中直接 输入 `typora` 是调不出程序的... 这里 editor_name 是调用 环境变量的，所以：

  ```console
  $ export JEKYLL_EDITOR='open -a typora'
  ```

- 上一步完成后还是觉得麻烦，必须在项目目录下打开终端才可以使用；因此

  ```console
  $ alias gitpage='cd [repo directory] && bundle exec jekyll'
  ```

  可以直接跳转到项目目录执行 `jekyll-compose` 命令，打开编辑器立刻开写！
  
- 至此，可以从命令行直接开写了，但我还有 [Alfred](https://www.alfredapp.com/)，于是...

  <img src="/2022-03-19-gitpage-blog-setup/alfred.png" alt="alfred" style="zoom:50%;" />
  _alfred_

  一开始想集成 jekyll 所有功能，做到后面才发现其他操作不需要在 alfred 快捷完成，反而会影响最重要的创建的快捷性，因此只做了单一功能。

  简单的实现就是 Keywords + Run Script，以下是脚本代码

  ```bash
  #!/bin/bash
  
  cd $project_path
  bundle exec $jekyll_path post $1
  filename=`ls -lt _posts | grep md | head -n 1 | awk '{print $9}'`
  
  # >&2 echo $filename
  open -a typora _posts/$filename
  ```

  

#### 图片插入

- 这里我是将图片直接放在 Github repo/_images/ 下，没有额外的做 cdn；

- 可以在 `_config.yml` 的 `img_cdn` 中设置文件夹路径，把路径中的 `tree` 改成 `raw` 就能用了，这样设置之后所有 post 的文件都是相对路径了，方便管理，我这边设置是每个 post 一个文件夹；

- 在本地编辑的话我选用的是 [Typora](https://typora.io/) 编辑器，很容易上手... 可以设置插入图片时复制到文件夹，把复制文件夹设为 repo/_images，然后再在 post 的 md 文件头 设置一行 `typora-root-url: "../_images"` 表示根路径，这样文档内就可以使用同 github 上的相对路径；

- 有一个小缺点就是 typora 显示的图片和线上效果不一样，这点后续再看看有没有改进方法。

  

## 总结

本以为搭博客会很快，结果踩了不少坑，但趁机接触了不少新东西，回顾了很多旧玩意儿。各种脚本比想象中有趣很多，有空了可以再继续深究... git workflow, alfred workflow ( iOS Remote ) 等等。

### TODO: 

- [ ] Alfred Workflows publish/update 脚本，用于快速 commit & push 指定 post
- [ ] 图片清理脚本，用于清理图片文件夹内没被用到的图片
- [ ] 解决 Typora 图片显示与网页不一致的问题
- [ ] 流量监控完善，这个不着急

  

  

  
