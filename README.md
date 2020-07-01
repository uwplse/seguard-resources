# SeGuard public resources

## What is it?

SeGuard is a static analyzer framework for building semantic graphs of android malware.

## Quick start

Download `seguard-java.tar.gz` and `seguard-python.tar.gz` to current folder from https://github.com/uwplse/seguard-resources/releases. Install dependencies [specified here](https://izgzhen.github.io/seguard-www/quickstart.html) or use the following quick instructions:

For ubuntu (assuming using `bash`):

```
$ wget https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u222-b10/OpenJDK8U-jdk_x64_linux_hotspot_8u222b10.tar.gz
$ tar xvf OpenJDK8U-jdk_x64_linux_hotspot_8u222b10.tar.gz
$ echo "export PATH=$PWD/jdk8u222-b10/bin:$PATH" >> ~/.bashrc
$ source ~/.bashrc
$ sudo apt-get install gcc python-virtualenv python3-dev maven graphviz libgraphviz-dev unzip
```

Then,

```
# project root
mkdir seguard-framework

# install java artifact
mv seguard-java.tar.gz seguard-framework
cd seguard-framework
tar xvf seguard-java.tar.gz
cd ..

# install python artifact
tar xvf seguard-python.tar.gz
mv seguard-0.1dev seguard-framework/tools/python`
cd seguard-framework
virtualenv -p python3 .venv
source .venv/bin/activate
pip install -e tools/python

# try it
tools/seguard-cli examples/drebin0-5fd871.apk
```

Check out https://izgzhen.github.io/seguard-www/troubleshooting.html if you got into any problems.

## Onboarding for New Contributor

[ONBOARDING.md](ONBOARDING.md)

## Wiki

https://github.com/izgzhen/seguard-resources/wiki

(Edit as will)

## Blog Posts and Write-ups

- 2020-04-04: [Soot-skeleton: Bootstrapping a Static Analyzer by Examples](files/Soot-skeleton_Xiaxuan.pdf) by Xiaxuan Gao, Zhen Zhang
- 2020-01-15: [Embedding Semantic Graph using Node Distance for Malware Detection](files/embeddings.pdf) by Luxi Wang, Zhen Zhang
- 2019-05-29: [Graph Representation in Malware Detection](https://github.com/izgzhen/seguard-resources/blob/master/posts/case-study-01.md) by Zhen Zhang

## Video Tutorials

- [SeGuard error anlaysis](https://vimeo.com/350633606), [transcript](files/transcript/error-analysis-01.txt)
- [Issue #66 Debug Demo](https://youtu.be/M6AUuDf7Qwg): [link to issue](https://github.com/izgzhen/seguard-framework/issues/66)

## Live demo (WIP)

http://bit.ly/SeGuardDemo

## Team

- Lead by [Zhen Zhang](https://homes.cs.washington.edu/~zgzhen/)
- Contributors: [Luxi Wang](https://github.com/LuxiWang99), [Xiaxuan Gao](https://github.com/MarkGaox), [Andrew Shi](https://github.com/andrewshi98), [Jack Zhang](https://github.com/JackZhangUW), [Lucy (Yanglu) Shu](https://github.com/yanglushu)
- Advisors: [Yu Feng](https://cs.ucsb.edu/people/faculty/feng), [Mike Ernst](https://homes.cs.washington.edu/~mernst/), [Isil Dillig](https://www.cs.utexas.edu/~isil/)
- Industry Collaborators: Sebastian (Google), Alec (Google)
