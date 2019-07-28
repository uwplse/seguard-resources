# SeGuard public resources

## Released artifacts

Download `seguard-java.tar.gz` and `seguard-python.tar.gz` to current folder from https://github.com/uwplse/seguard-resources/releases.

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

## Onboarding

[ONBOARDING.md](ONBOARDING.md)

## Wiki

https://github.com/izgzhen/seguard-resources/wiki

(Edit as will)

## Blog Posts

- 2019-05-29: [Graph Representation in Malware Detection](https://github.com/izgzhen/seguard-resources/blob/master/posts/case-study-01.md)

## Video Tutorials

- [SeGuard error anlaysis](https://vimeo.com/350633606), [transcript](files/transcript/error-analysis-01.txt)

## Surveys

https://www.surveymonkey.com/r/X5HHJBQ
