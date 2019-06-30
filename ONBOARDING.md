**Welcome to our project!**

For a new member, we recommend you to get started with our project following the steps below
so as to **start contributing and learning right away**.

**Pre-requisites**: To effectively collaborate with us, or anyone doing serious
engineering, you need to learn to use shell interface and git.

- Shell:
    + Our preferred working environment is macOS/Ubuntu, which should save you
      a lot of time and headache in the long run.
      But you can try to use other Linux or Windows.
    + If you are using Windows natively, you are recommended to use
      [Git Bash](https://gitforwindows.org/), which is installed along with Git
      for Windows.
    + Otherwise, I assume you have native access to bash or bash like shell.
      [CSE390 Bash Reference](https://courses.cs.washington.edu/courses/cse391/17sp/bash.html)
    + **Test yourself**: What does `echo $(date) > date.txt` mean?
- Git:
    + [Git Handbook by GitHub](https://guides.github.com/introduction/git-handbook/)
    + **Test yourself**: What is `git rebase` and why is it useful?

Depending on your interest, there are two tracks for you:

Track I: **Analyzer framework**

1. Send me (zhen) an email including your Github username, ask for access to [seguard-framework repo](https://github.com/izgzhen/seguard-framework). Follow the instructions in https://izgzhen.github.io/seguard-www/quickstart.html.
Your goal is to successful generate the graph visualization.
2. Find an issue with "good-first-issue" and assign yourself. Some of the descriptions might not be clear,
   be sure to ask for clarification and we are here to help!

Track II: **Machine learning**

1. Start with the Python notebook `classifier.ipynb` that loads in `DX`
   (graph features vector), `DY` (multi-class labels), and `DZ` (binary-class
   labels). Play around it and try to use Random Forest or whatever model
   you like to predict the result using 10-fold cross-validation method.
   Note down the precision.
2. Now, let dig deeper to see where all these features come from.
   In folder `data/graph`, there are folders who names are labels.
   In each sub-folder, there are a lot of `*.dot` files. Each dot file is a
   graphviz format of abstract graph that represents some program behavior.
   The naive way to transform a graph into feature vector is one-hot encoding
   of names of edges and nodes. However, there might be a better way to do it.
   Can you first try to implement the naive approach, and then come up with
   a better way to do it? (e.g. encoding connectivity, or use a feature learning
   tool like node2vec, or use graph neural networks! Search "Graph Embedding" on
   Google for related information. Welcome to discuss with me)
   Can you compare these different featurization methods
   by using the label as classification groundtruth and calculate recall etc.
   metrics?

## Learning

We are trying to maintain a library to needed knowledge and references
in this repo: https://github.com/izgzhen/seguard-resources.
They might not be complete, so ask by
[creating an issue here](https://github.com/izgzhen/seguard-resources/issues) would be helpful!

## Collaboration

* Real-time discussion: Ask me (zhen) to join our slack channel
* **Your contribution will always be useful to others and appreciated**.
Whatever it is fixing a typo, adding a utility class, or improving the document like this one,
feel free to open a pull request.

## Troubleshooting

https://izgzhen.github.io/seguard-www/troubleshooting.html lists problem
you might encounter when playing with the analyzer framework.

For problem regarding project code access and data access, please don't
hesitate to contact me (zgzhen cs washington edu). Sometimes I forgot
things to set up or reply in time, just remind me through email!
