---
title: Graph Representation in Malware Detection
author: Zhen Zhang
date: 2019-05-29
---

We are working on a new malware detection tool which uses a
novel graph representation as the program semantic abstraction,
in order to enable better performance when the labelled graphs
are used with a machine learning to, for example, classifying malware categories.

This post is not a paper, but as a case analysis to show the current capability of our tool
and foster discussion.
We intentionally hide some details as our work is still
under submission. However, welcome to contact me (zgzhen@cs.washington.edu) for details.

Below, we will show a subset of representative outputs from our tool when running end-to-end.
We will also briefly explain the graph and why it matches with the ground-truth label.

## Benign sample

![](images/87df2a8292e1ab1b3ce59fb74c8f7f48445642e3249dec1005cd1b663d3c41ca.apk.png)


Benign sample is a non-malicious sample. We have a set of benign samples that is very easy to mistake as malicious ones.
In this sample, the only suspicious behaviour is a dependency from background activity to SMS related operation.
However, this is not a strong pattern, thus being classifier as "benign".

APK: examples/benign-1.apk
(Please contact author if you need access to these APKs)

## Trojan

![](images/trojan-1.png)


*Trojan* is an application that appears to be benign and performs undesirable actions against the user.
In this sample, we captured the relationship between "Run in background" and "Execute system command/Root privilege escalation"

APK: examples/trojan-1.apk


## Backdoor

![](images/c80f4ed05c68a51c4c4524b2b94485ebd82d9b9b960976f086c2ae8877082477.apk.png)

Backdoor is an application that allows the execution of unwanted, potentially harmful, remote-controlled operations on a device.
This sample shows the use of API `java.lang.Runtime.exec` to run shell scripts in the background.

## Hostile downloader

![](images/d8ec5a42bd66669aba9fdc6ecf79eb1526c8ad19876ca3571c4976d4109be5d5.apk.png)


![](images/cf03447c98f9c316a8dab4a212b51d53c455800034897885eb3176b203e5182a.apk.png)


*Hostile downloader* will download other potentially harmful applications without consensus from user.
In the first abstraction, an internet/http connection is used in a separate thread, rather than being directly invoked by user activity.
In the second one, the app is apparently much noiser but we are still capable to capture the core dependency as an edge from
"Thread run" to "Internet/HTTP".

## Phising

![](images/a59569d59dae24d223349a5ea1055ae35e427a1e10af492e801f189a290fdd94.apk.png)


*Phishing* app pretends to come from a trustworthy source, request a users authentication credentials and/or billing information,and send the data to a third party.

In our abstraction, an edge from big integer to addFlags/setFlags/setContentView shows that the app tries to overlay the user interface with a large
cover dynamically, which potentially indicates the malicious intention of faking UI.

## SMS Fraud

![](images/5c953957985d3e7fa2b8bb9ec52e018b6ca234065a74413daefab7ad0498d791.apk.png)


*SMS fraud* will charge users to send premium SMS without consent, or tries to disguise its SMS activities by hiding disclosure agreements.

The core behaviour will be sending SMS messages in background ("background -> smsops").

## Spyware

*Spyware* transmits sensitive information off the device.

![](images/9569c774a21be8c5071a8ea7238d43a4ec0ce5dc93c0a0e97c1f250c37e54930.apk.png)

![](images/f44740bbf47f69e1e5abd51ce73e1b1fdfa9861fae0d6c74f300ff770c7f3a5b.apk.png)


In these two abstractions, you can observe the use of information source and information sink. Due to the limitation of static analysis,
it is difficult to find a data-flow dependency directly between source and sink. But our abstraction can show their co-existence, which
is also useful.
