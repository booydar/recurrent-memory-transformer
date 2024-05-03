Towards AI Complete Question Answering: A Set of Prerequisite Toy Tasks
-----------------------------------------------------------------------
In this directory is the first set of 20 tasks for testing text understanding and reasoning in the bAbI project.
The aim is that each task tests a unique aspect of text and reasoning, and hence test different capabilities of learning models. More tasks are planned in the future to capture more aspects.

For each task, there are 1000 questions for training, and 1000 for testing.
However, we emphasize that the goal is still to use as little data as possible to do well on the task (i.e. if you can use less than 1000 that's even better) -- and without resorting to engineering task-specific tricks that will not generalize to other tasks, as they may not be of much use subsequently. Note that the aim during evaluation is to use the _same_ learner across all tasks to evaluate its skills and capabilities.
Further while the MemNN results in the paper use full supervision (including of the supporting facts) results with weak supervision would also be ultimately preferable as this kind of data is easier to collect. Hence results of that form are very welcome.

For the reasons above there are currently several directories:

1) en/ -- the tasks in English, readable by humans.
2) hn/ -- the tasks in Hindi, readable by humans.
3) shuffled/ -- the same tasks with shuffled letters so they are not readable by humans, and for existing parsers and taggers cannot be used in a straight-forward fashion to leverage extra resources-- in this case the learner is more forced to rely on the given training data. This mimics a learner being first presented a language and having to learn from scratch.
4) en-10k/  shuffled-10k/ and hn-10k/  -- the same tasks in the three formats, but with 10,000 training examples, rather than 1000 training examples.
5) en-valid/ and en-valid-10k/ are the same as en/ and en10k/ except the train sets have been conveniently split into train and valid portions (90% and 10% split).

The file format for each task is as follows:
ID text
ID text
ID text
ID question[tab]answer[tab]supporting fact IDS.
...

The IDs for a given "story" start at 1 and increase.
When the IDs in a file reset back to 1 you can consider the following sentences as a new "story".
Supporting fact IDs only ever reference the sentences within a "story".

For Example:
1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary?        bathroom        1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel?      hallway 4
7 John moved to the office.
8 Sandra journeyed to the bathroom.
9 Where is Daniel?      hallway 4
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office  11
13 John went back to the garden.
14 John moved to the bedroom.
15 Where is Sandra?     bathroom        8
1 Sandra travelled to the office.
2 Sandra went to the bathroom.
3 Where is Sandra?      bathroom        2

Changes between versions.
=========================
V1.2 (this version) - Added Hindi versions of all the tasks. Fixed some problems with task 16, and added a separate set of directories for 10k training data, as we received requests for this.
V1.1 (this version) - Fixed some problems with task 3, and reduced the training set size available to 1000 as this matches the results in the paper cited above, in order to avoid confusion.
