# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

# Presentation

The presentation is uploaded as a file on the github. It goes over the main themes

# Overview

I collected data from various servers, parsed the events with time and ran LDA on the messages to get topic clusters.

# Implementations

## collect_events.py

Do not run this file
The collect events file has been run on a database, and due to the amount of time taken to run, I have attached the output as a json file.

## messages_w_time

This file stores data of the messages that contain time extracted from discord messages

## Jupyter notebook part

<ol>
  <li>Data preprocessing by getting rid of HTML tags, stemming, converting to lower case, stemming, getting rid of stopwords etc.</li>
  <li>Data visualization with seaborn/matplotlib etc.</li>
  <li>Applied a Bert-like model with Pytorch</li>
  <li>More details in the Jupyter Notebook</li>
</ol>

## Webapp

<ol>
  <li>Data preprocessing by getting rid of HTML tags, stemming, converting to lower case, stemming, getting rid of stopwords etc.</li>
  <li>Extract words and bi-grams from sentence</li>
  <li>Used tfidf to build features for model training</li>
  <li>Applied a logistic regression to serve the model, for demonstration purposes. Ran into a whole series of problems when deploying the app to the web due to the environment/dependencies/servers not having the most updated package available. Hang using a Bert-related model in the Jupyter Notebook, which is a way more fancier model than the one in the App. </li>
</ol>

# Contributions

<ul>
  <li>Anqi & Majharul: Data preprocessing, generating graphs</li>
  <li>Hang: Train a Bert-based model with Pytorch in Jupyter Notebook</li>
  <li>Tianhao: Webapp developed in Dash with packages such as scikit-learn, deployed to render.com</li>
 </ul>

# Jupter Notebook instruction

Assume you're already in the CourseProject folder, having done the following

CD into a directory that you want to work on

On terminal

```
git clone https://github.com/Chirag5241/CollegeEvents.git
```

Go to the CollegeEvents folder

```
cd CollegeEvents
```

Then, if you have Anaconda installed. You can simply do this in terminal

```
jupyter notebook
```

This opens the notebook on your browser

## LDA_on_events

Please run the LDA on events file.
This file has all the visualizations for topic clustering

# Discussion

## Model performance

Both precision, recall and F1 score are above 0.90 for the deep learning model. For details see Jupyter Notebook.

## Limitations of work

We just trained on the title itself, did not utilize the article link. Sometimes whether a title is sarcastic or not will be based on context (such as the article content). This could be a future work item. With that being said, we're satisfied with our current model performance.
