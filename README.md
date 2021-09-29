# Named-Entity-Recognition
Natural Language Processing (NLP) problem dealing with information extraction


Objective: To locate and classify named entities in text into predefined categories such as the names of persons, organizations, locations, events, expressions of times, etc. 

![image](https://user-images.githubusercontent.com/79351706/135356422-5300d8be-5010-4afd-a241-50a384136cc9.png)

**Dataset:** The Groningen Meaning Bank (GMB) corpus which is tagged, annotated and built specifically to train the classifier to predict named entities such as name, location, etc. The dataset consists of four columns: Sentence #, Word, POS (Parts of Speech) and Tag. The tags cover 8 types of named entities: 
![image](https://user-images.githubusercontent.com/79351706/135356564-067851e4-f3b1-469a-88b1-0f3b36dda1cf.png)

Entity tags are encoded using a BIO annotation scheme, where each entity label is prefixed with either B or I letter. A “B-” tag indicates the first term of a new entity (or only term of a single-term entity), while subsequent terms in an entity have an “I-” tag. For example, “New York” is tagged as ["B-geo", "I-geo"], “London” is tagged as "B-geo", “World War II” is tagged as [“B-eve”, “I-eve”, “I-eve”], etc. 

All other words, which don’t refer to entities of interest, are labelled with the O- tag (e.g., of, an, the, etc.). 
A total of 17 entity tags as:
['B-geo', 'I-geo', 'B-org', 'I-org', 'B-per', 'I-per’, 'B-gpe', 'I-gpe', 'B-tim', 'I-tim’, 'B-art', 'I-art', 'B-eve', 'I-eve’, 'B-nat', 'I-nat', 'O'] 


**Results:**

![image](https://user-images.githubusercontent.com/79351706/135356729-20956609-8534-467c-b803-6bd93dc02387.png)

Confusion Matrix:

![image](https://user-images.githubusercontent.com/79351706/135356762-29886892-5c2f-47bd-86e8-58601390dbcc.png)





