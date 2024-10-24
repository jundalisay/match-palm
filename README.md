## Team Superphysics: Match for KMC Hackathon

![match.jpg](match.jpg)

- Juan Dalisay Jr.
- Michael Lituanias
- Jose Felizco

### Presentation: https://www.canva.com/design/DAGUcrWEWrs/XULJvvXDj0NDhku6-fWr1w/view?utm_content=DAGUcrWEWrs&utm_campaign=designshare&utm_medium=link&utm_source=editor

This has 3 parts:

#### 1. Xamun UI at 
https://app.xamun.ai/walk-through-preview/663/1566


#### 2. Document Comparison (keyword.py)

Google Colab: https://colab.research.google.com/drive/1xq5qCDR391kVy9d1xofs0YQzPQVuYNw4?usp=sharing


Javascript Developer Duties: https://fullscale.io/blog/roles-and-responsibilities-of-a-javascript-developer/

Applicant 1: https://medium.com/@jojo_38618/a-personal-essay-on-my-journey-into-tech-518fcb8760d6

![applicant1.png](applicant1.png)


Applicant 2: https://zelig880.com/my-10-years-experience-as-a-javascript-software-engineer

![applicant2.png](applicant2.png)

This shows that applicant 2 is way better. 

This is even more true when the full essays are loaded at the Streamlit app (streamlit.py) which is live at https://apppy-barzqkyiobotiw23u5dmii.streamlit.app/

Applicant 1: 

![applicant1full.png](applicant1full.png)

Applicant 2:

![applicant2full.png](applicant2full.png)



#### 3. Keras image recognition (match.py)

Google Colab: https://colab.research.google.com/drive/1X6jxnzczZOrNn3o4VhrCncOQIKC0UbMv?usp=sharing

1. Import hands dataset from Kaggle and get only right hand palms  https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset/code?resource=download

2. Create Python Script on Google Colab

- import modules
- data
  - upload data to collab  
  - assign to var
- dataset
  - create dataset params
  - split dataset to training 80% (train_ds) and validation 20% (val_ds)
  - find or set class names
  - pass the dataset to the Keras Model.fit method for training 
  - config dataset for performance
  - standardize the data in the dataset by normalizing
- model
  - create and compile the model
  - get summary
- train
- visualize results
- test on samples
  - criminal_palm.jpg
  - civilian_palm.jpg


3. Do the same on Streamlit (image.py)

Applicant's palm compared to a set of known palms:

![palms.jpg](palms.jpg)


> start time: 10am Oct 24, 2024

> end time: 8am Oct 25, 2022



### Problem/s Encountered

- Limited samples causes overfitting
- Palm images too varied
- Lines on palm are not clear
- Xamun.ai is very difficult to use


### Workaround

- Data augmentation -- generate additional training data from existing examples
- Manually select palm images that are proper
