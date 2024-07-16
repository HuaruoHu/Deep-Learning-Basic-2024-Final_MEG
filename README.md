Deep-Learning-Basic-2024
final project MEG brainwave classification
1. Modified the src/models.py (re-wrote the model structure) 
2. The actual classification for testing was executed with main.ipynb instead of main.py for debugging (when testing, the model classes and hyper-parameters were directly written in the .ipynb file, though the config file and models.py were also modified eventually, I didn’t actually test a lot with the main.py) 
3. In the .ipynb file, sometimes the code needs to execute the distributed base model (renamed as BasicConvClassifier2) first before executing the customized model (BasicConvClassifier and BasicConvClassifier1, which are exactly same models, but one referenced from model.py and one was directly written in the .ipynb file for debug). Sorry I didn’t debug this issue also due to time limitations. 
