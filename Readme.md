Install virtualenv
```
sudo pip install virtualenv 
```

Create and activate virtual environment
```
virtualenv venv -p python3.5
source venv/bin/activate
```

Install all requirements
```
pip install -r requirement.txt
```

Train word2vec and generate files *sampleVectors.json* and *word_vectors.png*

**Note: Do not change the hyperparameters values in run.py script**  
```
python run.py
```

Sanity check on sampleVectors.json
```
python test_sample_vectors.py
```

Assignment submission(for students)
```
# zip the assignment submission folder
cd assignment2
sh collect_submission.sh
cd ..
```
