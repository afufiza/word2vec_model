Install virtualenv
```
sudo pip install virtualenv 
```

Create and activate virtual environment
```
virtualenv venv -p python3.5
source venv/bin/activate
```

Train word2vec and generate files sampleVectors.json and word_vectors.png
```
python run.py
```

Assignment submission(for students)
```
# zip the assignment submission folder
cd assignment2
sh collect_submission.sh
cd ..
```
