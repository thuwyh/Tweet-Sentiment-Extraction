# Tweet Sentiment Extraction

This repo is my solution of the Kaggle Tweet Sentiment Extraction competition. Our team, **Where is the magic :(**, is finally ranked 7th in the competition. 

Great writeup by my teammate [Xuan Cao](https://www.kaggle.com/naivelamb): https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159245

Leaderboard page: https://www.kaggle.com/c/tweet-sentiment-extraction/leaderboard

### Results
I have two RoBERTa-based models. The only difference between them is whether the original sentiment is included in the input.

- 10-fold model without original sentiment: cv 0.725153, lb 0.72034, pb 0.72850
- 10-fold model with original sentiment: cv 0.726468

The original sentiments are from the original dataset [[data]](https://www.kaggle.com/maxjon/complete-tweet-sentiment-extraction-data)[[discussion]](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/145363).

### Highlights
- model design
    - CNN layer to enhance feature extraction for span prediction
    - auxiliary head to classify whether the `selected_text` is the whole sentence
    - auxiliary head to classify whether a token belongs to `selected_text`
    - concat last 3 layers of RoBERTa output
- training method
    - FGM adversarial training
- pre&post-processing
    - This is the **MAGIC** we found. 
    - Preprocessing: get the correct label by shifting span window according to the extra spaces
    - Postprocessing: shift back

### Run the code
train & validation:
```bash
cd sh
./with_original_sentiment.sh # or ./without_original_sentiment.sh
```
inference kernel: https://www.kaggle.com/wuyhbb/roberta-inference-ensemble-v10 