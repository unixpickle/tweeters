# Extracting features from tweets

Twitter can serve as a rich catalogue of human interests. I want a Machine Learning model to grasp, at least on a basic level, what defines these interests. In the past, I tried making an [auto-encoder](https://github.com/unixpickle/tweetenc) for this purpose. While the auto-encoder worked, its learned latent variables were not particularly useful for things like sentiment analysis. I think neural nets can do better.

# The plan

Instead of auto-encoding tweets, I will train a network using a supervised learning task that forces some amount of latent structure. In a sense, the supervised learning task will be a meta-learning algorithm. Here's how it will work:

 * Extract *n > 1* tweets from one user, *A*.
 * Show an **encoder** *n-1* of those tweets, getting *n-1* vectors.
 * Average the encoded tweet vectors to get a vector *x*.
 * Select a final tweet:
   * With probability p, select the remaining tweet from user *A*.
   * With probability 1-p, select a random tweet.
 * Show the final tweet to the encoder, getting a vector *y*.
 * Show *x* and *y* to a **classifier**, which predicts if *y* was from user *A*.

# Goal

Hopefully, the network will learn that different users write about different things. The latent space will represent common topics in a linear way in order to deal with the average during training.

# Hypothesis

The model will probably perform better than random guessing. I'm not too worried about overfitting, since I have so much [tweet data](https://github.com/unixpickle/tweetdump).

I doubt that the latent features will be easy to interpret. However, I am optimistic that they will be useful for tasks like sentiment analysis. In a [separate project](https://github.com/unixpickle/rwa/tree/master/experiments/sentiment), I got a character-level language model to perform sentiment analysis. This makes me optimistic about the abilities of a character-level encoder.

# Self-test

I evaluated myself on the classification task to see how well I could do. For users with >1 tweet, I got 58/100 (58%). For many of the users, I only saw one tweet and then had to guess about a second tweet. This task seemed difficult, so I modified the code to filter out users with only two tweets. For users with >2 tweets, I got 36/50 (72%).

Now I am less confident that the model will learn about the contents of tweets. I often made decisions based solely on style (capitalization, usage of emojis, etc.). I suspect the model will focus more on style than content. However, when tweets were sports-related or relationship-related, I was generally able to make content-based predictions.

# Results

**TL;DR:** the model achieved 74.5% accuracy (tested on 7230 validation samples). This is close to my self-test results.

Initially, the model learned nothing. It never got better than a cross-entropy of *ln(1/2)*, which is random guessing. I looked at activations throughout the stabilized network, and was discouraged to find that the encoder was producing unvaried (collapsed) outputs.

To address the lack of learning, I resorted to pre-training. Ironically, I used an already-trained model from [tweetenc](https://github.com/unixpickle/tweetenc). See the [repurpose](repurpose) command for details on how I converted the tweetenc model.

After repurposing tweetenc, I also found that I had to use a lower learning rate. With a learning rate of 0.001, it seemed that the learned weights from tweetenc initially helped (got cross-entropy down by around 0.02), but eventually the model collapsed back to random guessing. I succeeded with a learning rate of 0.0001. I also used a larger batch size of 128. Eventually I increased the batch size to 256, which did not hurt performance at all on my Titan X, but it didn't seem to help very much either.
