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
