package tweetures

import (
	"crypto/md5"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
)

// Samples stores a list of users and tweets by those
// users.
type Samples struct {
	Users  []string
	Tweets map[string][][]byte
}

// ReadSamples loads tweets from a database file, which
// can be created by running a CSV file through the
// build_db tool.
func ReadSamples(dbPath string) (samples *Samples, err error) {
	defer essentials.AddCtxTo("read samples", &err)

	f, err := os.Open(dbPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	records, errChan := ReadDB(f)
	lastUser := ""
	res := &Samples{Tweets: map[string][][]byte{}}
	for record := range records {
		userStr := string(record.User)
		if userStr != lastUser {
			lastUser = userStr
			res.Users = append(res.Users, lastUser)
		}
		res.Tweets[userStr] = append(res.Tweets[userStr], record.Body)
	}
	if err := <-errChan; err != nil {
		return nil, err
	}
	return res, nil
}

// Partition splits the data up by username in a
// pseudo-random (but deterministic) way.
func (s *Samples) Partition(testingFrac float64) (training, testing *Samples) {
	var trainingUsers, testingUsers []string
	for _, user := range s.Users {
		hash := md5.Sum([]byte(user))
		if float64(hash[0]) < testingFrac*0x100 {
			testingUsers = append(testingUsers, user)
		} else {
			trainingUsers = append(trainingUsers, user)
		}
	}
	return &Samples{
			Users:  trainingUsers,
			Tweets: s.Tweets,
		}, &Samples{
			Users:  testingUsers,
			Tweets: s.Tweets,
		}
}

// Batch produces a training or validation batch.
//
// The p argument is the probability of a positive
// classification.
//
// The batchSize argument specifies the maximum number of
// tweets to feed the model.
// The batchSize is a soft-limit, not an absolute
// requirement.
//
// The batch has three components: a list of tweets, a
// list of average sizes, and a vector of desired
// classifier outputs.
// The tweets and average sizes are meant to be passed to
// Model.Averages, the output of which is then meant to be
// fed into the classifier.
func (s *Samples) Batch(p float64, batchSize int) (tweets [][]byte, avg []int,
	outs []float64) {
	for len(tweets) < batchSize {
		t := s.randomUserTweets()
		if rand.Float64() < p {
			outs = append(outs, 1)
		} else {
			t[len(t)-1] = s.randomTweet()
			outs = append(outs, 0)
		}
		tweets = append(tweets, t...)
		avg = append(avg, len(t)-1, 1)
	}
	return
}

func (s *Samples) randomUserTweets() [][]byte {
	tweets := s.Tweets[s.Users[rand.Intn(len(s.Users))]]
	randIdx := rand.Perm(len(tweets))[:2+rand.Intn(len(tweets)-2)]
	res := make([][]byte, len(randIdx))
	for i, j := range randIdx {
		res[i] = tweets[j]
	}
	return res
}

func (s *Samples) randomTweet() []byte {
	tw := s.Tweets[s.Users[rand.Intn(len(s.Users))]]
	return tw[rand.Intn(len(tw))]
}
