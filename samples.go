package tweeters

import (
	"math/rand"

	"github.com/unixpickle/essentials"
)

// Samples provides high-level access to the tweets in a
// DB.
type Samples struct {
	DB *DB

	// A list of the users to sample from.
	//
	// This might not include all users in the case of a
	// partitioned sample list.
	UserIndices []int
}

// NewSamples creates a Samples with all of the user
// indices in a DB.
func NewSamples(db *DB) *Samples {
	res := &Samples{DB: db, UserIndices: make([]int, db.NumUsers())}
	for i := 0; i < db.NumUsers(); i++ {
		res.UserIndices[i] = i
	}
	return res
}

// Partition splits the samples up by user in a
// pseudo-random (but deterministic) way.
func (s *Samples) Partition(testingFrac float64) (training, testing *Samples) {
	src := rand.NewSource(1337)
	users := rand.New(src).Perm(s.DB.NumUsers())
	testingCount := int(float64(len(users)) * testingFrac)
	return &Samples{DB: s.DB, UserIndices: users[testingCount:]},
		&Samples{DB: s.DB, UserIndices: users[:testingCount]}
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
// The min and max arguments specify the minimum and
// maximum number of tweets to select for one user.
// The min argument must be at least 2.
//
// The batch has three components: a list of tweets, a
// list of average sizes, and a vector of desired
// classifier outputs.
// The tweets and average sizes are meant to be passed to
// Model.Averages, the output of which is then meant to be
// fed into the classifier.
func (s *Samples) Batch(p float64, batchSize, min, max int) (tweets [][]byte, avg []int,
	outs []float64, err error) {
	if min < 2 {
		panic("invalid min argument")
	}
	for len(tweets) < batchSize {
		t, err := s.RandomUserTweets(min, max)
		if err != nil {
			return nil, nil, nil, err
		}
		if rand.Float64() < p {
			outs = append(outs, 1)
		} else {
			newTs, err := s.RandomUserTweets(1, 1)
			if err != nil {
				return nil, nil, nil, err
			}
			t[len(t)-1] = newTs[0]
			outs = append(outs, 0)
		}
		tweets = append(tweets, t...)
		avg = append(avg, len(t)-1, 1)
	}
	return
}

// RandomUserTweets randomly selects a subset of a random
// user's tweets.
//
// The min and max arguments limit the number of tweets to
// the range [min, max].
func (s *Samples) RandomUserTweets(min, max int) ([][]byte, error) {
	for {
		userIdx := s.UserIndices[rand.Intn(len(s.UserIndices))]
		records, err := s.DB.Read(userIdx)
		if err != nil {
			return nil, err
		}
		if len(records) < min {
			continue
		}
		clippedMax := essentials.MinInt(max, len(records))
		numTake := min + rand.Intn(clippedMax-(min-1))
		randIdx := rand.Perm(len(records))[:numTake]
		res := make([][]byte, len(randIdx))
		for i, j := range randIdx {
			res[i] = records[j].Body
		}
		return res, nil
	}
}
