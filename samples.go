package tweetures

import (
	"bytes"
	"crypto/md5"
	"encoding/csv"
	"errors"
	"io"
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

// ReadSamples loads tweets from a CSV file.
// The file should be formatted like the outputs of
// tweetdump: https://github.com/unixpickle/tweetdump.
func ReadSamples(csvPath string) (samples *Samples, err error) {
	defer essentials.AddCtxTo("read samples", &err)

	f, err := os.Open(csvPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)

	// Reduce memory consumption by counting tweets/user
	// before recording any tweet bodies in memory.

	counts := map[string]int{}
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		if len(row) < 3 {
			return nil, errors.New("expected at least 3 columns")
		}
		counts[row[1]]++
	}

	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	reader = csv.NewReader(f)

	// Read every tweet body into a single buffer, then slice
	// it up into individual tweets.
	// This avoids some memory fragmentation, although not a
	// lot as far as I can tell.

	samples = &Samples{Tweets: map[string][][]byte{}}
	indices := map[string][]int{}
	buffer := bytes.Buffer{}
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		} else if counts[row[1]] < 2 {
			continue
		}
		user := row[1]
		if _, ok := indices[user]; !ok {
			samples.Users = append(samples.Users, user)
		}
		msg := []byte(row[len(row)-1])
		indices[user] = append(indices[user], buffer.Len(), buffer.Len()+len(msg))
		buffer.Write(msg)
	}

	fullBytes := buffer.Bytes()
	for user, is := range indices {
		var strs [][]byte
		for i := 0; i < len(is); i += 2 {
			strs = append(strs, fullBytes[is[i]:is[i+1]])
		}
		samples.Tweets[user] = strs
	}

	return
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
