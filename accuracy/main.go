package main

import (
	"flag"
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/tweeters"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var modelPath string
	var dbPath string
	var validation float64
	var prob float64
	var batchSize int
	var minTweets, maxTweets int
	flag.StringVar(&modelPath, "model", "../train/model_out", "path to trained model")
	flag.StringVar(&dbPath, "data", "", "path to tweet DB")
	flag.Float64Var(&validation, "validation", 0.1, "validation fraction used to train")
	flag.Float64Var(&prob, "prob", 0.5, "probability of same user")
	flag.IntVar(&batchSize, "batch", 64, "batch size")
	flag.IntVar(&minTweets, "min", 3, "minimum tweets per user")
	flag.IntVar(&maxTweets, "max", 16, "maximum tweets per user")
	flag.Parse()

	if dbPath == "" {
		essentials.Die("Required flag: -data. See -help.")
	}

	log.Println("Loading model...")
	var model *tweeters.Model
	if err := serializer.LoadAny(modelPath, &model); err != nil {
		essentials.Die(err)
	}
	model.SetDropout(false)

	log.Println("Loading DB...")
	db, err := tweeters.OpenDB(dbPath)
	if err != nil {
		essentials.Die(err)
	}
	samples := tweeters.NewSamples(db)
	_, testing := samples.Partition(validation)
	log.Printf("%d testing users", len(testing.UserIndices))

	c := anyvec32.CurrentCreator()

	log.Println("Computing accuracy...")
	var numCorrect float64
	var numTotal float64
	for {
		tweets, avg, labelFloat, err := testing.Batch(prob, batchSize, minTweets, maxTweets)
		if err != nil {
			essentials.Die(err)
		}
		labels := c.MakeVectorData(c.MakeNumericList(labelFloat))
		latent := model.Averages(tweets, avg)
		out := model.Classifier.Apply(latent, len(avg)/2).Output()
		anyvec.GreaterThan(out, float32(0))

		numCorrect += float64(out.Dot(labels).(float32))
		anyvec.Complement(out)
		anyvec.Complement(labels)
		numCorrect += float64(out.Dot(labels).(float32))

		numTotal += float64(labels.Len())
		log.Printf("Got %.2f%% (out of %d)", 100*numCorrect/numTotal, int(numTotal))
	}
}
