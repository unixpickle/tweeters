package main

import (
	"flag"
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/tweeters"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var trainer Trainer
	var sgd anysgd.SGD
	var modelPath string
	var samplesPath string
	var stepSize float64
	var validation float64

	flag.StringVar(&modelPath, "out", "model_out", "path to model file")
	flag.StringVar(&samplesPath, "data", "", "path to tweet database")
	flag.IntVar(&sgd.BatchSize, "batch", 64, "batch size")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	flag.IntVar(&trainer.MinTweets, "min", 3, "minimum tweets per user")
	flag.IntVar(&trainer.MaxTweets, "max", 16, "maximum tweets per user")
	flag.Float64Var(&trainer.UserProb, "prob", 0.5, "probability of same user")
	flag.Float64Var(&validation, "validation", 0.1, "validation fraction")
	flag.Parse()

	if samplesPath == "" {
		essentials.Die("Required flag: -data. See -help.")
	}

	if err := serializer.LoadAny(modelPath, &trainer.Model); err == nil {
		log.Println("Loaded model.")
	} else {
		log.Println("Creating new model...")
		trainer.Model = tweeters.NewModel(anyvec32.CurrentCreator())
	}

	log.Println("Loading samples...")
	db, err := tweeters.OpenDB(samplesPath)
	if err != nil {
		essentials.Die(err)
	}
	samples := tweeters.NewSamples(db)
	training, testing := samples.Partition(validation)
	log.Printf("Samples: %d/%d training/testing users", len(training.UserIndices),
		len(testing.UserIndices))

	trainer.Samples = training

	sgd.Rater = anysgd.ConstRater(stepSize)
	sgd.Transformer = &anysgd.Adam{}
	sgd.Fetcher = &trainer
	sgd.Gradienter = &trainer
	sgd.Samples = anysgd.LengthSampleList(sgd.BatchSize)

	var iter int
	sgd.StatusFunc = func(b anysgd.Batch) {
		if iter%4 == 0 {
			validator := trainer
			validator.Samples = testing
			batch, err := validator.Fetch(sgd.Samples)
			if err != nil {
				essentials.Die(err)
			}
			cost := anyvec.Sum(validator.TotalCost(batch.(*Batch)).Output())
			log.Printf("iter %d: cost=%v validation=%v", iter, trainer.LastCost, cost)
		} else {
			log.Printf("iter %d: cost=%v", iter, trainer.LastCost)
		}
		iter++
	}

	log.Println("Training (ctrl+c to finish)...")
	sgd.Run(rip.NewRIP().Chan())

	log.Println("Saving model...")
	if err := serializer.SaveAny(modelPath, trainer.Model); err != nil {
		essentials.Die(err)
	}
}

// A Trainer fetches batches and computes gradients.
type Trainer struct {
	Model   *tweeters.Model
	Samples *tweeters.Samples

	MinTweets int
	MaxTweets int
	UserProb  float64

	// Set by Gradient().
	LastCost anyvec.Numeric
}

// Fetch produces a random batch of samples, using the
// length of s as the soft-limit on the batch size.
func (t *Trainer) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	tweets, avg, out, err := t.Samples.Batch(t.UserProb, s.Len(), t.MinTweets, t.MaxTweets)
	if err != nil {
		return nil, err
	}
	cr := t.Model.Parameters()[0].Vector.Creator()
	return &Batch{
		Tweets: tweets,
		Avg:    avg,
		Out:    anydiff.NewConst(cr.MakeVectorData(cr.MakeNumericList(out))),
	}, nil
}

// TotalCost computes the cost for a batch.
func (t *Trainer) TotalCost(b *Batch) anydiff.Res {
	latent := t.Model.Averages(b.Tweets, b.Avg)
	out := t.Model.Classifier.Apply(latent, len(b.Avg)/2)
	return anynet.SigmoidCE{Average: true}.Cost(b.Out, out, 1)
}

// Gradient computes the gradient for the batch.
func (t *Trainer) Gradient(batch anysgd.Batch) anydiff.Grad {
	grad := anydiff.NewGrad(t.Model.Parameters()...)

	cost := t.TotalCost(batch.(*Batch))
	t.LastCost = anyvec.Sum(cost.Output())

	c := cost.Output().Creator()
	one := cost.Output().Creator().MakeVector(1)
	one.AddScaler(c.MakeNumeric(1))
	cost.Propagate(one, grad)

	return grad
}

// A Batch is a single training batch, represented in a
// way that can be easily fed into a Model.
type Batch struct {
	Tweets [][]byte
	Avg    []int
	Out    *anydiff.Const
}
