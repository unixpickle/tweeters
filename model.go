package tweeters

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&Model{}).SerializerType(), DeserializeModel)
}

// A Model includes an encoder and a classifier.
//
// The Classifier's inputs should be twice as long as the
// encoder's outputs, that way it can take two latent
// vectors at a time (for comparison).
type Model struct {
	Encoder    anyrnn.Block
	Classifier anynet.Net
}

// NewModel creates a randomly-initialized model.
func NewModel(c anyvec.Creator, hidden int) *Model {
	return &Model{
		Encoder: anyrnn.Stack{
			anyrnn.NewLSTM(c, 0x100, hidden).ScaleInWeights(c.MakeNumeric(0x10)),
			anyrnn.NewLSTM(c, hidden, hidden).ScaleInWeights(c.MakeNumeric(2)),
			anyrnn.NewLSTM(c, hidden, hidden).ScaleInWeights(c.MakeNumeric(2)),
		},
		Classifier: anynet.Net{
			anynet.NewFC(c, hidden*2, 0x200),
			anynet.Tanh,
			anynet.NewFC(c, 0x200, 0x100),
			anynet.Tanh,
			anynet.NewFCZero(c, 0x100, 1),
		},
	}
}

// DeserializeModel deserializes a Model.
func DeserializeModel(d []byte) (*Model, error) {
	var res Model
	if err := serializer.DeserializeAny(d, &res.Encoder, &res.Classifier); err != nil {
		return nil, essentials.AddCtx("deserialize model", err)
	}
	return &res, nil
}

// Encode produces latent vectors for all of the tweets.
// The latent vectors are packed into a single result.
func (m *Model) Encode(tweets [][]byte) anydiff.Res {
	creator := m.creator()
	var batches []*anyseq.Batch
	var idx int
	for {
		var oneHot []float64
		var present []bool
		for _, tweet := range tweets {
			pres := idx < len(tweet)
			present = append(present, pres)
			if pres {
				oh := make([]float64, 0x100)
				oh[tweet[idx]] = 1
				oneHot = append(oneHot, oh...)
			}
		}
		if len(oneHot) == 0 {
			break
		}
		batches = append(batches, &anyseq.Batch{
			Present: present,
			Packed:  creator.MakeVectorData(creator.MakeNumericList(oneHot)),
		})
		idx++
	}
	constIn := anyseq.ConstSeq(creator, batches)
	return anyseq.Tail(anyrnn.Map(constIn, m.Encoder))
}

// Averages is like Encode, but it averages groups of
// latent vectors.
//
// The avgSizes slice specifies the size for each average.
// Averages are always taken over consecutive vectors.
// For example, if avgSlices is [1, 3, 2], then the first
// vector, an average of the next three vectors, and an
// average of the next two vectors are returned.
// The sum of all the average sizes should equal the total
// number of tweets.
func (m *Model) Averages(tweets [][]byte, avgSizes []int) anydiff.Res {
	latent := m.Encode(tweets)
	return anydiff.Pool(latent, func(latent anydiff.Res) anydiff.Res {
		latentSize := latent.Output().Len() / len(tweets)
		offset := 0
		var res []anydiff.Res
		for _, size := range avgSizes {
			subset := anydiff.Slice(latent, offset, offset+size*latentSize)
			offset += size * latentSize
			mat := &anydiff.Matrix{Data: subset, Rows: size, Cols: latentSize}
			sum := anydiff.SumRows(mat)
			divisor := sum.Output().Creator().MakeNumeric(1 / float64(size))
			res = append(res, anydiff.Scale(sum, divisor))
		}
		return anydiff.Concat(res...)
	})
}

// Parameters returns the model's parameters.
func (m *Model) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	for _, obj := range []interface{}{m.Encoder, m.Classifier} {
		if p, ok := obj.(anynet.Parameterizer); ok {
			res = append(res, p.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a Model with the serializer package.
func (m *Model) SerializerType() string {
	return "github.com/unixpickle/tweeters.Model"
}

// Serialize serializes the Model.
func (m *Model) Serialize() ([]byte, error) {
	return serializer.SerializeAny(m.Encoder, m.Classifier)
}

func (m *Model) creator() anyvec.Creator {
	return m.Parameters()[0].Vector.Creator()
}
