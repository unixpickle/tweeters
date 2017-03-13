package tweetures

import (
	"github.com/unixpickle/anydiff"
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
func NewModel(c anyvec.Creator) *Model {
	return &Model{
		Encoder: anyrnn.Stack{
			anyrnn.NewLSTM(c, 0x100, 0x200),
			anyrnn.NewLSTM(c, 0x200, 0x200),
			anyrnn.NewLSTM(c, 0x200, 0x200),
		},
		Classifier: anynet.Net{
			anynet.NewFC(c, 0x200*2, 0x100),
			anynet.Tanh,
			anynet.NewFC(c, 0x100, 0x80),
			anynet.Tanh,
			anynet.NewFC(c, 0x80, 1),
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
	return "github.com/unixpickle/tweetures.Model"
}

// Serialize serializes the Model.
func (m *Model) Serialize() ([]byte, error) {
	return serializer.SerializeAny(m.Encoder, m.Classifier)
}
