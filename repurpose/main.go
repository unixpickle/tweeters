// Command repurpose converts a tweetenc encoder into a
// model suitable for author-sameness classification.
package main

import (
	"flag"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/tweetenc"
	"github.com/unixpickle/tweeters"
)

func main() {
	var inPath string
	var outPath string
	flag.StringVar(&inPath, "in", "", "path to tweetenc encoder")
	flag.StringVar(&outPath, "out", "", "path to output classifier")
	flag.Parse()
	if inPath == "" || outPath == "" {
		essentials.Die("Required flags: -in and -out. See -help.")
	}
	var enc *tweetenc.Encoder
	if err := serializer.LoadAny(inPath, &enc); err != nil {
		essentials.Die(err)
	}
	meanEnc := enc.MeanEncoder.(anynet.Net)[0].(*anynet.FC)
	outSize := meanEnc.InCount
	c := anyvec32.CurrentCreator()
	model := &tweeters.Model{
		Encoder: enc.Block,
		Classifier: anynet.Net{
			anynet.NewFC(c, outSize*2, 0x200),
			anynet.Tanh,
			anynet.NewFC(c, 0x200, 0x100),
			anynet.Tanh,
			anynet.NewFC(c, 0x100, 1),
		},
	}
	if err := serializer.SaveAny(outPath, model); err != nil {
		essentials.Die(err)
	}
}
