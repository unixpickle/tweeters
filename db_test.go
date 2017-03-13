package tweeters

import (
	"bytes"
	"reflect"
	"testing"
)

func TestDB(t *testing.T) {
	var buf bytes.Buffer
	inRecords := []Record{
		Record{User: []byte("unixpickle"), Body: []byte("This is a tweet.")},
		Record{User: []byte("bob"), Body: []byte("")},
		Record{User: []byte(""), Body: []byte("Tweet, this doth be.")},
		Record{User: []byte("unixpickle"), Body: []byte("This is another tweet.")},
	}
	inChan := make(chan Record, len(inRecords))
	for _, r := range inRecords {
		inChan <- r
	}
	close(inChan)
	if err := WriteDB(&buf, inChan); err != nil {
		t.Fatal(err)
	}
	records, errChan := ReadDB(&buf)
	var outRecords []Record
	for r := range records {
		outRecords = append(outRecords, r)
	}
	if err := <-errChan; err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(inRecords, outRecords) {
		t.Errorf("expected %#v but got %#v", inRecords, outRecords)
	}
}
