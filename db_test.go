package tweeters

import (
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"testing"
)

func TestDB(t *testing.T) {
	inRecords := []Record{
		Record{User: []byte("unixpickle"), Body: []byte("This is a tweet.")},
		Record{User: []byte("unixpickle"), Body: []byte("This is another tweet.")},
		Record{User: []byte("bob"), Body: []byte("")},
		Record{User: []byte("bob"), Body: []byte("hey")},
		Record{User: []byte("bob"), Body: []byte("test tweet")},
		Record{User: []byte(""), Body: []byte("Tweet, this doth be.")},
		Record{User: []byte("joe"), Body: []byte("Tweet, this dothn't be.")},
	}
	inChan := make(chan Record, len(inRecords))
	for _, r := range inRecords {
		inChan <- r
	}
	close(inChan)

	f, err := ioutil.TempFile("", "dbtest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	if err := WriteDB(f, inChan); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}

	db := &DB{file: f}
	if err := db.buildIndex(); err != nil {
		t.Fatal(err)
	}

	indices := []int{1, 2, 0, 3}
	expected := [][]Record{
		{
			Record{User: []byte("bob"), Body: []byte("")},
			Record{User: []byte("bob"), Body: []byte("hey")},
			Record{User: []byte("bob"), Body: []byte("test tweet")},
		},
		{
			Record{User: []byte(""), Body: []byte("Tweet, this doth be.")},
		},
		{
			Record{User: []byte("unixpickle"), Body: []byte("This is a tweet.")},
			Record{User: []byte("unixpickle"), Body: []byte("This is another tweet.")},
		},
		{
			Record{User: []byte("joe"), Body: []byte("Tweet, this dothn't be.")},
		},
	}

	for i, index := range indices {
		actualRecords, err := db.Read(index)
		if err != nil {
			t.Fatal(err)
		}
		expectedRecords := expected[i]
		if !reflect.DeepEqual(actualRecords, expectedRecords) {
			t.Errorf("user %d: expected %#v but got %#v", index, expectedRecords, actualRecords)
		}
	}
}
