package tweetures

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"

	"github.com/unixpickle/essentials"
)

var dbByteOrder = binary.LittleEndian

// Record is a single tweet-username pair.
type Record struct {
	User []byte
	Body []byte
}

// WriteDB writes the entries in a format that ReadSamples
// can interpret.
func WriteDB(w io.Writer, records <-chan Record) (err error) {
	defer essentials.AddCtxTo("write DB", &err)
	bw := bufio.NewWriter(w)
	for record := range records {
		for _, data := range [][]byte{record.User, record.Body} {
			err := binary.Write(bw, dbByteOrder, int32(len(data)))
			if err != nil {
				return err
			}
			_, err = bw.Write(data)
			if err != nil {
				return err
			}
		}
	}
	return bw.Flush()
}

// ReadDB reads the entries from a database.
//
// Both channels are closed when reading is finished.
//
// The error channel is never sent an io.EOF.
//
// The records are guaranteed to arrive in order.
func ReadDB(r io.Reader) (<-chan Record, <-chan error) {
	records := make(chan Record, 1)
	errs := make(chan error, 1)

	byteRecords := make(chan []byte, 1)
	byteErr := make(chan error, 1)
	go func() {
		defer close(byteRecords)
		defer close(byteErr)
		br := bufio.NewReader(r)
		for {
			var size int32
			if err := binary.Read(br, dbByteOrder, &size); err == io.EOF {
				return
			} else if err != nil {
				byteErr <- err
				return
			}
			next := make([]byte, int(size))
			if _, err := io.ReadFull(br, next); err != nil {
				byteErr <- err
				return
			}
			byteRecords <- next
		}
	}()

	go func() {
		defer close(records)
		defer close(errs)
		for {
			user, ok := <-byteRecords
			if !ok {
				if err := <-byteErr; err != nil {
					errs <- essentials.AddCtx("read DB", err)
				}
				return
			}
			body, ok := <-byteRecords
			if !ok {
				if err := <-byteErr; err != nil {
					errs <- essentials.AddCtx("read DB", err)
				} else {
					errs <- essentials.AddCtx("read DB", errors.New("unexpected EOF"))
				}
				return
			}
			records <- Record{User: user, Body: body}
		}
	}()

	return records, errs
}
