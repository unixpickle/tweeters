package tweeters

import (
	"bufio"
	"encoding/binary"
	"io"
	"os"

	"github.com/unixpickle/essentials"
)

var dbByteOrder = binary.LittleEndian

// Record is a single tweet-username pair.
type Record struct {
	User []byte
	Body []byte
}

// WriteDB writes the records to the database.
//
// The records should be grouped by username.
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

// DB is a read-only handle to a database.
type DB struct {
	userOffsets []int
	file        *os.File
	bufReader   *bufio.Reader
}

// OpenDB opens a database and builds an index for it.
func OpenDB(path string) (db *DB, err error) {
	defer essentials.AddCtxTo("open DB", &err)
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()

	db = &DB{file: f, bufReader: bufio.NewReader(f)}
	if err := db.buildIndex(); err != nil {
		return nil, err
	}
	return db, nil
}

// NumUsers returns the number of users in the database.
func (d *DB) NumUsers() int {
	return len(d.userOffsets)
}

// Read reads the records for a user, which is identified
// by index.
func (d *DB) Read(userIdx int) (records []Record, err error) {
	defer essentials.AddCtxTo("read DB record", &err)
	off := d.userOffsets[userIdx]
	if _, err := d.file.Seek(int64(off), io.SeekStart); err != nil {
		return nil, err
	}
	d.bufReader.Reset(d.file)
	var lastUser string
	for {
		username, err := d.readField()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		if len(records) == 0 {
			lastUser = string(username)
		} else if string(username) != lastUser {
			break
		}
		body, err := d.readField()
		if err != nil {
			return nil, err
		}
		records = append(records, Record{User: username, Body: body})
	}
	return
}

// Close closes the database handle.
func (d *DB) Close() error {
	return d.file.Close()
}

func (d *DB) buildIndex() error {
	var lastUsername string
	var offset int
	for {
		username, err := d.readField()
		if err == io.EOF {
			break
		} else if err != nil {
			return err
		}
		if string(username) != lastUsername {
			lastUsername = string(username)
			d.userOffsets = append(d.userOffsets, offset)
		}
		offset += len(username) + 4
		n, err := d.skipField()
		if err != nil {
			return err
		}
		offset += n
	}
	return nil
}

func (d *DB) readField() ([]byte, error) {
	var size int32
	if err := binary.Read(d.bufReader, dbByteOrder, &size); err != nil {
		return nil, err
	}
	next := make([]byte, int(size))
	if _, err := io.ReadFull(d.bufReader, next); err != nil {
		return nil, err
	}
	return next, nil
}

func (d *DB) skipField() (int, error) {
	var size int32
	if err := binary.Read(d.bufReader, dbByteOrder, &size); err != nil {
		return 0, err
	}
	if _, err := d.bufReader.Discard(int(size)); err != nil {
		return 0, err
	}
	return int(size) + 4, nil
}
