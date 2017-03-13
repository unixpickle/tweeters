package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/tweetures"
)

const Separator = "---------------"

var QuestionDelay = time.Second

func main() {
	var dataPath string
	var minTweets, maxTweets int
	flag.StringVar(&dataPath, "data", "", "path to tweet DB")
	flag.IntVar(&minTweets, "min", 3, "minimum tweets for one user")
	flag.IntVar(&maxTweets, "max", 15, "maximum tweets for one user")
	flag.Parse()
	if dataPath == "" {
		essentials.Die("Required flag: -data. See -help.")
	}

	fmt.Println("Loading samples...")
	data, err := tweetures.ReadSamples(dataPath)
	if err != nil {
		essentials.Die(err)
	}
	fmt.Println()
	fmt.Println(Separator)
	fmt.Println()

	rand.Seed(time.Now().UnixNano())

	br := bufio.NewReader(os.Stdin)

	var total, correct int
	for {
		tweets := data.RandomUserTweets(minTweets, maxTweets)
		same := rand.Intn(2) == 0
		if !same {
			tweets[len(tweets)-1] = data.RandomUserTweets(1, 1)[0]
		}
		for _, tweet := range tweets[:len(tweets)-1] {
			fmt.Println(string(tweet))
			fmt.Println()
		}
		fmt.Println(Separator)
		fmt.Println()
		fmt.Println(string(tweets[len(tweets)-1]))
		fmt.Println()
		var answer bool
		for {
			fmt.Print("Is the last tweet by the same author? [y/n]: ")
			line, err := br.ReadString('\n')
			if err != nil {
				os.Exit(1)
			}

			if line[0] != 'y' && line[0] != 'n' {
				fmt.Println("Enter 'y' or 'n'")
				continue
			}
			answer = line[0] == 'y'
			break
		}
		if answer == same {
			fmt.Println("Correct!")
			correct++
		} else {
			fmt.Println("Incorrect!")
		}
		total++
		fmt.Printf("Score: %d/%d (%.3f)", correct, total, float64(correct)/float64(total))
		fmt.Println()

		time.Sleep(QuestionDelay)

		fmt.Println(Separator)
		fmt.Println()
	}
}
