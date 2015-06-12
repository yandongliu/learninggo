package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

var DEBUG = true
var logger = log.New(os.Stderr, "logger: ", log.Lshortfile)

type Rating struct {
	uid    int
	iid    int
	rating float64
}

type Matrix struct {
	ratings  []Rating
	rank     int
	P        [][]float64
	Q        [][]float64
	numUsers int
	numItems int
}

func (m *Matrix) init(rank int) {
	m.ratings = make([]Rating, 0)
	m.rank = rank
}

func check(msg string, e error) {
	if e != nil {
		fmt.Println(msg)
		panic(e)
	}
}

func innerproduct(p []float64, q []float64) float64 {
	n := len(p)
	if n != len(q) {
		panic("Length are different!")
	}
	prdt := 0.0
	for i := 0; i < n; i++ {
		prdt += p[i] * q[i]
	}
	return prdt
}

func printArray(p []float64) {
	for _, v := range p {
		fmt.Print(v)
	}
	for i := 0; i < len(p); i++ {
		fmt.Print(p[i], " ")
	}
	fmt.Println()
}

func NMF_sgd(m *Matrix, numIters int) {
	var gamma = 0.00005
	var lambda = 0.0000001

	for iter := 0; iter < numIters; iter++ {
		rmse := 0.0
		for _, r := range m.ratings {
			p := m.P[r.uid]
			q := m.Q[r.iid]
			predict := innerproduct(p, q)
			diff := r.rating - predict
			rmse += diff * diff
			for j := 0; j < m.rank; j++ {
				q[j] += (gamma * (p[j]*diff - lambda*q[j]))
				p[j] += (gamma * (q[j]*diff - lambda*p[j]))
			}
		}
		logger.Println(fmt.Sprintf("rmse: %f", rmse))
	}
}

func readMovieLensFile(fn string, rank int) *Matrix {
	logger.Println("reading file:" + fn)
	splitter := "::"
	fp, err := os.Open(fn)
	check("Read file", err)
	defer fp.Close()

	var m = Matrix{}
	m.init(rank)
	var user2id = make(map[string]int)
	var item2id = make(map[string]int)
	var numUser = 0
	var numItem = 0
	var ok bool

	scanner := bufio.NewScanner(fp)
	for scanner.Scan() {
		row := scanner.Text()
		flds := strings.Split(row, splitter)
		if len(flds) != 4 {
			continue
		}
		user := flds[0]
		item := flds[1]
		r, err := strconv.ParseFloat(flds[2], 64)
		check("Atof", err)
		_, ok = user2id[user]
		if !ok {
			user2id[user] = numUser
			numUser++
		}
		_, ok = item2id[item]
		if !ok {
			item2id[item] = numItem
			numItem++
		}
		rating := Rating{user2id[user], item2id[item], r}
		m.ratings = append(m.ratings, rating)
	}
	m.numUsers = numUser
	m.numItems = numItem
	logger.Println(fmt.Sprintf("#rank %d, #users: %d, #items: %d\n", m.rank, numUser, numItem))
	m.P = make([][]float64, numUser)
	m.Q = make([][]float64, numItem)
	for i := 0; i < numUser; i++ {
		m.P[i] = make([]float64, m.rank)
		for j := 0; j < m.rank; j++ {
			m.P[i][j] = 0.1
		}
	}
	for i := 0; i < numItem; i++ {
		m.Q[i] = make([]float64, m.rank)
		for j := 0; j < m.rank; j++ {
			m.Q[i][j] = 0.1
		}
	}
	return &m
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: data_file")
	} else {
		m := readMovieLensFile(os.Args[1], 20)
		NMF_sgd(m, 50)
	}
}
