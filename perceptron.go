/*
<!--
Copyright (c) 2016 Christoph Berger. Some rights reserved.
Use of this text is governed by a Creative Commons Attribution Non-Commercial
Share-Alike License that can be found in the LICENSE.txt file.

The source code contained in this file may import third-party source code
whose licenses are provided in the respective license files.
-->

+++
title = "Perceptrons - the most basic form of a neural network"
description = "A Go implementation of a perceptron as the building block of neural networks and as the most basic form of pattern recognition and machine learning."
author = "Christoph Berger"
email = "chris@appliedgo.com"
date = "2016-06-09"
publishdate = "2016-06-09"
domains = ["Artificial Intelligence"]
tags = ["Pattern Recognition", "Neural Network", "Machine Learning"]
categories = ["Tutorial"]
+++

Artificial Neural Networks have gained attention during the recent years, driven by advances in deep learning. But what is an Artificial Neural Network and what is it made of? Meet the perceptron.

<!--more-->

<!-- TODO -->


## The code: A perceptron for classifying points
*/

// ### Imports and globals
package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/appliedgo/perceptron/draw"
)

// The learing rate adjusts the speed and quality of learning. Learing will be faster with higher values and more accurate with lower values.[^7]
const (
	learningRate = 0.1 // Allowed range: 0 < learningRate <= 1
)

// a and b specify the linear function that describes the separation line; see below for details.
var (
	a, b int32
)

// The separation line is described as a linear function of the form `y = ax + b`.
func f(x int32) int32 {
	return a*x + b
}

// The Heaviside Step function[^3] returns zero if the input is negative
// and one if the input is zero or positive.
// This is our activation function for the perceptron.
func heaviside(f float32) int32 {
	if f < 0 {
		return 0
	}
	return 1
}

// ### The perceptron
//
// First we define the perceptron. A new perceptron uses random weights and biases that will be modified during the training process. The perceptron performs two tasks:
//
// * Process input signals
// * Adjust the input weights as instructed by the "trainer".

// Our perceptron is a simple struct that holds the input weights and the bias.
type Perceptron struct {
	weights []float32
	bias    float32
}

// Create a new perceptron. Weights and bias are initialized with random values
// between -1 and 1.
func NewPerceptron(ni int32) *Perceptron {
	var i int32
	w := make([]float32, ni, ni)
	for i = 0; i < ni; i++ {
		w[i] = rand.Float32()*2 - 1
	}
	return &Perceptron{
		weights: w,
		bias:    rand.Float32()*2 - 1,
	}
}

// The basic task of a perceptron is to process the input signals and generate an output signal.
// The activation function is a simple yes/no decision, so the output will be either 0 or 1.
func (p *Perceptron) Process(inputs []int32) int32 {
	sum := p.bias
	for i, input := range inputs {
		sum += float32(input) * p.weights[i]
	}
	return heaviside(sum)
}

// Adjust the weights and the bias according to the difference between expected and observed result.
func (p *Perceptron) Adjust(inputs []int32, delta int32) {
	for i, input := range inputs {
		p.weights[i] += float32(input) * float32(delta) * learningRate
	}
	p.bias += float32(delta) * learningRate
}

/* ### The task the perceptron shall solve

Since a single perceptron can only classify data that is linearly separable[^6], we simply let it classify points in a two-dimensional space. That is, we define a separation line and train the perceptron to tell us whether a given point *(x,y)* is on one side of the line or on the other.
We rule out the case where the line would be vertical. This allows us to specify the line as a linear function equation:

    y = ax + b

`a` specifices how steep the line is, and `b` sets the offset. See these examples:

![Separation lines](separationlines.png)

Without knowing anything about our line, the perceptron must learn to distinguish between those points above the line and those points below. Hence we need to train the perceptron.

### Training functions
*/

// isAboveLine returns 1 if the point *(x,y)* is above the line *y = ax + b*, else 0.
// f is the generated
func isAboveLine(point []int32, f func(int32) int32) int32 {
	x := point[0]
	y := point[1]
	if y > f(x) {
		return 1
	}
	return 0
}

// To train the perceptron, we generate random points
func train(p *Perceptron) {

	for i := 0; i < 1000; i++ {
		// Generate a random point between -100 and 100.
		point := []int32{
			rand.Int31n(201) - 101,
			rand.Int31n(201) - 101,
		}

		// Feed the point to the perceptron and evaluate the result.
		actual := p.Process(point)
		expected := isAboveLine(point, f)

		// Have the perceptron adjust its internal values accordingly.
		p.Adjust(point, expected-actual)
	}
}

/*
### Showtime!

Now it is time to see how well the perceptron has learned the task. Again we throw random points
at it, but this time there is no feedback from the trainer. Will the perceptron classify every
point correctly?
*/

// This is our test function. It returns the number of correct answers.
func verify(p *Perceptron) int32 {
	var correctAnswers int32 = 0

	// Create a new drawing canvas.
	c := draw.NewCanvas()
	c.DrawLinearFunction(a, b)

	for i := 0; i < 1000; i++ {
		// Generate a random point between -100 and 100.
		point := []int32{
			rand.Int31n(201) - 101,
			rand.Int31n(201) - 101,
		}

		// Feed the point to the perceptron and evaluate the result.
		result := p.Process(point)
		c.DrawPoint(point[0], point[1], result == 1)
		if result == isAboveLine(point, f) {
			correctAnswers += 1
		} else {
		}
	}
	// Save the image as `./result.png`.
	c.Save()

	return correctAnswers
}

// Main: Set up, train, and test the perceptron.
func main() {

	// Setup.
	rand.Seed(time.Now().UnixNano())
	a = rand.Int31n(11) - 6
	b = rand.Int31n(51) - 26

	// Create a new perceptron with two inputs (one for x and one for y).
	p := NewPerceptron(2)

	// TODO We first need to train the perceptron. The "trainer" knows the right answers
	// to the training questions and tells the perceptron how much its guess was off
	// the correct answer.
	train(p)

	// Now the perceptron is ready for testing.
	rate := float32(verify(p)) / 10
	fmt.Printf("%.2f%% of the answers were correct.\n", rate)
}

/*
Ensure to open `result.png` to see how the perceptron classified the points.

## Further reading

[^1:] [Perceptrons](https://en.wikipedia.org/wiki/Perceptron)

[^2:] [Artificial Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)

[^3:] [The Heaviside Step function](https://en.wikipedia.org/wiki/Heaviside_step_function)

[^4:] [Chapter 10](http://natureofcode.com/book/chapter-10-neural-networks/) of the book "The Nature Of Code"

[^5:] [A neural network in 11 lines of Python](http://iamtrask.github.io/2015/07/12/basic-python-network/)

[^6:] [Linear separability](https://en.wikipedia.org/wiki/Linear_separability)

[^7:] [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
*/
