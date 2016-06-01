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

## Artificial neural networks as a model of the human brain

Have you ever wondered why there are tasks that are dead simple for any human but incredibly difficult for computers?
Artificial neural networks (short: ANN's) were inspired by the central nervous system of humans. Like their biological counterpart, ANN's are built upon simple signal processing elements that are connected together into a large mesh.

## What can neural networks do?

ANN's have been sucessfully applied to a number of problem domains:

* Classify data by recognizing patterns
* Detect anomalies or novelties, when test data does *not* match a learned pattern
* Process signals, for example, by filtering, sepataring, or compressing
* Approximate a target function--useful for predictions and forecasting

Agreed, this sounds a bit abstract, so let's look at some real-world applications.
Neural networks can -

* identify faces,
* recognize speech,
* read your handwriting (mine perhaps not),
* translate texts,
* play games (typically board games or card games)
* control autonomous vehicles and robots
* and surely a couple more things!


## The topology of a neural network

There are many ways of knitting the nodes of a neural network together, and each way results in a more or less complex behavior. Possibly the simplest of all topologies is the feed-forward network. Signals flow in one direction only; there is never any feedback loop in the system. The following image shows a very simple network:

![A feed-forward neural network](threelayernetwork.png)

The ANN is organised into layers. The input layer picks up the input signals and passes them on to the next layer, the so-called 'hidden' layer. (Actually, there may be more than one hidden layer in a neural network.) Last comes the output layer that delivers the result.



## Neural networks must learn

### Neurons: The building blocks of neural networks

#### Biology vs technology

#### Inside an artificial neuron

### The perceptron: things can get even simpler!


### Can a single perceptron achieve anything?

## Linearly classifiable data

##

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

var (
	// a and b specify the linear function that describes the separation line; see below for details.
	a, b int32
)

// The separation line is described as a linear function of the form `y = ax + b`.
func f(x int32) int32 {
	return a*x + b
}

// The Heaviside Step function[^heavi] returns zero if the input is negative
// and one if the input is zero or positive.
// This is our activation function for the perceptron.
func heaviside(f float32) int32 {
	if f < 0 {
		return 0
	}
	return 1
}

/*
### The perceptron

First we define the perceptron. A new perceptron uses random weights and biases that will be modified during the training process. The perceptron performs two tasks:

* Process input signals
* Adjust the input weights as instructed by the "trainer".

*/

// Our perceptron is a simple struct that holds the input weights and the bias.
type Perceptron struct {
	weights []float32
	bias    float32
}

// Create a new perceptron with n inputs. Weights and bias are initialized with random values
// between -1 and 1.
func NewPerceptron(n int32) *Perceptron {
	var i int32
	w := make([]float32, n, n)
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

// During the learning phase, the perceptron adjusts the weights and the bias based on how much the perceptron's answer differs from the correct answer.
func (p *Perceptron) Adjust(inputs []int32, delta int32, learningRate float32) {
	for i, input := range inputs {
		p.weights[i] += float32(input) * float32(delta) * learningRate
	}
	p.bias += float32(delta) * learningRate
}

/* ### The task the perceptron shall solve

Since a single perceptron can only classify data that is linearly separable[^linsep], we simply let it classify points in a two-dimensional space. That is, we define a separation line and train the perceptron to tell us whether a given point *(x,y)* is on one side of the line or on the other.
We rule out the case where the line would be vertical. This allows us to specify the line as a linear function equation:

    y = ax + b

Parameter `a` specifices how steep the line is, and `b` sets the offset. See these examples:

![Separation lines](separationlines.png)

Without knowing anything about our line, the perceptron must learn to distinguish between those points above the line and those points below. Hence we need to train the perceptron.

### Training functions
*/

// Function isAboveLine returns 1 if the point *(x,y)* is above the line *y = ax + b*, else 0. This is our teacher's solution manual.
func isAboveLine(point []int32, f func(int32) int32) int32 {
	x := point[0]
	y := point[1]
	if y > f(x) {
		return 1
	}
	return 0
}

// Function train is our teacher. The teacher generates random test points and feeds them to the perceptron. Then the teacher compares the answer against the solution from the 'solution manual'.
func train(p *Perceptron, iters int, rate float32) {

	for i := 0; i < iters; i++ {
		// Generate a random point between -100 and 100.
		point := []int32{
			rand.Int31n(201) - 101,
			rand.Int31n(201) - 101,
		}

		// Feed the point to the perceptron and evaluate the result.
		actual := p.Process(point)
		expected := isAboveLine(point, f)

		// Have the perceptron adjust its internal values accordingly.
		p.Adjust(point, expected-actual, rate)
	}
}

/*
### Showtime!

Now it is time to see how well the perceptron has learned the task. Again we throw random points
at it, but this time there is no feedback from the teacher. Will the perceptron classify every
point correctly?
*/

// This is our test function. It returns the number of correct answers.
func verify(p *Perceptron) int32 {
	var correctAnswers int32 = 0

	// Create a new drawing canvas. x and y range from -100 to 100.
	c := draw.NewCanvas()

	// Draw the separation line `y = a*x + b`.
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

	// Set up the line parameters.
	// a (the gradient of the line) can vary between -5 and 5,
	// and b (the offset) between -50 and 50.
	rand.Seed(time.Now().UnixNano())
	a = rand.Int31n(11) - 6
	b = rand.Int31n(101) - 51

	// Create a new perceptron with two inputs (one for x and one for y).
	p := NewPerceptron(2)

	// First, the perceptron needs to learn. The 'teacher' knows the right answer
	// to each training question and tells the perceptron how much its guess was off
	// the correct answer.
	// Second parameter: number of training iterations.
	// Third parameter: learning rate. Allowed range: 0 < learning rate <= 1.
	train(p, 1000, 0.1)

	// Now the perceptron is ready for testing.
	hits := verify(p)
	rate := float32(hits) / 10 // divide by 1000 (num of iterations) then multiply by 100 to get the percent value.
	fmt.Printf("%.2f%% of the answers were correct.\n", rate)
}

/*
Ensure to open `result.png` to see how the perceptron classified the points.

Run the code a few times to see if the accuracy of the results changes considerably.

## Exercises

1. Play with the number of training iterations!
   * Will the accuracy increase if you train the perceptron 10,000 times?
   * Try fewer iterations. What happens if you train the perceptron only 100 times? 10 times?
   * What happens if you skip the training completely?

2. Change the learning rate to 0.01, 0.2, 0.0001, 0.5, 1,... while keeping the training interations constant. Do you see the accuracy change?


## Further reading

[^ptron:] [Perceptrons](https://en.wikipedia.org/wiki/Perceptron)

[^ann:] [Artificial Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)

[^heavi:] [The Heaviside Step function](https://en.wikipedia.org/wiki/Heaviside_step_function)

[^nature:] [Chapter 10](http://natureofcode.com/book/chapter-10-neural-networks/) of the book "The Nature Of Code"

[^eleven:] [A neural network in 11 lines of Python](http://iamtrask.github.io/2015/07/12/basic-python-network/)

[^linsep:] [Linear separability](https://en.wikipedia.org/wiki/Linear_separability)

[^backprop:] [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
*/
