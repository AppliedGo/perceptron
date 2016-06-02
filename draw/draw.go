package draw

import (
	"log"

	"github.com/fogleman/gg"
)

type Canvas struct {
	dc *gg.Context
}

func NewCanvas() Canvas {
	// Create a context that spans -100 to 100 in both directions.
	dc := gg.NewContext(201, 201)
	dc.Translate(100, -100)

	// Let y increase from bottom to top
	dc.InvertY()

	// Set white background and clear the canvas
	dc.SetRGB(1, 1, 1)
	dc.Clear()

	return Canvas{dc}
}

func (c *Canvas) DrawLinearFunction(ai, bi int32) {
	a := float64(ai)
	b := float64(bi)
	c.dc.SetRGBA(0, 0, 0, 0.7)
	c.dc.SetLineWidth(3)
	c.dc.DrawLine(-100, a*(-100)+b, 100, a*100+b)
	c.dc.Stroke()
}

func (c *Canvas) DrawPoint(xi, yi int32, above bool) {
	x := float64(xi)
	y := float64(yi)
	c.dc.SetRGB(0, 0, 0)
	c.dc.SetLineWidth(1)
	c.dc.DrawCircle(x, y, 3)
	if above {
		c.dc.SetRGB(1, 0.7, 0.5)
	} else {
		c.dc.SetRGB(0, 0, 0.7)
	}
	c.dc.Fill()
	c.dc.Stroke()
}

func (c *Canvas) Save() {
	c.dc.Stroke()
	err := c.dc.SavePNG("./result.png")
	if err != nil {
		log.Fatal("Unable to save the image.", err.Error())
	}
}
