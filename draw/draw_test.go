package draw

import "testing"

func TestCanvas(t *testing.T) {
	c := NewCanvas()
	c.DrawPoint(0, 0, false)
	c.DrawPoint(10, 10, true)
	c.DrawPoint(-30, -50, true)
	c.DrawLinearFunction(1, 0)
	c.Save()
}
