let canvas_quad = 1000
let char_quad = 60
let gate = 90

let sampler
let bepoints
let p5bezier
let todo = []

class Sampler {
  constructor (sampleInterval = 300) {
    // Setup the sound inputs
    this.mic = new p5.AudioIn()
    this.fft = new p5.FFT()
    this.fft.setInput(this.mic)

    this.sampleInterval = sampleInterval
    this.lastSampleTime = 0

    this.loudness = 0
    this.centroid = 0

    this.active = false

    this.mic.start()
  }

  // Sample if the time has come
  sample () {
    let currentTime = millis()

    if (currentTime - this.lastSampleTime >= this.sampleInterval) {
      this.lastSampleTime = currentTime

      let spec = this.fft.analyze()

      this.loudness = this.fft.getEnergy('lowMid', 'highMid')
      this.centroid = this.fft.getCentroid()

      if (this.loudness > gate) {
        this.active = true
      }

      // drawSpectrum(spec)
    }

    return {
      loudness: this.loudness,
      centroid: this.centroid
    }
  }

  active () {
    return this.active
  }
}

class BePoints {
  constructor () {
    this.points = []
    this.interpolatedPoints = []
    this.index = 0
    this.size = 200

    this.x = 0
    this.y = canvas_quad / 2
  }

  addBlocks (coordinates) {
    // Points are [0~60, 0~60] squares
    for (let i = 0; i < coordinates.length; i++) {
      let [x, y] = coordinates[i]
      x = this.x + (x / char_quad) * this.size
      y = this.y + (y / char_quad - 0.5) * this.size
      // console.log('add points: ', x, y)
      this.points.push(createVector(x, y))
    }

    this.x += this.size
  }

  interpolatePoints () {
    if (this.points.length < 2) return

    for (let i = 0; i < this.points.length - 1; i++) {
      let start = this.points[i]
      let end = this.points[i + 1]

      let numNewPoints = 4
      for (let j = 0; j <= numNewPoints; j++) {
        let t = j / (numNewPoints + 1)
        let interpolatedPoint = p5.Vector.lerp(start, end, t)
        // console.log('interpolated points: ', interpolatedPoint)
        this.interpolatedPoints.push(interpolatedPoint)
      }
    }

    // console.log(this.interpolatedPoints)
  }

  pop (loudness, centroid) {
    let skip = floor(map(centroid, 1000, 8000, 12, 1))
    let scale = map(loudness, 0, 250, 0.8, 1.2)
    // console.log(skip, scale)

    skip = skip < 1 ? 1 : skip
    if (this.index + skip >= this.interpolatedPoints.length) {
      return null
    }

    let p = this.interpolatedPoints[this.index]
    this.index += skip

    // Processing points
    p.mult(scale)

    return p
  }
}

function setup () {
  // Setup the canvas.
  const c = createCanvas(canvas_quad, canvas_quad)
  background(0)

  sampler = new Sampler(100)

  bepoints = new BePoints()
  bepoints.addBlocks([
    [0.0 * 60, 30],
    [0.1 * 60, 30],
    [0.2 * 60, 30],
    [0.3 * 60, 30],
    [0.4 * 60, 30],
    [0.5 * 60, 30],
    [0.6 * 60, 30],
    [0.7 * 60, 30],
    [0.8 * 60, 30],
    [0.9 * 60, 30],
    [1.0 * 60, 30]
  ])
  bepoints.addBlocks(data.p)
  bepoints.addBlocks([
    [0.0 * 60, 30],
    [0.1 * 60, 30],
    [0.2 * 60, 30],
    [0.3 * 60, 30],
    [0.4 * 60, 30],
    [0.5 * 60, 30],
    [0.6 * 60, 30],
    [0.7 * 60, 30],
    [0.8 * 60, 30],
    [0.9 * 60, 30],
    [1.0 * 60, 30]
  ])
  bepoints.interpolatePoints()

  p5bezier = initBezier(c)
}

function draw () {
  let s = sampler.sample()
  // console.log(s)

  if (sampler.active) {
    // Ok, here we do our jobs
    let p = bepoints.pop(s.loudness, s.centroid)

    if (p) {
      background(0, 40)
      todo.push([p.x, p.y])
  
      push()
      noFill()
      stroke(255)
      strokeWeight(3)
      if (todo.length >= 2) p5bezier.draw(todo)
      pop()
  
      // noStroke()
      // fill(255, 0, 0)
      // ellipse(p.x, p.y, 10, 10)
    }
  }
}

function drawSpectrum (spectrum) {
  background(0)
  stroke(255)
  noFill()
  beginShape()
  for (let i = 0; i < spectrum.length; i++) {
    let x = map(i, 0, spectrum.length, 0, width)
    let y = map(spectrum[i], 0, 255, height, 0)
    vertex(x, y)
  }
  endShape()
}

// Store our datas
data = {
  p: [
    [0, 30],
    [2.96, 29.896],
    [5.919, 29.815],
    [8.879, 29.543],
    [11.704, 29.143],
    [13.662, 28.722],
    [15.432, 28.201],
    [17.622, 27.53],
    [19.798, 26.634],
    [22.018, 25.621],
    [24.488, 24.278],
    [26.319, 23.305],
    [28.067, 22.284],
    [30.019, 21.264],
    [31.815, 20.006],
    [33, 18.935],
    [34.343, 17.375],
    [35.447, 14.943],
    [36.118, 12.749],
    [36.219, 10.644],
    [35.852, 8.338],
    [35.18, 6.902],
    [34.343, 5.75],
    [32.886, 4.549],
    [31.543, 3.878],
    [29.544, 3.783],
    [27.682, 4.381],
    [26.451, 5.078],
    [25.229, 6.231],
    [24.111, 7.433],
    [22.677, 9.205],
    [21.642, 11.092],
    [20.971, 12.979],
    [20.676, 14.943],
    [20.801, 18.356],
    [21.248, 20.418],
    [21.919, 22.504],
    [23.012, 25.073],
    [24.019, 27.724],
    [24.69, 30.214],
    [25.362, 32.143],
    [25.745, 33.84],
    [26.533, 36.008],
    [27.204, 38.011],
    [28.067, 40.326],
    [28.738, 42.641],
    [29.61, 45.244],
    [30.521, 47.337],
    [31.192, 49.769],
    [31.864, 51.595],
    [32.23, 54.458],
    [31.864, 56.151],
    [30.229, 56.612],
    [28.939, 55.13],
    [28.546, 52.938],
    [28.428, 50.823],
    [28.387, 48.78],
    [28.428, 46.173],
    [30.282, 40.998],
    [31.948, 39.138],
    [33.673, 37.594],
    [35.702, 36.008],
    [37.733, 34.88],
    [39.502, 34.001],
    [41.708, 33.115],
    [43.593, 32.468],
    [45.134, 31.914],
    [46.862, 31.627],
    [48.86, 31.121],
    [50.018, 30.929],
    [52.204, 30.567],
    [54.716, 30.214],
    [56.85, 29.973],
    [59.977, 30]
  ]
}
